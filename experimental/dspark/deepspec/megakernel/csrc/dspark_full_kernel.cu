#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>
#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace cg = cooperative_groups;
constexpr int kThreads = 256;
constexpr int kWeightsPerLayer = 11;
constexpr int kFullTraceColumns = 3;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr int kAttentionWarpsPerVector = 4;
constexpr int kAttentionVectorsPerBlock = kWarpsPerBlock / kAttentionWarpsPerVector;
constexpr int kFastAttentionHeadDim = 128;
constexpr int kFastAttentionMaxKeys = 32;
constexpr int kMarkovRank = 256;
constexpr int kMarkovTileColumns = 32;
constexpr int kMarkovTileRows = 32;
constexpr int kMarkovTileStride = kMarkovTileRows + 1;
constexpr int kLmTcgenRows = 8;
constexpr int kLmTcgenColumns = 128;
constexpr int kLmTcgenReductionTile = 64;
constexpr int kLmTcgenStages = 8;
constexpr int kLmTcgenAccumulatorStages = 2;
constexpr int kLmTcgenDynamicSharedBytes = 139520;

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

#ifdef DSPARK_ENABLE_DEVICE_TRACE
__device__ __forceinline__ void record_full_timestamp(
    int64_t* trace,
    int phase,
    int column) {
  if (threadIdx.x == 0) {
    trace[(static_cast<int64_t>(blockIdx.x) * 160 + phase) * kFullTraceColumns + column] =
        static_cast<int64_t>(globaltimer_ns());
  }
}
#else
__device__ __forceinline__ void record_full_timestamp(int64_t*, int, int) {}
#endif

template <typename scalar_t>
struct LayerWeights {
  const scalar_t* input_norm;
  const scalar_t* q;
  const scalar_t* k;
  const scalar_t* v;
  const scalar_t* o;
  const scalar_t* q_norm;
  const scalar_t* k_norm;
  const scalar_t* post_norm;
  const scalar_t* gate;
  const scalar_t* up;
  const scalar_t* down;
};

template <typename scalar_t>
struct LinearTask {
  const scalar_t* input;
  const scalar_t* weight;
  scalar_t* output;
  int rows;
  int input_width;
  int output_width;
};

template <typename scalar_t>
struct FullWeights {
  const scalar_t* embedding;
  const scalar_t* context_projection;
  const scalar_t* context_norm;
  const scalar_t* final_norm;
  const scalar_t* lm_head;
  const scalar_t* markov_w1;
  const scalar_t* markov_w2;
  const scalar_t* confidence_weight;
  const scalar_t* confidence_bias;
  LayerWeights<scalar_t> layers[5];
};

template <typename scalar_t>
struct FullWorkspace {
  scalar_t* context;
  scalar_t* hidden;
  scalar_t* normalized;
  scalar_t* q;
  scalar_t* k_context;
  scalar_t* v_context;
  scalar_t* k_draft;
  scalar_t* v_draft;
  scalar_t* attention;
  scalar_t* projection;
  scalar_t* gate;
  scalar_t* up;
  scalar_t* base_logits;
  float* corrected_logits;
  float* probabilities;
  int64_t* token_ids;
  float* confidence_logits;
  scalar_t* cache_k;
  scalar_t* cache_v;
  float* block_max;
  int* block_index;
  float* block_sum;
  float* global_values;
  scalar_t* target_padded;
  int64_t* trace;
  float* linear_partials;
  int64_t* verify_input_ids;
};

template <typename scalar_t>
__device__ __forceinline__ float as_float(const scalar_t* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

using LmTcgenType = cutlass::bfloat16_t;
using LmTcgenTiledMma = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_SS<
        LmTcgenType,
        LmTcgenType,
        float,
        kLmTcgenColumns,
        kLmTcgenRows,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>{}));
using LmTcgenMmaTiler = decltype(cute::make_shape(
    cute::Int<kLmTcgenColumns>{},
    cute::Int<kLmTcgenRows>{},
    cute::Int<kLmTcgenReductionTile>{}));
using LmTcgenMmaShapeA = decltype(cute::partition_shape_A(
    LmTcgenTiledMma{},
    cute::make_shape(
        cute::Int<kLmTcgenColumns>{},
        cute::Int<kLmTcgenReductionTile>{},
        cute::Int<kLmTcgenStages>{})));
using LmTcgenMmaShapeB = decltype(cute::partition_shape_B(
    LmTcgenTiledMma{},
    cute::make_shape(
        cute::Int<kLmTcgenRows>{},
        cute::Int<kLmTcgenReductionTile>{},
        cute::Int<kLmTcgenStages>{})));
using LmTcgenSmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<LmTcgenType>{},
    LmTcgenMmaShapeA{}));
using LmTcgenSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<LmTcgenType>{},
    LmTcgenMmaShapeB{}));
using LmTcgenWeightLayout = decltype(cute::make_layout(
    cute::make_shape(cute::Int<151936>{}, cute::Int<2560>{}),
    cute::make_stride(cute::Int<2560>{}, cute::Int<1>{})));
using LmTcgenInputLayout = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kLmTcgenRows>{}, cute::Int<2560>{}),
    cute::make_stride(cute::Int<2560>{}, cute::Int<1>{})));
using LmTcgenWeightTensor = decltype(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<const LmTcgenType*>(nullptr)),
    LmTcgenWeightLayout{}));
using LmTcgenInputTensor = decltype(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<const LmTcgenType*>(nullptr)),
    LmTcgenInputLayout{}));
using LmTcgenTmaAtomA = decltype(cute::make_tma_atom(
    cute::SM90_TMA_LOAD{},
    std::declval<LmTcgenWeightTensor>(),
    LmTcgenSmemLayoutA{}(
        cute::_, cute::_, cute::_, cute::Int<0>{}),
    cute::select<0, 2>(LmTcgenMmaTiler{})));
using LmTcgenTmaAtomB = decltype(cute::make_tma_atom(
    cute::SM90_TMA_LOAD{},
    std::declval<LmTcgenInputTensor>(),
    LmTcgenSmemLayoutB{}(
        cute::_, cute::_, cute::_, cute::Int<0>{}),
    cute::select<1, 2>(LmTcgenMmaTiler{})));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
// Adapted from NVIDIA CUTLASS's public SM100 warp-specialized collective and
// Mirage's persistent Blackwell linear task. This remains a device function in
// the one persistent DSpark launch; it never launches a nested/library kernel.
struct LmTcgenSharedStorage {
  alignas(128)
      cute::ArrayEngine<LmTcgenType, cute::cosize_v<LmTcgenSmemLayoutA>> a;
  alignas(128)
      cute::ArrayEngine<LmTcgenType, cute::cosize_v<LmTcgenSmemLayoutB>> b;
  alignas(16) cute::uint64_t ab_full[kLmTcgenStages];
  alignas(16) cute::uint64_t ab_empty[kLmTcgenStages];
  alignas(16) cute::uint64_t accumulator_full[kLmTcgenAccumulatorStages];
  alignas(16) cute::uint64_t accumulator_empty[kLmTcgenAccumulatorStages];
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a() {
    return cute::make_tensor(
        cute::make_smem_ptr(a.begin()), LmTcgenSmemLayoutA{});
  }

  CUTE_DEVICE constexpr auto tensor_b() {
    return cute::make_tensor(
        cute::make_smem_ptr(b.begin()), LmTcgenSmemLayoutB{});
  }
};

template <typename scalar_t>
__device__ void linear_b200_tcgen(
    const scalar_t* input,
    scalar_t* output,
    int rows,
    int input_width,
    int output_width,
    const LmTcgenTmaAtomA& tma_atom_a,
    const LmTcgenTmaAtomB& tma_atom_b) {
  using namespace cute;
  static_assert(sizeof(LmTcgenSharedStorage) <= kLmTcgenDynamicSharedBytes);
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  LmTcgenSharedStorage& storage =
      *reinterpret_cast<LmTcgenSharedStorage*>(dynamic_shared_memory);

  LmTcgenTiledMma tiled_mma;
  LmTcgenMmaTiler mma_tiler;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto shared_a = storage.tensor_a();
  auto shared_b = storage.tensor_b();
  auto fragment_a = cta_mma.make_fragment_A(shared_a);
  auto fragment_b = cta_mma.make_fragment_B(shared_b);

  auto tma_weight = tma_atom_a.get_tma_tensor(
      make_shape(Int<151936>{}, Int<2560>{}));
  auto tma_input = tma_atom_b.get_tma_tensor(
      make_shape(Int<kLmTcgenRows>{}, Int<2560>{}));
  auto global_a = local_tile(
      tma_weight, mma_tiler, make_coord(_, _, _), Step<_1, X, _1>{});
  auto global_b = local_tile(
      tma_input, mma_tiler, make_coord(_, _, _), Step<X, _1, _1>{});
  auto partitioned_global_a = cta_mma.partition_A(global_a);
  auto partitioned_global_b = cta_mma.partition_B(global_b);
  auto [tma_global_a, tma_shared_a] = tma_partition(
      tma_atom_a,
      Int<0>{},
      Layout<_1>{},
      group_modes<0, 3>(shared_a),
      group_modes<0, 3>(partitioned_global_a));
  auto [tma_global_b, tma_shared_b] = tma_partition(
      tma_atom_b,
      Int<0>{},
      Layout<_1>{},
      group_modes<0, 3>(shared_b),
      group_modes<0, 3>(partitioned_global_b));

  auto accumulator_shape = partition_shape_C(
      tiled_mma,
      make_shape(
          Int<kLmTcgenColumns>{},
          Int<kLmTcgenRows>{},
          Int<kLmTcgenAccumulatorStages>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  constexpr int kTmemColumns = 32;
  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  const int warp = threadIdx.x / 32;
  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        kLmTcgenStages>(storage.ab_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        kLmTcgenStages>(storage.ab_empty, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        kLmTcgenAccumulatorStages>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        kLmTcgenAccumulatorStages>(storage.accumulator_empty, 4);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  cutlass::arch::NamedBarrier tmem_allocation_barrier(
      32 + 128,
      cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_barrier(
      128,
      cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  if (warp == 0) {
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  if (warp <= 4) {
    tmem_allocation_barrier.arrive_and_wait();
    tmem_accumulator.data() = storage.tmem_base_ptr;
  }

  constexpr int kOutputTileCount = 151936 / kLmTcgenColumns;
  constexpr int kReductionTileCount = 2560 / kLmTcgenReductionTile;
  const int work_begin = static_cast<int>(
      static_cast<int64_t>(blockIdx.x) * kOutputTileCount / gridDim.x);
  const int work_end = static_cast<int>(
      static_cast<int64_t>(blockIdx.x + 1) * kOutputTileCount / gridDim.x);
  const int work_tiles = work_end - work_begin;
  constexpr int kTmaTransactionBytes =
      sizeof(LmTcgenType) *
      (kLmTcgenColumns * kLmTcgenReductionTile +
       kLmTcgenRows * kLmTcgenReductionTile);

  if (warp == 5) {
    const int stage_iterations = work_tiles * kReductionTileCount;
    for (int iteration = 0; iteration < stage_iterations; ++iteration) {
      const int stage = iteration % kLmTcgenStages;
      const int empty_phase = (iteration / kLmTcgenStages) % 2 ^ 1;
      wait_barrier(storage.ab_empty[stage], empty_phase);
      if (elect_one_sync()) {
        const int output_tile = work_begin + iteration / kReductionTileCount;
        const int reduction_tile = iteration % kReductionTileCount;
        set_barrier_transaction_bytes(
            storage.ab_full[stage], kTmaTransactionBytes);
        copy(
            tma_atom_a.with(storage.ab_full[stage]),
            tma_global_a(_, output_tile, reduction_tile),
            tma_shared_a(_, stage));
        copy(
            tma_atom_b.with(storage.ab_full[stage]),
            tma_global_b(_, 0, reduction_tile),
            tma_shared_b(_, stage));
      }
    }
  } else if (warp == 4) {
    int iteration = 0;
    for (int tile_index = 0; tile_index < work_tiles; ++tile_index) {
      const int accumulator_stage =
          tile_index % kLmTcgenAccumulatorStages;
      const int empty_phase =
          (tile_index / kLmTcgenAccumulatorStages) % 2 ^ 1;
      wait_barrier(storage.accumulator_empty[accumulator_stage], empty_phase);
      auto accumulator =
          tmem_accumulator(_, _, _, accumulator_stage);
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      for (int reduction_tile = 0;
           reduction_tile < kReductionTileCount;
           ++reduction_tile, ++iteration) {
        const int stage = iteration % kLmTcgenStages;
        const int full_phase = (iteration / kLmTcgenStages) % 2;
        wait_barrier(storage.ab_full[stage], full_phase);
        for (int k_block = 0; k_block < size<2>(fragment_a); ++k_block) {
          gemm(
              tiled_mma,
              fragment_a(_, _, k_block, stage),
              fragment_b(_, _, k_block, stage),
              accumulator);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&storage.ab_empty[stage]);
      }
      cutlass::arch::umma_arrive(
          &storage.accumulator_full[accumulator_stage]);
    }
  } else if (warp < 4) {
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{},
        tmem_accumulator(_, _, _, 0));
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_tmem = thread_copy.partition_S(tmem_accumulator);
    auto output_tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<LmTcgenType*>(output)),
        make_layout(
            make_shape(Int<151936>{}, Int<kLmTcgenRows>{}),
            make_stride(Int<1>{}, Int<151936>{})));
    cutlass::NumericConverter<LmTcgenType, float> convert;
    for (int tile_index = 0; tile_index < work_tiles; ++tile_index) {
      const int output_tile = work_begin + tile_index;
      const int accumulator_stage =
          tile_index % kLmTcgenAccumulatorStages;
      const int full_phase =
          (tile_index / kLmTcgenAccumulatorStages) % 2;
      auto global_output = local_tile(
          output_tensor,
          mma_tiler,
          make_coord(output_tile, Int<0>{}, _),
          Step<_1, _1, X>{});
      auto partitioned_global_output = cta_mma.partition_C(global_output);
      auto thread_global = thread_copy.partition_D(partitioned_global_output);
      auto register_accumulator = make_tensor<float>(shape(thread_global));
      auto register_output =
          make_tensor<LmTcgenType>(shape(thread_global));
      wait_barrier(
          storage.accumulator_full[accumulator_stage], full_phase);
      copy(
          tmem_to_register,
          thread_tmem(_, _, _, _, accumulator_stage),
          register_accumulator);
      epilogue_barrier.arrive_and_wait();
      if (elect_one_sync()) {
        arrive_barrier(storage.accumulator_empty[accumulator_stage]);
      }
      CUTE_UNROLL
      for (int element = 0; element < size(register_output); ++element) {
        register_output(element) = convert(register_accumulator(element));
      }
      copy(register_output, thread_global);
    }
  }

  __syncthreads();
  if (warp == 0) {
    // Persistent megakernel CTAs retain their rasterization permit.
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}
#endif

__device__ __forceinline__ int global_thread_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int global_thread_count() {
  return gridDim.x * blockDim.x;
}

__device__ __forceinline__ int balanced_warp_work_begin(int total_work) {
  const int resident_warps = gridDim.x * kWarpsPerBlock;
  if (total_work <= resident_warps) {
    return blockIdx.x * kWarpsPerBlock;
  }
  return static_cast<int>(
      static_cast<int64_t>(blockIdx.x) * total_work / gridDim.x);
}

__device__ __forceinline__ int balanced_warp_work_end(int total_work) {
  const int resident_warps = gridDim.x * kWarpsPerBlock;
  if (total_work <= resident_warps) {
    return min((blockIdx.x + 1) * kWarpsPerBlock, total_work);
  }
  return static_cast<int>(
      static_cast<int64_t>(blockIdx.x + 1) * total_work / gridDim.x);
}

__device__ __forceinline__ uint64_t pack_max_index(float value, int index) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t ordered =
      bits ^ ((bits & 0x80000000U) != 0 ? 0xffffffffU : 0x80000000U);
  return (static_cast<uint64_t>(ordered) << 32) |
      static_cast<uint32_t>(0xffffffffU - static_cast<uint32_t>(index));
}

__device__ __forceinline__ float unpack_max_value(uint64_t packed) {
  const uint32_t ordered = static_cast<uint32_t>(packed >> 32);
  const uint32_t bits =
      ordered ^ ((ordered & 0x80000000U) != 0 ? 0x80000000U : 0xffffffffU);
  return __uint_as_float(bits);
}

__device__ __forceinline__ int unpack_max_index(uint64_t packed) {
  return static_cast<int>(0xffffffffU - static_cast<uint32_t>(packed));
}

template <typename scalar_t>
__device__ void linear(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int rows,
    int input_width,
    int output_width,
    int64_t* trace,
    int phase,
    const LmTcgenTmaAtomA& lm_tma_a,
    const LmTcgenTmaAtomB& lm_tma_b,
    bool input_has_sixteen_rows = true) {
  record_full_timestamp(trace, phase, 0);
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (input_has_sixteen_rows && rows <= 16 && input_width == 2560 &&
        output_width == 151936) {
      linear_b200_tcgen(
          input,
          output,
          rows,
          input_width,
          output_width,
          lm_tma_a,
          lm_tma_b);
      record_full_timestamp(trace, phase, 1);
      cg::this_grid().sync();
      record_full_timestamp(trace, phase, 2);
      return;
    }
    // The released LM head has one padded 16-row input tile and 9,496
    // vocabulary tiles. Reuse each A fragment across five adjacent N tiles on
    // Blackwell. Every accumulator still visits K tiles in the original order,
    // so this removes redundant loads without changing the stochastic
    // arithmetic tree.
    if (input_has_sixteen_rows && rows <= 16 && input_width % 16 == 0 &&
        output_width >= 65536 && output_width % 80 == 16) {
      using namespace nvcuda;
      constexpr int kOutputTilesPerTask = 5;
      using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
      __shared__ float accumulator_tiles[8 * 16 * 16];
      const int warp = threadIdx.x / 32;
      const int lane = threadIdx.x % 32;
      const int tiles_m = (rows + 15) / 16;
      const int tiles_n = output_width / 16;
      const int tile_groups_n = tiles_n / kOutputTilesPerTask;
      const int task_count = tiles_m * tile_groups_n;
      const int work_begin = balanced_warp_work_begin(task_count);
      const int work_end = balanced_warp_work_end(task_count);
      const auto* input_bf16 = reinterpret_cast<const __nv_bfloat16*>(input);
      const auto* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(weight);
      for (int task = work_begin + warp; task < work_end; task += kWarpsPerBlock) {
        const int tile_m = task / tile_groups_n;
        const int first_tile_n =
            (task - tile_m * tile_groups_n) * kOutputTilesPerTask;
        Accumulator accumulators[kOutputTilesPerTask];
#pragma unroll
        for (int tile = 0; tile < kOutputTilesPerTask; ++tile) {
          wmma::fill_fragment(accumulators[tile], 0.0f);
        }
        for (int tile_k = 0; tile_k < input_width; tile_k += 16) {
          wmma::fragment<
              wmma::matrix_a,
              16,
              16,
              16,
              __nv_bfloat16,
              wmma::row_major>
              a;
          wmma::load_matrix_sync(
              a,
              input_bf16 + static_cast<int64_t>(tile_m * 16) * input_width + tile_k,
              input_width);
#pragma unroll
          for (int tile = 0; tile < kOutputTilesPerTask; ++tile) {
            wmma::fragment<
                wmma::matrix_b,
                16,
                16,
                16,
                __nv_bfloat16,
                wmma::col_major>
                b;
            wmma::load_matrix_sync(
                b,
                weight_bf16 +
                    static_cast<int64_t>(first_tile_n + tile) * 16 * input_width +
                    tile_k,
                input_width);
            wmma::mma_sync(accumulators[tile], a, b, accumulators[tile]);
          }
        }
        float* tile_storage = accumulator_tiles + warp * 16 * 16;
#pragma unroll
        for (int tile = 0; tile < kOutputTilesPerTask; ++tile) {
          wmma::store_matrix_sync(
              tile_storage,
              accumulators[tile],
              16,
              wmma::mem_row_major);
          __syncwarp();
          for (int element = lane; element < 16 * 16; element += 32) {
            const int row = element / 16;
            const int column = element % 16;
            output[
                static_cast<int64_t>(tile_m * 16 + row) * output_width +
                (first_tile_n + tile) * 16 + column] =
                static_cast<scalar_t>(tile_storage[element]);
          }
          __syncwarp();
        }
      }
      // The released vocabulary leaves one 16-column tile after the fixed
      // five-tile groups. Warp 7 in CTA 0 has one group for this geometry, so
      // assigning the remainder here keeps it below the two-group critical
      // path while retaining a compile-time single-accumulator shape.
      if (blockIdx.x == 0 && warp == 7) {
        const int tile_n = tile_groups_n * kOutputTilesPerTask;
        Accumulator accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        for (int tile_k = 0; tile_k < input_width; tile_k += 16) {
          wmma::fragment<
              wmma::matrix_a,
              16,
              16,
              16,
              __nv_bfloat16,
              wmma::row_major>
              a;
          wmma::fragment<
              wmma::matrix_b,
              16,
              16,
              16,
              __nv_bfloat16,
              wmma::col_major>
              b;
          wmma::load_matrix_sync(a, input_bf16 + tile_k, input_width);
          wmma::load_matrix_sync(
              b,
              weight_bf16 + static_cast<int64_t>(tile_n) * 16 * input_width + tile_k,
              input_width);
          wmma::mma_sync(accumulator, a, b, accumulator);
        }
        float* tile_storage = accumulator_tiles + warp * 16 * 16;
        wmma::store_matrix_sync(
            tile_storage,
            accumulator,
            16,
            wmma::mem_row_major);
        __syncwarp();
        for (int element = lane; element < 16 * 16; element += 32) {
          const int row = element / 16;
          const int column = element % 16;
          output[static_cast<int64_t>(row) * output_width + tile_n * 16 + column] =
              static_cast<scalar_t>(tile_storage[element]);
        }
        __syncwarp();
      }
      record_full_timestamp(trace, phase, 1);
      cg::this_grid().sync();
      record_full_timestamp(trace, phase, 2);
      return;
    }
#endif
    if (input_has_sixteen_rows && input_width % 16 == 0 && output_width % 16 == 0) {
      using namespace nvcuda;
      __shared__ float accumulator_tiles[8 * 16 * 16];
      const int warp = threadIdx.x / 32;
      const int lane = threadIdx.x % 32;
      const int tiles_m = (rows + 15) / 16;
      const int tiles_n = output_width / 16;
      const int tile_count = tiles_m * tiles_n;
      const int work_begin = balanced_warp_work_begin(tile_count);
      const int work_end = balanced_warp_work_end(tile_count);
      const auto* input_bf16 = reinterpret_cast<const __nv_bfloat16*>(input);
      const auto* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(weight);
      for (int tile = work_begin + warp; tile < work_end; tile += kWarpsPerBlock) {
        const int tile_m = tile / tiles_n;
        const int tile_n = tile % tiles_n;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        for (int tile_k = 0; tile_k < input_width; tile_k += 16) {
          wmma::fragment<
              wmma::matrix_a,
              16,
              16,
              16,
              __nv_bfloat16,
              wmma::row_major>
              a;
          wmma::fragment<
              wmma::matrix_b,
              16,
              16,
              16,
              __nv_bfloat16,
              wmma::col_major>
              b;
          wmma::load_matrix_sync(
              a,
              input_bf16 + static_cast<int64_t>(tile_m * 16) * input_width + tile_k,
              input_width);
          wmma::load_matrix_sync(
              b,
              weight_bf16 + static_cast<int64_t>(tile_n * 16) * input_width + tile_k,
              input_width);
          wmma::mma_sync(accumulator, a, b, accumulator);
        }
        float* tile_storage = accumulator_tiles + warp * 16 * 16;
        wmma::store_matrix_sync(
            tile_storage,
            accumulator,
            16,
            wmma::mem_row_major);
        __syncwarp();
        for (int element = lane; element < 16 * 16; element += 32) {
          const int row = element / 16;
          const int column = element % 16;
          output[
              static_cast<int64_t>(tile_m * 16 + row) * output_width +
              tile_n * 16 + column] = static_cast<scalar_t>(tile_storage[element]);
        }
        __syncwarp();
      }
      record_full_timestamp(trace, phase, 1);
      cg::this_grid().sync();
      record_full_timestamp(trace, phase, 2);
      return;
    }
  }
  const int thread = global_thread_id();
  const int stride = global_thread_count();
  const int64_t elements = static_cast<int64_t>(rows) * output_width;
  for (int64_t index = thread; index < elements; index += stride) {
    const int row = static_cast<int>(index / output_width);
    const int column = static_cast<int>(index - static_cast<int64_t>(row) * output_width);
    float sum = 0.0f;
    const int64_t input_offset = static_cast<int64_t>(row) * input_width;
    const int64_t weight_offset = static_cast<int64_t>(column) * input_width;
    for (int inner = 0; inner < input_width; ++inner) {
      sum = fmaf(
          as_float(input, input_offset + inner),
          as_float(weight, weight_offset + inner),
          sum);
    }
    output[index] = static_cast<scalar_t>(sum);
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void linear_group(
    const LinearTask<scalar_t>* tasks,
    int task_count,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    using namespace nvcuda;
    __shared__ float accumulator_tiles[8 * 16 * 16];
    int total_tiles = 0;
    for (int task = 0; task < task_count; ++task) {
      total_tiles += ((tasks[task].rows + 15) / 16) * (tasks[task].output_width / 16);
    }
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int work_begin = balanced_warp_work_begin(total_tiles);
    const int work_end = balanced_warp_work_end(total_tiles);
    for (int combined_tile = work_begin + warp; combined_tile < work_end;
         combined_tile += kWarpsPerBlock) {
      int task_index = 0;
      int tile = combined_tile;
      for (; task_index < task_count; ++task_index) {
        const int task_tiles =
            ((tasks[task_index].rows + 15) / 16) *
            (tasks[task_index].output_width / 16);
        if (tile < task_tiles) {
          break;
        }
        tile -= task_tiles;
      }
      const LinearTask<scalar_t>& current = tasks[task_index];
      const int tiles_n = current.output_width / 16;
      const int tile_m = tile / tiles_n;
      const int tile_n = tile % tiles_n;
      const auto* input_bf16 = reinterpret_cast<const __nv_bfloat16*>(current.input);
      const auto* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(current.weight);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      for (int tile_k = 0; tile_k < current.input_width; tile_k += 16) {
        wmma::fragment<
            wmma::matrix_a,
            16,
            16,
            16,
            __nv_bfloat16,
            wmma::row_major>
            a;
        wmma::fragment<
            wmma::matrix_b,
            16,
            16,
            16,
            __nv_bfloat16,
            wmma::col_major>
            b;
        wmma::load_matrix_sync(
            a,
            input_bf16 + static_cast<int64_t>(tile_m * 16) * current.input_width + tile_k,
            current.input_width);
        wmma::load_matrix_sync(
            b,
            weight_bf16 + static_cast<int64_t>(tile_n * 16) * current.input_width + tile_k,
            current.input_width);
        wmma::mma_sync(accumulator, a, b, accumulator);
      }
      float* tile_storage = accumulator_tiles + warp * 16 * 16;
      wmma::store_matrix_sync(tile_storage, accumulator, 16, wmma::mem_row_major);
      __syncwarp();
      for (int element = lane; element < 16 * 16; element += 32) {
        const int row = element / 16;
        const int column = element % 16;
        current.output[
            static_cast<int64_t>(tile_m * 16 + row) * current.output_width +
            tile_n * 16 + column] = static_cast<scalar_t>(tile_storage[element]);
      }
      __syncwarp();
    }
  } else {
    int64_t total_elements = 0;
    for (int task = 0; task < task_count; ++task) {
      total_elements += static_cast<int64_t>(tasks[task].rows) * tasks[task].output_width;
    }
    for (int64_t combined = global_thread_id(); combined < total_elements;
         combined += global_thread_count()) {
      int task_index = 0;
      int64_t index = combined;
      for (; task_index < task_count; ++task_index) {
        const int64_t task_elements =
            static_cast<int64_t>(tasks[task_index].rows) * tasks[task_index].output_width;
        if (index < task_elements) {
          break;
        }
        index -= task_elements;
      }
      const LinearTask<scalar_t>& current = tasks[task_index];
      const int row = static_cast<int>(index / current.output_width);
      const int column = static_cast<int>(index % current.output_width);
      float sum = 0.0f;
      for (int inner = 0; inner < current.input_width; ++inner) {
        sum = fmaf(
            as_float(current.input, static_cast<int64_t>(row) * current.input_width + inner),
            as_float(current.weight, static_cast<int64_t>(column) * current.input_width + inner),
            sum);
      }
      current.output[index] = static_cast<scalar_t>(sum);
    }
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t, bool fuse_swiglu = false>
__device__ void linear_group_split_k(
    const LinearTask<scalar_t>* tasks,
    int task_count,
    int splits,
    float* partials,
    int64_t* trace,
    int phase_base) {
  record_full_timestamp(trace, phase_base, 0);
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    using namespace nvcuda;
    int total_work = 0;
    for (int task = 0; task < task_count; ++task) {
      total_work +=
          ((tasks[task].rows + 15) / 16) * (tasks[task].output_width / 16) * splits;
    }
    const int warp = threadIdx.x / 32;
    const int work_begin = balanced_warp_work_begin(total_work);
    const int work_end = balanced_warp_work_end(total_work);
    for (int combined = work_begin + warp; combined < work_end;
         combined += kWarpsPerBlock) {
      int task_index = 0;
      int task_work = combined;
      int64_t partial_offset = 0;
      for (; task_index < task_count; ++task_index) {
        const int padded_rows = ((tasks[task_index].rows + 15) / 16) * 16;
        const int output_tiles = (padded_rows / 16) * (tasks[task_index].output_width / 16);
        const int work_items = output_tiles * splits;
        if (task_work < work_items) {
          break;
        }
        task_work -= work_items;
        partial_offset +=
            static_cast<int64_t>(splits) * padded_rows * tasks[task_index].output_width;
      }
      const LinearTask<scalar_t>& current = tasks[task_index];
      const int padded_rows = ((current.rows + 15) / 16) * 16;
      const int tiles_n = current.output_width / 16;
      const int output_tile = task_work / splits;
      const int split = task_work % splits;
      const int tile_m = output_tile / tiles_n;
      const int tile_n = output_tile % tiles_n;
      const int k_tiles = current.input_width / 16;
      const int k_tiles_per_split = (k_tiles + splits - 1) / splits;
      const int first_k_tile = split * k_tiles_per_split;
      const int last_k_tile = min(first_k_tile + k_tiles_per_split, k_tiles);
      const auto* input_bf16 = reinterpret_cast<const __nv_bfloat16*>(current.input);
      const auto* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(current.weight);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      for (int k_tile = first_k_tile; k_tile < last_k_tile; ++k_tile) {
        const int tile_k = k_tile * 16;
        wmma::fragment<
            wmma::matrix_a,
            16,
            16,
            16,
            __nv_bfloat16,
            wmma::row_major>
            a;
        wmma::fragment<
            wmma::matrix_b,
            16,
            16,
            16,
            __nv_bfloat16,
            wmma::col_major>
            b;
        wmma::load_matrix_sync(
            a,
            input_bf16 + static_cast<int64_t>(tile_m * 16) * current.input_width + tile_k,
            current.input_width);
        wmma::load_matrix_sync(
            b,
            weight_bf16 + static_cast<int64_t>(tile_n * 16) * current.input_width + tile_k,
            current.input_width);
        wmma::mma_sync(accumulator, a, b, accumulator);
      }
      const int64_t task_elements = static_cast<int64_t>(padded_rows) * current.output_width;
      float* destination = partials + partial_offset + split * task_elements +
          static_cast<int64_t>(tile_m * 16) * current.output_width + tile_n * 16;
      wmma::store_matrix_sync(destination, accumulator, current.output_width, wmma::mem_row_major);
    }
  } else {
    int64_t total_elements = 0;
    for (int task = 0; task < task_count; ++task) {
      total_elements += static_cast<int64_t>(tasks[task].rows) * tasks[task].output_width;
    }
    for (int64_t combined = global_thread_id(); combined < total_elements;
         combined += global_thread_count()) {
      int task_index = 0;
      int64_t index = combined;
      for (; task_index < task_count; ++task_index) {
        const int64_t task_elements =
            static_cast<int64_t>(tasks[task_index].rows) * tasks[task_index].output_width;
        if (index < task_elements) {
          break;
        }
        index -= task_elements;
      }
      const LinearTask<scalar_t>& current = tasks[task_index];
      const int row = static_cast<int>(index / current.output_width);
      const int column = static_cast<int>(index % current.output_width);
      float sum = 0.0f;
      for (int inner = 0; inner < current.input_width; ++inner) {
        sum = fmaf(
            as_float(current.input, static_cast<int64_t>(row) * current.input_width + inner),
            as_float(current.weight, static_cast<int64_t>(column) * current.input_width + inner),
            sum);
      }
      current.output[index] = static_cast<scalar_t>(sum);
    }
  }
  record_full_timestamp(trace, phase_base, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase_base, 2);

  record_full_timestamp(trace, phase_base + 1, 0);
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    if constexpr (fuse_swiglu) {
      const int padded_rows = ((tasks[0].rows + 15) / 16) * 16;
      const int64_t padded_elements =
          static_cast<int64_t>(padded_rows) * tasks[0].output_width;
      const int64_t elements =
          static_cast<int64_t>(tasks[0].rows) * tasks[0].output_width;
      for (int64_t index = global_thread_id(); index < elements;
           index += global_thread_count()) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int split = 0; split < splits; ++split) {
          gate_sum += partials[static_cast<int64_t>(split) * padded_elements + index];
          up_sum += partials[
              static_cast<int64_t>(splits + split) * padded_elements + index];
        }
        const scalar_t rounded_gate = static_cast<scalar_t>(gate_sum);
        const scalar_t rounded_up = static_cast<scalar_t>(up_sum);
        const float gate_value = static_cast<float>(rounded_gate);
        const float silu = gate_value / (1.0f + expf(-gate_value));
        tasks[0].output[index] =
            static_cast<scalar_t>(silu * static_cast<float>(rounded_up));
      }
    } else {
      int64_t total_elements = 0;
      for (int task = 0; task < task_count; ++task) {
        total_elements +=
            static_cast<int64_t>(((tasks[task].rows + 15) / 16) * 16) *
            tasks[task].output_width;
      }
      for (int64_t combined = global_thread_id(); combined < total_elements;
           combined += global_thread_count()) {
        int task_index = 0;
        int64_t index = combined;
        int64_t partial_offset = 0;
        for (; task_index < task_count; ++task_index) {
          const int64_t task_elements =
              static_cast<int64_t>(((tasks[task_index].rows + 15) / 16) * 16) *
              tasks[task_index].output_width;
          if (index < task_elements) {
            break;
          }
          index -= task_elements;
          partial_offset += splits * task_elements;
        }
        const LinearTask<scalar_t>& current = tasks[task_index];
        const int padded_rows = ((current.rows + 15) / 16) * 16;
        const int64_t task_elements =
            static_cast<int64_t>(padded_rows) * current.output_width;
        float sum = 0.0f;
        for (int split = 0; split < splits; ++split) {
          sum += partials[partial_offset + split * task_elements + index];
        }
        current.output[index] = static_cast<scalar_t>(sum);
      }
    }
  } else if constexpr (fuse_swiglu) {
    const int64_t elements =
        static_cast<int64_t>(tasks[0].rows) * tasks[0].output_width;
    for (int64_t index = global_thread_id(); index < elements;
         index += global_thread_count()) {
      const float gate_value = as_float(tasks[0].output, index);
      const float silu = gate_value / (1.0f + expf(-gate_value));
      tasks[0].output[index] =
          static_cast<scalar_t>(silu * as_float(tasks[1].output, index));
    }
  }
  record_full_timestamp(trace, phase_base + 1, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase_base + 1, 2);
}

template <typename scalar_t>
__device__ void linear_split_k(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int rows,
    int input_width,
    int output_width,
    float* partials,
    int64_t* trace,
    int phase_base) {
  record_full_timestamp(trace, phase_base, 0);
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    using namespace nvcuda;
    const int padded_rows = ((rows + 15) / 16) * 16;
    const int tiles_m = padded_rows / 16;
    const int tiles_n = output_width / 16;
    const int output_tiles = tiles_m * tiles_n;
    const int k_tiles = input_width / 16;
    const int splits = min(8, k_tiles);
    const int k_tiles_per_split = (k_tiles + splits - 1) / splits;
    const int warp = threadIdx.x / 32;
    const int total_work = output_tiles * splits;
    const int work_begin = balanced_warp_work_begin(total_work);
    const int work_end = balanced_warp_work_end(total_work);
    const auto* input_bf16 = reinterpret_cast<const __nv_bfloat16*>(input);
    const auto* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(weight);
    for (int work = work_begin + warp; work < work_end; work += kWarpsPerBlock) {
      const int output_tile = work / splits;
      const int split = work % splits;
      const int tile_m = output_tile / tiles_n;
      const int tile_n = output_tile % tiles_n;
      const int first_k_tile = split * k_tiles_per_split;
      const int last_k_tile = min(first_k_tile + k_tiles_per_split, k_tiles);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      for (int k_tile = first_k_tile; k_tile < last_k_tile; ++k_tile) {
        const int tile_k = k_tile * 16;
        wmma::fragment<
            wmma::matrix_a,
            16,
            16,
            16,
            __nv_bfloat16,
            wmma::row_major>
            a;
        wmma::fragment<
            wmma::matrix_b,
            16,
            16,
            16,
            __nv_bfloat16,
            wmma::col_major>
            b;
        wmma::load_matrix_sync(
            a,
            input_bf16 + static_cast<int64_t>(tile_m * 16) * input_width + tile_k,
            input_width);
        wmma::load_matrix_sync(
            b,
            weight_bf16 + static_cast<int64_t>(tile_n * 16) * input_width + tile_k,
            input_width);
        wmma::mma_sync(accumulator, a, b, accumulator);
      }
      float* destination =
          partials + static_cast<int64_t>(split) * padded_rows * output_width +
          static_cast<int64_t>(tile_m * 16) * output_width + tile_n * 16;
      wmma::store_matrix_sync(destination, accumulator, output_width, wmma::mem_row_major);
    }
  } else {
    const int64_t count = static_cast<int64_t>(rows) * output_width;
    for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
      const int row = static_cast<int>(index / output_width);
      const int column = static_cast<int>(index % output_width);
      float sum = 0.0f;
      for (int inner = 0; inner < input_width; ++inner) {
        sum = fmaf(
            as_float(input, static_cast<int64_t>(row) * input_width + inner),
            as_float(weight, static_cast<int64_t>(column) * input_width + inner),
            sum);
      }
      output[index] = static_cast<scalar_t>(sum);
    }
  }
  record_full_timestamp(trace, phase_base, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase_base, 2);

  record_full_timestamp(trace, phase_base + 1, 0);
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    const int padded_rows = ((rows + 15) / 16) * 16;
    const int splits = min(8, input_width / 16);
    const int64_t count = static_cast<int64_t>(padded_rows) * output_width;
    for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
      float sum = 0.0f;
      for (int split = 0; split < splits; ++split) {
        sum += partials[static_cast<int64_t>(split) * count + index];
      }
      output[index] = static_cast<scalar_t>(sum);
    }
  }
  record_full_timestamp(trace, phase_base + 1, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase_base + 1, 2);
}

template <typename scalar_t>
__device__ void rms_norm(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int rows,
    int width,
    float epsilon,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  __shared__ float reduction[kThreads];
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    float sum = 0.0f;
    const int64_t offset = static_cast<int64_t>(row) * width;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
      const float value = as_float(input, offset + column);
      sum = fmaf(value, value, sum);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
      output[offset + column] = static_cast<scalar_t>(
          as_float(input, offset + column) * inverse_rms * as_float(weight, column));
    }
    __syncthreads();
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void add_in_place(
    scalar_t* destination,
    const scalar_t* source,
    int64_t count,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
    destination[index] = static_cast<scalar_t>(
        as_float(destination, index) + as_float(source, index));
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void add_and_rms_norm(
    scalar_t* destination,
    const scalar_t* source,
    const scalar_t* weight,
    scalar_t* output,
    int rows,
    int width,
    float epsilon,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  __shared__ float reduction[kThreads];
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t offset = static_cast<int64_t>(row) * width;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
      const int64_t index = offset + column;
      const scalar_t rounded = static_cast<scalar_t>(
          as_float(destination, index) + as_float(source, index));
      destination[index] = rounded;
      const float value = static_cast<float>(rounded);
      sum = fmaf(value, value, sum);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
      const int64_t index = offset + column;
      output[index] = static_cast<scalar_t>(
          as_float(destination, index) * inverse_rms * as_float(weight, column));
    }
    __syncthreads();
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void gather_embeddings(
    const int64_t* token_ids,
    const int64_t* first_prev_token,
    const scalar_t* embedding,
    scalar_t* output,
    int rows,
    int hidden_size,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  const int64_t count = static_cast<int64_t>(rows) * hidden_size;
  for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
    const int row = static_cast<int>(index / hidden_size);
    const int column = static_cast<int>(index - static_cast<int64_t>(row) * hidden_size);
    const int64_t token = row == 0 ? first_prev_token[0] : token_ids[row];
    output[index] = embedding[token * static_cast<int64_t>(hidden_size) + column];
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void rms_norm_and_gather_embeddings(
    scalar_t* context,
    const scalar_t* context_weight,
    int context_rows,
    int hidden_size,
    float epsilon,
    const int64_t* token_ids,
    const int64_t* first_prev_token,
    const scalar_t* embedding,
    scalar_t* hidden,
    int hidden_rows,
    int padded_hidden_rows,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  __shared__ float reduction[kThreads];
  if (blockIdx.x == 0) {
    for (int row = 0; row < context_rows; ++row) {
      float sum = 0.0f;
      const int64_t offset = static_cast<int64_t>(row) * hidden_size;
      for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float value = as_float(context, offset + column);
        sum = fmaf(value, value, sum);
      }
      reduction[threadIdx.x] = sum;
      __syncthreads();
      for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
        if (threadIdx.x < delta) {
          reduction[threadIdx.x] += reduction[threadIdx.x + delta];
        }
        __syncthreads();
      }
      const float inverse_rms = rsqrtf(reduction[0] / hidden_size + epsilon);
      for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        context[offset + column] = static_cast<scalar_t>(
            as_float(context, offset + column) * inverse_rms *
            as_float(context_weight, column));
      }
      __syncthreads();
    }
  }
  const int64_t embedding_count = static_cast<int64_t>(padded_hidden_rows) * hidden_size;
  if (gridDim.x == 1) {
    for (int64_t index = threadIdx.x; index < embedding_count; index += blockDim.x) {
      const int row = static_cast<int>(index / hidden_size);
      const int column = static_cast<int>(index - static_cast<int64_t>(row) * hidden_size);
      if (row < hidden_rows) {
        const int64_t token = row == 0 ? first_prev_token[0] : token_ids[row];
        hidden[index] = embedding[token * static_cast<int64_t>(hidden_size) + column];
      } else {
        hidden[index] = static_cast<scalar_t>(0.0f);
      }
    }
  } else if (blockIdx.x > 0) {
    const int worker = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    const int worker_count = (gridDim.x - 1) * blockDim.x;
    for (int64_t index = worker; index < embedding_count; index += worker_count) {
      const int row = static_cast<int>(index / hidden_size);
      const int column = static_cast<int>(index - static_cast<int64_t>(row) * hidden_size);
      if (row < hidden_rows) {
        const int64_t token = row == 0 ? first_prev_token[0] : token_ids[row];
        hidden[index] = embedding[token * static_cast<int64_t>(hidden_size) + column];
      } else {
        hidden[index] = static_cast<scalar_t>(0.0f);
      }
    }
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void pad_rows(
    const scalar_t* input,
    scalar_t* output,
    int rows,
    int padded_rows,
    int width,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  const int64_t count = static_cast<int64_t>(padded_rows) * width;
  const int64_t input_count = static_cast<int64_t>(rows) * width;
  for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
    output[index] = index < input_count ? input[index] : static_cast<scalar_t>(0.0f);
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void head_rms_norm(
    scalar_t* values,
    const scalar_t* weight,
    int vectors,
    int head_dim,
    float epsilon,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  __shared__ float reduction[kThreads];
  for (int vector = blockIdx.x; vector < vectors; vector += gridDim.x) {
    float sum = 0.0f;
    const int64_t offset = static_cast<int64_t>(vector) * head_dim;
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
      const float value = as_float(values, offset + column);
      sum = fmaf(value, value, sum);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / head_dim + epsilon);
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
      values[offset + column] = static_cast<scalar_t>(
          as_float(values, offset + column) * inverse_rms * as_float(weight, column));
    }
    __syncthreads();
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void head_rms_norm_qkv(
    scalar_t* q,
    scalar_t* k_context,
    scalar_t* k_draft,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    int q_vectors,
    int context_k_vectors,
    int draft_k_vectors,
    int head_dim,
    float epsilon,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  __shared__ float reduction[kThreads];
  const int total_vectors = q_vectors + context_k_vectors + draft_k_vectors;
  for (int combined_vector = blockIdx.x; combined_vector < total_vectors;
       combined_vector += gridDim.x) {
    scalar_t* values;
    const scalar_t* weight;
    int vector;
    if (combined_vector < q_vectors) {
      values = q;
      weight = q_weight;
      vector = combined_vector;
    } else if (combined_vector < q_vectors + context_k_vectors) {
      values = k_context;
      weight = k_weight;
      vector = combined_vector - q_vectors;
    } else {
      values = k_draft;
      weight = k_weight;
      vector = combined_vector - q_vectors - context_k_vectors;
    }
    float sum = 0.0f;
    const int64_t offset = static_cast<int64_t>(vector) * head_dim;
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
      const float value = as_float(values, offset + column);
      sum = fmaf(value, value, sum);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / head_dim + epsilon);
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
      values[offset + column] = static_cast<scalar_t>(
          as_float(values, offset + column) * inverse_rms * as_float(weight, column));
    }
    __syncthreads();
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void apply_rope(
    scalar_t* values,
    const int64_t* position_ids,
    int position_offset,
    int rows,
    int heads,
    int head_dim,
    float theta,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  const int half = head_dim / 2;
  const int64_t count = static_cast<int64_t>(rows) * heads * half;
  for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
    const int frequency = static_cast<int>(index % half);
    const int64_t vector = index / half;
    const int head = static_cast<int>(vector % heads);
    const int row = static_cast<int>(vector / heads);
    const int64_t offset =
        (static_cast<int64_t>(row) * heads + head) * head_dim;
    const float exponent = -2.0f * static_cast<float>(frequency) / head_dim;
    const float inverse_frequency = powf(theta, exponent);
    const float angle =
        static_cast<float>(position_ids[position_offset + row]) * inverse_frequency;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);
    const float first = as_float(values, offset + frequency);
    const float second = as_float(values, offset + half + frequency);
    values[offset + frequency] = static_cast<scalar_t>(first * cosine - second * sine);
    values[offset + half + frequency] =
        static_cast<scalar_t>(second * cosine + first * sine);
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void apply_rope_qkv(
    scalar_t* q,
    scalar_t* k_context,
    scalar_t* k_draft,
    const int64_t* position_ids,
    int context_length,
    int block_size,
    int query_heads,
    int kv_heads,
    int head_dim,
    float theta,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  const int half = head_dim / 2;
  const int64_t q_count = static_cast<int64_t>(block_size) * query_heads * half;
  const int64_t context_k_count =
      static_cast<int64_t>(context_length) * kv_heads * half;
  const int64_t draft_k_count = static_cast<int64_t>(block_size) * kv_heads * half;
  const int64_t total_count = q_count + context_k_count + draft_k_count;
  for (int64_t combined = global_thread_id(); combined < total_count;
       combined += global_thread_count()) {
    scalar_t* values;
    int64_t index;
    int heads;
    int position_offset;
    if (combined < q_count) {
      values = q;
      index = combined;
      heads = query_heads;
      position_offset = context_length;
    } else if (combined < q_count + context_k_count) {
      values = k_context;
      index = combined - q_count;
      heads = kv_heads;
      position_offset = 0;
    } else {
      values = k_draft;
      index = combined - q_count - context_k_count;
      heads = kv_heads;
      position_offset = context_length;
    }
    const int frequency = static_cast<int>(index % half);
    const int64_t vector = index / half;
    const int head = static_cast<int>(vector % heads);
    const int row = static_cast<int>(vector / heads);
    const int64_t offset =
        (static_cast<int64_t>(row) * heads + head) * head_dim;
    const float exponent = -2.0f * static_cast<float>(frequency) / head_dim;
    const float inverse_frequency = powf(theta, exponent);
    const float angle =
        static_cast<float>(position_ids[position_offset + row]) * inverse_frequency;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);
    const float first = as_float(values, offset + frequency);
    const float second = as_float(values, offset + half + frequency);
    values[offset + frequency] = static_cast<scalar_t>(first * cosine - second * sine);
    values[offset + half + frequency] =
        static_cast<scalar_t>(second * cosine + first * sine);
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void head_rms_norm_rope_qkv(
    scalar_t* q,
    scalar_t* k_context,
    scalar_t* k_draft,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    const int64_t* position_ids,
    int context_length,
    int block_size,
    int query_heads,
    int kv_heads,
    int head_dim,
    float epsilon,
    float theta,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  const int q_vectors = block_size * query_heads;
  const int context_k_vectors = context_length * kv_heads;
  const int draft_k_vectors = block_size * kv_heads;
  const int total_vectors = q_vectors + context_k_vectors + draft_k_vectors;
  const int half = head_dim / 2;
  if (head_dim <= 128) {
    __shared__ float reduction[2][128];
    const int vector_slot = threadIdx.x / 128;
    const int group_thread = threadIdx.x % 128;
    for (int vector_base = blockIdx.x * 2; vector_base < total_vectors;
         vector_base += gridDim.x * 2) {
      const int combined_vector = vector_base + vector_slot;
      const bool valid = combined_vector < total_vectors;
      scalar_t* values = nullptr;
      const scalar_t* weight = nullptr;
      int vector = 0;
      int heads = 1;
      int position_offset = 0;
      if (valid && combined_vector < q_vectors) {
        values = q;
        weight = q_weight;
        vector = combined_vector;
        heads = query_heads;
        position_offset = context_length;
      } else if (valid && combined_vector < q_vectors + context_k_vectors) {
        values = k_context;
        weight = k_weight;
        vector = combined_vector - q_vectors;
        heads = kv_heads;
      } else if (valid) {
        values = k_draft;
        weight = k_weight;
        vector = combined_vector - q_vectors - context_k_vectors;
        heads = kv_heads;
        position_offset = context_length;
      }
      float sum = 0.0f;
      const int64_t offset = static_cast<int64_t>(vector) * head_dim;
      if (valid && group_thread < head_dim) {
        const float value = as_float(values, offset + group_thread);
        sum = fmaf(value, value, sum);
      }
      reduction[vector_slot][group_thread] = sum;
      __syncthreads();
      for (int delta = 64; delta > 0; delta /= 2) {
        if (group_thread < delta) {
          reduction[vector_slot][group_thread] +=
              reduction[vector_slot][group_thread + delta];
        }
        __syncthreads();
      }
      if (valid && group_thread < head_dim) {
        const float inverse_rms =
            rsqrtf(reduction[vector_slot][0] / head_dim + epsilon);
        values[offset + group_thread] = static_cast<scalar_t>(
            as_float(values, offset + group_thread) * inverse_rms *
            as_float(weight, group_thread));
      }
      __syncthreads();
      if (valid && group_thread < half) {
        const int row = vector / heads;
        const float exponent = -2.0f * static_cast<float>(group_thread) / head_dim;
        const float inverse_frequency = powf(theta, exponent);
        const float angle =
            static_cast<float>(position_ids[position_offset + row]) * inverse_frequency;
        float sine;
        float cosine;
        sincosf(angle, &sine, &cosine);
        const float first = as_float(values, offset + group_thread);
        const float second = as_float(values, offset + half + group_thread);
        values[offset + group_thread] =
            static_cast<scalar_t>(first * cosine - second * sine);
        values[offset + half + group_thread] =
            static_cast<scalar_t>(second * cosine + first * sine);
      }
      __syncthreads();
    }
  } else {
    __shared__ float reduction[kThreads];
    for (int combined_vector = blockIdx.x; combined_vector < total_vectors;
         combined_vector += gridDim.x) {
      scalar_t* values;
      const scalar_t* weight;
      int vector;
      int heads;
      int position_offset;
      if (combined_vector < q_vectors) {
        values = q;
        weight = q_weight;
        vector = combined_vector;
        heads = query_heads;
        position_offset = context_length;
      } else if (combined_vector < q_vectors + context_k_vectors) {
        values = k_context;
        weight = k_weight;
        vector = combined_vector - q_vectors;
        heads = kv_heads;
        position_offset = 0;
      } else {
        values = k_draft;
        weight = k_weight;
        vector = combined_vector - q_vectors - context_k_vectors;
        heads = kv_heads;
        position_offset = context_length;
      }
      float sum = 0.0f;
      const int64_t offset = static_cast<int64_t>(vector) * head_dim;
      for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        const float value = as_float(values, offset + column);
        sum = fmaf(value, value, sum);
      }
      reduction[threadIdx.x] = sum;
      __syncthreads();
      for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
        if (threadIdx.x < delta) {
          reduction[threadIdx.x] += reduction[threadIdx.x + delta];
        }
        __syncthreads();
      }
      const float inverse_rms = rsqrtf(reduction[0] / head_dim + epsilon);
      for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        values[offset + column] = static_cast<scalar_t>(
            as_float(values, offset + column) * inverse_rms * as_float(weight, column));
      }
      __syncthreads();
      const int row = vector / heads;
      for (int frequency = threadIdx.x; frequency < half; frequency += blockDim.x) {
        const float exponent = -2.0f * static_cast<float>(frequency) / head_dim;
        const float inverse_frequency = powf(theta, exponent);
        const float angle =
            static_cast<float>(position_ids[position_offset + row]) * inverse_frequency;
        float sine;
        float cosine;
        sincosf(angle, &sine, &cosine);
        const float first = as_float(values, offset + frequency);
        const float second = as_float(values, offset + half + frequency);
        values[offset + frequency] = static_cast<scalar_t>(first * cosine - second * sine);
        values[offset + half + frequency] =
            static_cast<scalar_t>(second * cosine + first * sine);
      }
      __syncthreads();
    }
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ __forceinline__ float key_value(
    const scalar_t* cache,
    const scalar_t* context,
    const scalar_t* draft,
    int layer,
    int key_position,
    int kv_head,
    int dimension,
    int past_length,
    int context_length,
    int max_cache_length,
    int kv_heads,
    int head_dim) {
  if (key_position < past_length) {
    const int64_t offset =
        (((static_cast<int64_t>(layer) * max_cache_length + key_position) * kv_heads +
          kv_head) * head_dim + dimension);
    return as_float(cache, offset);
  }
  if (key_position < past_length + context_length) {
    const int row = key_position - past_length;
    return as_float(
        context,
        (static_cast<int64_t>(row) * kv_heads + kv_head) * head_dim + dimension);
  }
  const int row = key_position - past_length - context_length;
  return as_float(
      draft,
      (static_cast<int64_t>(row) * kv_heads + kv_head) * head_dim + dimension);
}

template <typename scalar_t>
__device__ __forceinline__ void append_context_cache_worker(
    const scalar_t* key,
    const scalar_t* value,
    scalar_t* cache_k,
    scalar_t* cache_v,
    int layer,
    int past_length,
    int context_length,
    int max_cache_length,
    int kv_heads,
    int head_dim,
    int worker,
    int worker_count) {
  const int64_t count = static_cast<int64_t>(context_length) * kv_heads * head_dim;
  for (int64_t index = worker; index < count; index += worker_count) {
    const int row = static_cast<int>(index / (kv_heads * head_dim));
    const int64_t within = index - static_cast<int64_t>(row) * kv_heads * head_dim;
    const int64_t cache_offset =
        (static_cast<int64_t>(layer) * max_cache_length + past_length + row) *
            kv_heads * head_dim +
        within;
    cache_k[cache_offset] = key[index];
    cache_v[cache_offset] = value[index];
  }
}

template <typename scalar_t>
__device__ void attention(
    const scalar_t* q,
    scalar_t* cache_k,
    scalar_t* cache_v,
    const scalar_t* k_context,
    const scalar_t* v_context,
    const scalar_t* k_draft,
    const scalar_t* v_draft,
    scalar_t* output,
    int layer,
    int rows,
    int past_length,
    int context_length,
    int max_cache_length,
    int query_heads,
    int kv_heads,
    int head_dim,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  __shared__ float reduction[kThreads];
  __shared__ float fast_scores[kAttentionVectorsPerBlock][kFastAttentionMaxKeys];
  __shared__ float fast_max[kAttentionVectorsPerBlock];
  __shared__ float fast_inverse_sum[kAttentionVectorsPerBlock];
  const int key_count = past_length + context_length + rows;
  const int groups = query_heads / kv_heads;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    if (head_dim == kFastAttentionHeadDim && key_count <= kFastAttentionMaxKeys) {
      const int warp = threadIdx.x / 32;
      const int lane = threadIdx.x % 32;
      const int vector_slot = warp / kAttentionWarpsPerVector;
      const int value_tile = warp % kAttentionWarpsPerVector;
      const int vector_count = rows * query_heads;
      const int attention_blocks =
          min(gridDim.x, (vector_count + kAttentionVectorsPerBlock - 1) /
                  kAttentionVectorsPerBlock);
      if (blockIdx.x < attention_blocks) {
        for (int vector_base = blockIdx.x * kAttentionVectorsPerBlock;
             vector_base < vector_count;
             vector_base += attention_blocks * kAttentionVectorsPerBlock) {
          const int vector = vector_base + vector_slot;
          if (value_tile == 0 && vector < vector_count) {
            const int query_head = vector % query_heads;
            const int kv_head = query_head / groups;
            const int64_t q_offset = static_cast<int64_t>(vector) * head_dim;
            float local_max = -3.402823466e+38F;
            if (lane < key_count) {
              float score = 0.0f;
              for (int dimension = 0; dimension < head_dim; ++dimension) {
                score = fmaf(
                    as_float(q, q_offset + dimension),
                    key_value(
                        cache_k,
                        k_context,
                        k_draft,
                        layer,
                        lane,
                        kv_head,
                        dimension,
                        past_length,
                        context_length,
                        max_cache_length,
                        kv_heads,
                        head_dim),
                    score);
              }
              local_max = score * scale;
              fast_scores[vector_slot][lane] = local_max;
            }
            for (int delta = 16; delta > 0; delta /= 2) {
              local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, delta));
            }
            const float maximum = __shfl_sync(0xffffffff, local_max, 0);
            float local_sum = lane < key_count
                ? expf(fast_scores[vector_slot][lane] - maximum)
                : 0.0f;
            for (int delta = 16; delta > 0; delta /= 2) {
              local_sum += __shfl_down_sync(0xffffffff, local_sum, delta);
            }
            if (lane == 0) {
              fast_max[vector_slot] = maximum;
              fast_inverse_sum[vector_slot] = 1.0f / local_sum;
            }
          }
          __syncthreads();
          if (vector < vector_count) {
            const int query_head = vector % query_heads;
            const int kv_head = query_head / groups;
            const int dimension = value_tile * 32 + lane;
            float value = 0.0f;
            for (int key = 0; key < key_count; ++key) {
              const float probability =
                  expf(fast_scores[vector_slot][key] - fast_max[vector_slot]) *
                  fast_inverse_sum[vector_slot];
              value = fmaf(
                  probability,
                  key_value(
                      cache_v,
                      v_context,
                      v_draft,
                      layer,
                      key,
                      kv_head,
                      dimension,
                      past_length,
                      context_length,
                      max_cache_length,
                      kv_heads,
                      head_dim),
                  value);
            }
            output[static_cast<int64_t>(vector) * head_dim + dimension] =
                static_cast<scalar_t>(value);
          }
          __syncthreads();
        }
      } else {
        const int cache_block = blockIdx.x - attention_blocks;
        append_context_cache_worker(
            k_context,
            v_context,
            cache_k,
            cache_v,
            layer,
            past_length,
            context_length,
            max_cache_length,
            kv_heads,
            head_dim,
            cache_block * blockDim.x + threadIdx.x,
            (gridDim.x - attention_blocks) * blockDim.x);
      }
      if (attention_blocks == gridDim.x) {
        append_context_cache_worker(
            k_context,
            v_context,
            cache_k,
            cache_v,
            layer,
            past_length,
            context_length,
            max_cache_length,
            kv_heads,
            head_dim,
            global_thread_id(),
            global_thread_count());
      }
      record_full_timestamp(trace, phase, 1);
      cg::this_grid().sync();
      record_full_timestamp(trace, phase, 2);
      return;
    }
  }
  for (int vector = blockIdx.x; vector < rows * query_heads; vector += gridDim.x) {
    const int row = vector / query_heads;
    const int query_head = vector % query_heads;
    const int kv_head = query_head / groups;
    const int64_t q_offset = static_cast<int64_t>(vector) * head_dim;
    float local_max = -3.402823466e+38F;
    for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
      float score = 0.0f;
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        score = fmaf(
            as_float(q, q_offset + dimension),
            key_value(
                cache_k,
                k_context,
                k_draft,
                layer,
                key,
                kv_head,
                dimension,
                past_length,
                context_length,
                max_cache_length,
                kv_heads,
                head_dim),
            score);
      }
      local_max = fmaxf(local_max, score * scale);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + delta]);
      }
      __syncthreads();
    }
    const float maximum = reduction[0];
    float local_sum = 0.0f;
    for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
      float score = 0.0f;
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        score = fmaf(
            as_float(q, q_offset + dimension),
            key_value(
                cache_k,
                k_context,
                k_draft,
                layer,
                key,
                kv_head,
                dimension,
                past_length,
                context_length,
                max_cache_length,
                kv_heads,
                head_dim),
            score);
      }
      local_sum += expf(score * scale - maximum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_sum = 1.0f / reduction[0];
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
      float value = 0.0f;
      for (int key = 0; key < key_count; ++key) {
        float score = 0.0f;
        for (int inner = 0; inner < head_dim; ++inner) {
          score = fmaf(
              as_float(q, q_offset + inner),
              key_value(
                  cache_k,
                  k_context,
                  k_draft,
                  layer,
                  key,
                  kv_head,
                  inner,
                  past_length,
                  context_length,
                  max_cache_length,
                  kv_heads,
                  head_dim),
              score);
        }
        const float probability = expf(score * scale - maximum) * inverse_sum;
        value = fmaf(
            probability,
            key_value(
                cache_v,
                v_context,
                v_draft,
                layer,
                key,
                kv_head,
                dimension,
                past_length,
                context_length,
                max_cache_length,
                kv_heads,
                head_dim),
            value);
      }
      output[q_offset + dimension] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
  append_context_cache_worker(
      k_context,
      v_context,
      cache_k,
      cache_v,
      layer,
      past_length,
      context_length,
      max_cache_length,
      kv_heads,
      head_dim,
      global_thread_id(),
      global_thread_count());
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void append_context_cache(
    const scalar_t* key,
    const scalar_t* value,
    scalar_t* cache_k,
    scalar_t* cache_v,
    int layer,
    int past_length,
    int context_length,
    int max_cache_length,
    int kv_heads,
    int head_dim,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  const int64_t count = static_cast<int64_t>(context_length) * kv_heads * head_dim;
  for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
    const int row = static_cast<int>(index / (kv_heads * head_dim));
    const int64_t within = index - static_cast<int64_t>(row) * kv_heads * head_dim;
    const int64_t cache_offset =
        (static_cast<int64_t>(layer) * max_cache_length + past_length + row) *
            kv_heads * head_dim +
        within;
    cache_k[cache_offset] = key[index];
    cache_v[cache_offset] = value[index];
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void swiglu(
    scalar_t* gate,
    const scalar_t* up,
    int64_t count,
    int64_t* trace,
    int phase) {
  record_full_timestamp(trace, phase, 0);
  for (int64_t index = global_thread_id(); index < count; index += global_thread_count()) {
    const float gate_value = as_float(gate, index);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    gate[index] = static_cast<scalar_t>(silu * as_float(up, index));
  }
  record_full_timestamp(trace, phase, 1);
  cg::this_grid().sync();
  record_full_timestamp(trace, phase, 2);
}

template <typename scalar_t>
__device__ void proposal_tail(
    const FullWeights<scalar_t>& weights,
    FullWorkspace<scalar_t>& workspace,
    const int64_t* draft_input_ids,
    const int64_t* first_prev_token,
    const float* uniforms,
    float temperature,
    int steps,
    int vocab_size,
    int hidden_size,
    int rank,
    int phase_base) {
  __shared__ float reduction[kThreads];
  __shared__ int reduction_index[kThreads];
  __shared__ c10::BFloat16 markov_embedding[16 * kMarkovRank];
  __shared__ c10::BFloat16
      markov_w2_tile[kWarpsPerBlock][kMarkovTileColumns][kMarkovTileStride];
  __shared__ float markov_accumulator_tiles[kWarpsPerBlock * 16 * 16];
  int64_t previous_token = first_prev_token[0];
  for (int step = 0; step < steps; ++step) {
    const int chunk_size = (vocab_size + gridDim.x - 1) / gridDim.x;
    const int chunk_begin = min(blockIdx.x * chunk_size, vocab_size);
    const int chunk_end = min(chunk_begin + chunk_size, vocab_size);
    const int markov_phase = phase_base + step * 4;
    const int exponential_phase = markov_phase + 1;
    const int sum_phase = markov_phase + 2;
    const int normalize_sample_phase = markov_phase + 3;
    record_full_timestamp(workspace.trace, markov_phase, 0);
    float local_max = -3.402823466e+38F;
    int local_index = vocab_size;
    if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      if (rank == kMarkovRank) {
        const auto* markov_w1 =
            reinterpret_cast<const c10::BFloat16*>(weights.markov_w1);
        const auto* markov_w2 =
            reinterpret_cast<const c10::BFloat16*>(weights.markov_w2);
        markov_embedding[threadIdx.x] =
            markov_w1[previous_token * static_cast<int64_t>(rank) + threadIdx.x];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        for (int index = kMarkovRank + threadIdx.x;
             index < 16 * kMarkovRank;
             index += blockDim.x) {
          markov_embedding[index] = c10::BFloat16(0.0f);
        }
#endif
        __syncthreads();
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        using namespace nvcuda;
        const int tiles_n = vocab_size / 16;
        const int work_begin = balanced_warp_work_begin(tiles_n);
        const int work_end = balanced_warp_work_end(tiles_n);
        const auto* input_bf16 =
            reinterpret_cast<const __nv_bfloat16*>(markov_embedding);
        const auto* markov_w2_bf16 =
            reinterpret_cast<const __nv_bfloat16*>(markov_w2);
        for (int tile_n = work_begin + warp; tile_n < work_end;
             tile_n += kWarpsPerBlock) {
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
          wmma::fill_fragment(accumulator, 0.0f);
          for (int tile_k = 0; tile_k < rank; tile_k += 16) {
            wmma::fragment<
                wmma::matrix_a,
                16,
                16,
                16,
                __nv_bfloat16,
                wmma::row_major>
                a;
            wmma::fragment<
                wmma::matrix_b,
                16,
                16,
                16,
                __nv_bfloat16,
                wmma::col_major>
                b;
            wmma::load_matrix_sync(a, input_bf16 + tile_k, rank);
            wmma::load_matrix_sync(
                b,
                markov_w2_bf16 +
                    static_cast<int64_t>(tile_n) * 16 * rank + tile_k,
                rank);
            wmma::mma_sync(accumulator, a, b, accumulator);
          }
          float* tile_storage =
              markov_accumulator_tiles + warp * 16 * 16;
          wmma::store_matrix_sync(
              tile_storage,
              accumulator,
              16,
              wmma::mem_row_major);
          __syncwarp();
          if (lane < 16) {
            const int vocab = tile_n * 16 + lane;
            const scalar_t rounded_bias = static_cast<scalar_t>(tile_storage[lane]);
            const scalar_t rounded_logit = static_cast<scalar_t>(
                as_float(
                    workspace.base_logits,
                    static_cast<int64_t>(step) * vocab_size + vocab) +
                static_cast<float>(rounded_bias));
            const float corrected = static_cast<float>(rounded_logit);
            workspace.corrected_logits[
                static_cast<int64_t>(step) * vocab_size + vocab] = corrected;
            if (corrected > local_max ||
                (corrected == local_max && vocab < local_index)) {
              local_max = corrected;
              local_index = vocab;
            }
          }
          __syncwarp();
        }
#else
        const int warp_tiles =
            (vocab_size + kMarkovTileRows - 1) / kMarkovTileRows;
        const int work_begin = blockIdx.x * kWarpsPerBlock;
        const int work_end = warp_tiles;
        const int work_stride = gridDim.x * kWarpsPerBlock;
        for (int warp_tile = work_begin + warp; warp_tile < work_end;
             warp_tile += work_stride) {
          const int warp_vocab_begin = warp_tile * kMarkovTileRows;
          const int vocab = warp_vocab_begin + lane;
          float dot = 0.0f;
          for (int rank_tile = 0; rank_tile < rank; rank_tile += kMarkovTileColumns) {
            for (int row = 0; row < kMarkovTileRows; ++row) {
              const int load_vocab = warp_vocab_begin + row;
              markov_w2_tile[warp][lane][row] = load_vocab < vocab_size
                  ? markov_w2[static_cast<int64_t>(load_vocab) * rank + rank_tile + lane]
                  : c10::BFloat16(0.0f);
            }
            __syncwarp();
            for (int column = 0; column < kMarkovTileColumns; ++column) {
              dot = fmaf(
                  as_float(markov_embedding, rank_tile + column),
                  as_float(markov_w2_tile[warp][column], lane),
                  dot);
            }
            __syncwarp();
          }
          if (vocab < vocab_size) {
            const scalar_t rounded_bias = static_cast<scalar_t>(dot);
            const scalar_t rounded_logit = static_cast<scalar_t>(
                as_float(
                    workspace.base_logits,
                    static_cast<int64_t>(step) * vocab_size + vocab) +
                static_cast<float>(rounded_bias));
            const float corrected = static_cast<float>(rounded_logit);
            workspace.corrected_logits[static_cast<int64_t>(step) * vocab_size + vocab] =
                corrected;
            if (corrected > local_max || (corrected == local_max && vocab < local_index)) {
              local_max = corrected;
              local_index = vocab;
            }
          }
        }
#endif
      } else {
        for (int vocab = global_thread_id(); vocab < vocab_size;
             vocab += global_thread_count()) {
          float dot = 0.0f;
          for (int column = 0; column < rank; ++column) {
            dot = fmaf(
                as_float(
                    weights.markov_w1,
                    previous_token * static_cast<int64_t>(rank) + column),
                as_float(weights.markov_w2, static_cast<int64_t>(vocab) * rank + column),
                dot);
          }
          const scalar_t rounded_bias = static_cast<scalar_t>(dot);
          const scalar_t rounded_logit = static_cast<scalar_t>(
              as_float(
                  workspace.base_logits,
                  static_cast<int64_t>(step) * vocab_size + vocab) +
              static_cast<float>(rounded_bias));
          const float corrected = static_cast<float>(rounded_logit);
          workspace.corrected_logits[static_cast<int64_t>(step) * vocab_size + vocab] =
              corrected;
          if (corrected > local_max || (corrected == local_max && vocab < local_index)) {
            local_max = corrected;
            local_index = vocab;
          }
        }
      }
    } else {
      for (int vocab = global_thread_id(); vocab < vocab_size;
           vocab += global_thread_count()) {
        float dot = 0.0f;
        for (int column = 0; column < rank; ++column) {
          dot = fmaf(
              as_float(
                  weights.markov_w1,
                  previous_token * static_cast<int64_t>(rank) + column),
              as_float(weights.markov_w2, static_cast<int64_t>(vocab) * rank + column),
              dot);
        }
        const scalar_t rounded_bias = static_cast<scalar_t>(dot);
        const scalar_t rounded_logit = static_cast<scalar_t>(
            as_float(
                workspace.base_logits,
                static_cast<int64_t>(step) * vocab_size + vocab) +
            static_cast<float>(rounded_bias));
        const float corrected = static_cast<float>(rounded_logit);
        workspace.corrected_logits[static_cast<int64_t>(step) * vocab_size + vocab] =
            corrected;
        if (corrected > local_max || (corrected == local_max && vocab < local_index)) {
          local_max = corrected;
          local_index = vocab;
        }
      }
    }
    reduction[threadIdx.x] = local_max;
    reduction_index[threadIdx.x] = local_index;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        const float other = reduction[threadIdx.x + delta];
        const int other_index = reduction_index[threadIdx.x + delta];
        if (other > reduction[threadIdx.x] ||
            (other == reduction[threadIdx.x] && other_index < reduction_index[threadIdx.x])) {
          reduction[threadIdx.x] = other;
          reduction_index[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicMax(
          reinterpret_cast<unsigned long long*>(workspace.block_max) + step,
          static_cast<unsigned long long>(pack_max_index(reduction[0], reduction_index[0])));
    }
    record_full_timestamp(workspace.trace, markov_phase, 1);
    cg::this_grid().sync();
    record_full_timestamp(workspace.trace, markov_phase, 2);
    const uint64_t packed_max =
        reinterpret_cast<const unsigned long long*>(workspace.block_max)[step];
    const float maximum = unpack_max_value(packed_max);
    const int maximum_index = unpack_max_index(packed_max);

    record_full_timestamp(workspace.trace, exponential_phase, 0);
    float local_sum = 0.0f;
    if (temperature >= 1.0e-5f) {
      for (int vocab = chunk_begin + threadIdx.x; vocab < chunk_end; vocab += blockDim.x) {
        const int64_t offset = static_cast<int64_t>(step) * vocab_size + vocab;
        const float value =
            expf((workspace.corrected_logits[offset] - maximum) / temperature);
        workspace.probabilities[offset] = value;
        local_sum += value;
      }
    } else {
      for (int vocab = chunk_begin + threadIdx.x; vocab < chunk_end; vocab += blockDim.x) {
        workspace.probabilities[static_cast<int64_t>(step) * vocab_size + vocab] =
            vocab == maximum_index ? 1.0f : 0.0f;
      }
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      workspace.block_sum[blockIdx.x] = reduction[0];
    }
    record_full_timestamp(workspace.trace, exponential_phase, 1);
    cg::this_grid().sync();
    record_full_timestamp(workspace.trace, exponential_phase, 2);
    record_full_timestamp(workspace.trace, sum_phase, 0);
    if (blockIdx.x == 0 && threadIdx.x == 0 && temperature >= 1.0e-5f) {
      float sum = 0.0f;
      for (int block = 0; block < gridDim.x; ++block) {
        sum += workspace.block_sum[block];
      }
      workspace.global_values[0] = sum;
      const float threshold = uniforms[step] * sum;
      float prefix = 0.0f;
      int selected_block = gridDim.x - 1;
      for (int block = 0; block < gridDim.x; ++block) {
        const float next = prefix + workspace.block_sum[block];
        if (next >= threshold) {
          selected_block = block;
          break;
        }
        prefix = next;
      }
      workspace.global_values[2] = static_cast<float>(selected_block);
      workspace.global_values[3] = prefix;
    }
    record_full_timestamp(workspace.trace, sum_phase, 1);
    cg::this_grid().sync();
    record_full_timestamp(workspace.trace, sum_phase, 2);
    record_full_timestamp(workspace.trace, normalize_sample_phase, 0);
    if (temperature >= 1.0e-5f) {
      const float inverse_sum = 1.0f / workspace.global_values[0];
      for (int vocab = chunk_begin + threadIdx.x; vocab < chunk_end; vocab += blockDim.x) {
        workspace.probabilities[static_cast<int64_t>(step) * vocab_size + vocab] *= inverse_sum;
      }
    }
    __syncthreads();
    const int selected_block = static_cast<int>(workspace.global_values[2]);
    if (temperature >= 1.0e-5f && blockIdx.x == selected_block) {
      const float total = workspace.global_values[0];
      const float local_threshold =
          uniforms[step] * total - workspace.global_values[3];
      const int chunk_length = chunk_end - chunk_begin;
      const int items_per_thread = (chunk_length + blockDim.x - 1) / blockDim.x;
      const int local_begin = chunk_begin + threadIdx.x * items_per_thread;
      const int local_end = min(local_begin + items_per_thread, chunk_end);
      float thread_sum = 0.0f;
      for (int vocab = local_begin; vocab < local_end; ++vocab) {
        thread_sum += workspace.probabilities[
            static_cast<int64_t>(step) * vocab_size + vocab] * total;
      }
      reduction[threadIdx.x] = thread_sum;
      __syncthreads();
      for (int stride = 1; stride < blockDim.x; stride *= 2) {
        const int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
          reduction[index] += reduction[index - stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        reduction[blockDim.x - 1] = 0.0f;
        workspace.block_index[blockIdx.x] = max(chunk_end - 1, 0);
      }
      __syncthreads();
      for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        const int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
          const float prefix = reduction[index - stride];
          reduction[index - stride] = reduction[index];
          reduction[index] += prefix;
        }
        __syncthreads();
      }
      float cumulative = reduction[threadIdx.x];
      for (int vocab = local_begin; vocab < local_end; ++vocab) {
        cumulative += workspace.probabilities[
            static_cast<int64_t>(step) * vocab_size + vocab] * total;
        if (cumulative >= local_threshold) {
          atomicMin(workspace.block_index + blockIdx.x, vocab);
          break;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        workspace.token_ids[step] = workspace.block_index[blockIdx.x];
        workspace.verify_input_ids[step + 1] = workspace.token_ids[step];
      }
    } else if (temperature < 1.0e-5f && blockIdx.x == 0 && threadIdx.x == 0) {
      workspace.token_ids[step] = static_cast<int64_t>(maximum_index);
      workspace.verify_input_ids[step + 1] = workspace.token_ids[step];
    }
    float confidence = 0.0f;
    if (blockIdx.x == 0) {
      for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        confidence = fmaf(
            as_float(workspace.hidden, static_cast<int64_t>(step) * hidden_size + column),
            as_float(weights.confidence_weight, column),
            confidence);
      }
      for (int column = threadIdx.x; column < rank; column += blockDim.x) {
        confidence = fmaf(
            as_float(weights.markov_w1, previous_token * static_cast<int64_t>(rank) + column),
            as_float(weights.confidence_weight, hidden_size + column),
            confidence);
      }
    }
    reduction[threadIdx.x] = confidence;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      confidence = reduction[0] + as_float(weights.confidence_bias, 0);
      workspace.confidence_logits[step] =
          static_cast<float>(static_cast<scalar_t>(confidence));
    }
    record_full_timestamp(workspace.trace, normalize_sample_phase, 1);
    if (step + 1 < steps) {
      cg::this_grid().sync();
    }
    record_full_timestamp(workspace.trace, normalize_sample_phase, 2);
    if (step + 1 < steps) {
      previous_token = workspace.token_ids[step];
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void dspark_full_kernel(
    const int64_t* draft_input_ids,
    const int64_t* first_prev_token,
    const scalar_t* target_hidden_states,
    const int64_t* position_ids,
    const float* uniforms,
    float temperature,
    int past_length,
    int context_length,
    int max_cache_length,
    int block_size,
    int hidden_size,
    int intermediate_size,
    int vocab_size,
    int rank,
    int feature_width,
    int num_layers,
    int query_heads,
    int kv_heads,
    int head_dim,
    float rms_epsilon,
    float rope_theta,
    CUTE_GRID_CONSTANT LmTcgenTmaAtomA const lm_tma_a,
    CUTE_GRID_CONSTANT LmTcgenTmaAtomB const lm_tma_b,
    FullWeights<scalar_t> weights,
    FullWorkspace<scalar_t> workspace) {
  int phase = 0;
  for (int step = global_thread_id(); step < block_size; step += global_thread_count()) {
    reinterpret_cast<unsigned long long*>(workspace.block_max)[step] =
        static_cast<unsigned long long>(pack_max_index(-3.402823466e+38F, vocab_size));
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    workspace.verify_input_ids[0] = first_prev_token[0];
  }
  const int padded_context_rows = ((context_length + 15) / 16) * 16;
  pad_rows(
      target_hidden_states,
      workspace.target_padded,
      context_length,
      padded_context_rows,
      feature_width,
      workspace.trace,
      phase++);
  linear_split_k(
      workspace.target_padded,
      weights.context_projection,
      workspace.context,
      context_length,
      feature_width,
      hidden_size,
      workspace.linear_partials,
      workspace.trace,
      phase);
  phase += 2;
  rms_norm_and_gather_embeddings(
      workspace.context,
      weights.context_norm,
      context_length,
      hidden_size,
      rms_epsilon,
      draft_input_ids,
      first_prev_token,
      weights.embedding,
      workspace.hidden,
      block_size,
      kLmTcgenRows,
      workspace.trace,
      phase++);

  const int q_width = query_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  for (int layer = 0; layer < num_layers; ++layer) {
    const LayerWeights<scalar_t>& current = weights.layers[layer];
    if (layer == 0) {
      rms_norm(
          workspace.hidden,
          current.input_norm,
          workspace.normalized,
          block_size,
          hidden_size,
          rms_epsilon,
          workspace.trace,
          phase++);
    }
    const LinearTask<scalar_t> qkv_tasks[] = {
        {workspace.normalized, current.q, workspace.q, block_size, hidden_size, q_width},
        {workspace.context, current.k, workspace.k_context, context_length, hidden_size, kv_width},
        {workspace.context, current.v, workspace.v_context, context_length, hidden_size, kv_width},
        {workspace.normalized, current.k, workspace.k_draft, block_size, hidden_size, kv_width},
        {workspace.normalized, current.v, workspace.v_draft, block_size, hidden_size, kv_width},
    };
    linear_group_split_k(
        qkv_tasks,
        5,
        2,
        workspace.linear_partials,
        workspace.trace,
        phase);
    phase += 2;
    head_rms_norm_rope_qkv(
        workspace.q,
        workspace.k_context,
        workspace.k_draft,
        current.q_norm,
        current.k_norm,
        position_ids,
        context_length,
        block_size,
        query_heads,
        kv_heads,
        head_dim,
        rms_epsilon,
        rope_theta,
        workspace.trace,
        phase++);
    attention(
        workspace.q,
        workspace.cache_k,
        workspace.cache_v,
        workspace.k_context,
        workspace.v_context,
        workspace.k_draft,
        workspace.v_draft,
        workspace.attention,
        layer,
        block_size,
        past_length,
        context_length,
        max_cache_length,
        query_heads,
        kv_heads,
        head_dim,
        workspace.trace,
        phase++);
    linear_split_k(
        workspace.attention,
        current.o,
        workspace.projection,
        block_size,
        q_width,
        hidden_size,
        workspace.linear_partials,
        workspace.trace,
        phase);
    phase += 2;
    add_and_rms_norm(
        workspace.hidden,
        workspace.projection,
        current.post_norm,
        workspace.normalized,
        block_size,
        hidden_size,
        rms_epsilon,
        workspace.trace,
        phase++);
    const LinearTask<scalar_t> gate_up_tasks[] = {
        {
            workspace.normalized,
            current.gate,
            workspace.gate,
            block_size,
            hidden_size,
            intermediate_size,
        },
        {
            workspace.normalized,
            current.up,
            workspace.up,
            block_size,
            hidden_size,
            intermediate_size,
        },
    };
    linear_group_split_k<scalar_t, true>(
        gate_up_tasks,
        2,
        8,
        workspace.linear_partials,
        workspace.trace,
        phase);
    phase += 2;
    linear_split_k(
        workspace.gate,
        current.down,
        workspace.projection,
        block_size,
        intermediate_size,
        hidden_size,
        workspace.linear_partials,
        workspace.trace,
        phase);
    phase += 2;
    const bool has_next_layer = layer + 1 < num_layers;
    const scalar_t* next_norm_weight = has_next_layer
        ? weights.layers[layer + 1].input_norm
        : weights.final_norm;
    scalar_t* next_normalized =
        has_next_layer ? workspace.normalized : workspace.hidden;
    add_and_rms_norm(
        workspace.hidden,
        workspace.projection,
        next_norm_weight,
        next_normalized,
        block_size,
        hidden_size,
        rms_epsilon,
        workspace.trace,
        phase++);
  }
  linear(
      workspace.hidden,
      weights.lm_head,
      workspace.base_logits,
      block_size,
      hidden_size,
      vocab_size,
      workspace.trace,
      phase++,
      lm_tma_a,
      lm_tma_b);
  proposal_tail(
      weights,
      workspace,
      draft_input_ids,
      first_prev_token,
      uniforms,
      temperature,
      block_size,
      vocab_size,
      hidden_size,
      rank,
      phase);
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> dspark_full_cuda(
    const torch::Tensor& draft_input_ids,
    const torch::Tensor& first_prev_token,
    const torch::Tensor& target_hidden_states,
    const torch::Tensor& position_ids,
    const torch::Tensor& uniforms,
    double temperature,
    int64_t past_length,
    double rms_epsilon,
    double rope_theta,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    const std::vector<torch::Tensor>& top_weights,
    const std::vector<torch::Tensor>& layer_weights,
    const std::vector<torch::Tensor>& workspace) {
  check_cuda_contiguous(draft_input_ids, "draft_input_ids");
  check_cuda_contiguous(first_prev_token, "first_prev_token");
  check_cuda_contiguous(target_hidden_states, "target_hidden_states");
  check_cuda_contiguous(position_ids, "position_ids");
  check_cuda_contiguous(uniforms, "uniforms");
  TORCH_CHECK(draft_input_ids.scalar_type() == torch::kInt64, "draft ids must be int64");
  TORCH_CHECK(
      first_prev_token.scalar_type() == torch::kInt64 && first_prev_token.numel() == 1,
      "first_prev_token must contain one int64 token");
  TORCH_CHECK(position_ids.scalar_type() == torch::kInt64, "position ids must be int64");
  TORCH_CHECK(uniforms.scalar_type() == torch::kFloat32, "uniforms must be float32");
  TORCH_CHECK(top_weights.size() == 9, "expected 9 top-level weights");
  TORCH_CHECK(layer_weights.size() % kWeightsPerLayer == 0, "invalid layer weights");
  int num_layers = static_cast<int>(layer_weights.size() / kWeightsPerLayer);
  TORCH_CHECK(num_layers > 0 && num_layers <= 5, "full kernel supports one to five layers");
  TORCH_CHECK(workspace.size() == 27, "expected 27 workspace tensors");
  for (const auto& tensor : top_weights) {
    check_cuda_contiguous(tensor, "top weight");
    TORCH_CHECK(tensor.scalar_type() == target_hidden_states.scalar_type(), "weight dtype mismatch");
  }
  for (const auto& tensor : layer_weights) {
    check_cuda_contiguous(tensor, "layer weight");
    TORCH_CHECK(tensor.scalar_type() == target_hidden_states.scalar_type(), "weight dtype mismatch");
  }
  for (const auto& tensor : workspace) {
    check_cuda_contiguous(tensor, "workspace");
  }

  int block_size = static_cast<int>(draft_input_ids.numel());
  int context_length = static_cast<int>(target_hidden_states.size(0));
  int feature_width = static_cast<int>(target_hidden_states.size(1));
  int hidden_size = static_cast<int>(top_weights[0].size(1));
  int vocab_size = static_cast<int>(top_weights[0].size(0));
  int intermediate_size = static_cast<int>(layer_weights[8].size(0));
  int rank = static_cast<int>(top_weights[5].size(1));
  int max_cache_length = static_cast<int>(workspace[17].size(1));
  TORCH_CHECK(position_ids.numel() == context_length + block_size, "position length mismatch");
  TORCH_CHECK(past_length >= 0 && past_length + context_length <= max_cache_length,
              "cache capacity exceeded");
  TORCH_CHECK(workspace[15].numel() == block_size, "token workspace shape mismatch");
  TORCH_CHECK(
      workspace[23].numel() >=
          static_cast<int64_t>(((context_length + 15) / 16) * 16) * feature_width,
      "padded target workspace is too small");

  const c10::cuda::CUDAGuard device_guard(target_hidden_states.device());
  const int device = target_hidden_states.get_device();
  int blocks = 0;
  int cooperative = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(blocks > 0 && cooperative != 0, "cooperative launch unsupported");
  const auto stream = at::cuda::getCurrentCUDAStream(device);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      target_hidden_states.scalar_type(),
      "dspark_full_kernel",
      [&] {
        FullWeights<scalar_t> weights{};
        weights.embedding = top_weights[0].data_ptr<scalar_t>();
        weights.context_projection = top_weights[1].data_ptr<scalar_t>();
        weights.context_norm = top_weights[2].data_ptr<scalar_t>();
        weights.final_norm = top_weights[3].data_ptr<scalar_t>();
        weights.lm_head = top_weights[4].data_ptr<scalar_t>();
        weights.markov_w1 = top_weights[5].data_ptr<scalar_t>();
        weights.markov_w2 = top_weights[6].data_ptr<scalar_t>();
        weights.confidence_weight = top_weights[7].data_ptr<scalar_t>();
        weights.confidence_bias = top_weights[8].data_ptr<scalar_t>();
        for (int layer = 0; layer < num_layers; ++layer) {
          const int offset = layer * kWeightsPerLayer;
          weights.layers[layer] = {
              layer_weights[offset + 0].data_ptr<scalar_t>(),
              layer_weights[offset + 1].data_ptr<scalar_t>(),
              layer_weights[offset + 2].data_ptr<scalar_t>(),
              layer_weights[offset + 3].data_ptr<scalar_t>(),
              layer_weights[offset + 4].data_ptr<scalar_t>(),
              layer_weights[offset + 5].data_ptr<scalar_t>(),
              layer_weights[offset + 6].data_ptr<scalar_t>(),
              layer_weights[offset + 7].data_ptr<scalar_t>(),
              layer_weights[offset + 8].data_ptr<scalar_t>(),
              layer_weights[offset + 9].data_ptr<scalar_t>(),
              layer_weights[offset + 10].data_ptr<scalar_t>(),
          };
        }
        FullWorkspace<scalar_t> work{
            workspace[0].data_ptr<scalar_t>(),
            workspace[1].data_ptr<scalar_t>(),
            workspace[2].data_ptr<scalar_t>(),
            workspace[3].data_ptr<scalar_t>(),
            workspace[4].data_ptr<scalar_t>(),
            workspace[5].data_ptr<scalar_t>(),
            workspace[6].data_ptr<scalar_t>(),
            workspace[7].data_ptr<scalar_t>(),
            workspace[8].data_ptr<scalar_t>(),
            workspace[9].data_ptr<scalar_t>(),
            workspace[10].data_ptr<scalar_t>(),
            workspace[11].data_ptr<scalar_t>(),
            workspace[12].data_ptr<scalar_t>(),
            workspace[13].data_ptr<float>(),
            workspace[14].data_ptr<float>(),
            workspace[15].data_ptr<int64_t>(),
            workspace[16].data_ptr<float>(),
            workspace[17].data_ptr<scalar_t>(),
            workspace[18].data_ptr<scalar_t>(),
            workspace[19].data_ptr<float>(),
            workspace[20].data_ptr<int>(),
            workspace[21].data_ptr<float>(),
            workspace[22].data_ptr<float>(),
            workspace[23].data_ptr<scalar_t>(),
            workspace[24].data_ptr<int64_t>(),
            workspace[25].data_ptr<float>(),
            workspace[26].data_ptr<int64_t>(),
        };
        auto lm_weight_tensor = cute::make_tensor(
            cute::make_gmem_ptr(
                reinterpret_cast<const LmTcgenType*>(weights.lm_head)),
            LmTcgenWeightLayout{});
        auto lm_input_tensor = cute::make_tensor(
            cute::make_gmem_ptr(
                reinterpret_cast<const LmTcgenType*>(work.hidden)),
            LmTcgenInputLayout{});
        LmTcgenTmaAtomA lm_tma_a = cute::make_tma_atom(
            cute::SM90_TMA_LOAD{},
            lm_weight_tensor,
            LmTcgenSmemLayoutA{}(
                cute::_, cute::_, cute::_, cute::Int<0>{}),
            cute::select<0, 2>(LmTcgenMmaTiler{}));
        LmTcgenTmaAtomB lm_tma_b = cute::make_tma_atom(
            cute::SM90_TMA_LOAD{},
            lm_input_tensor,
            LmTcgenSmemLayoutB{}(
                cute::_, cute::_, cute::_, cute::Int<0>{}),
            cute::select<1, 2>(LmTcgenMmaTiler{}));
        int compute_capability_major = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(
            &compute_capability_major, cudaDevAttrComputeCapabilityMajor, device));
        const int dynamic_shared_bytes =
            compute_capability_major >= 10 ? kLmTcgenDynamicSharedBytes : 0;
        if (dynamic_shared_bytes > 0) {
          C10_CUDA_CHECK(cudaFuncSetAttribute(
              dspark_full_kernel<scalar_t>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              dynamic_shared_bytes));
        }
        int blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            dspark_full_kernel<scalar_t>,
            kThreads,
            dynamic_shared_bytes));
        TORCH_CHECK(blocks_per_sm >= 1, "full kernel cannot reside on the GPU");
        auto* draft_ptr = draft_input_ids.data_ptr<int64_t>();
        auto* first_prev_ptr = first_prev_token.data_ptr<int64_t>();
        auto* target_ptr = target_hidden_states.data_ptr<scalar_t>();
        auto* position_ptr = position_ids.data_ptr<int64_t>();
        auto* uniform_ptr = uniforms.data_ptr<float>();
        float temperature_value = static_cast<float>(temperature);
        int past = static_cast<int>(past_length);
        float epsilon = static_cast<float>(rms_epsilon);
        float theta = static_cast<float>(rope_theta);
        int query_heads = static_cast<int>(num_attention_heads);
        int kv_heads = static_cast<int>(num_key_value_heads);
        int head_dimension = static_cast<int>(head_dim);
        void* arguments[] = {
            &draft_ptr,
            &first_prev_ptr,
            &target_ptr,
            &position_ptr,
            &uniform_ptr,
            &temperature_value,
            &past,
            &context_length,
            &max_cache_length,
            &block_size,
            &hidden_size,
            &intermediate_size,
            &vocab_size,
            &rank,
            &feature_width,
            &num_layers,
            &query_heads,
            &kv_heads,
            &head_dimension,
            &epsilon,
            &theta,
            &lm_tma_a,
            &lm_tma_b,
            &weights,
            &work,
        };
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(dspark_full_kernel<scalar_t>),
            dim3(blocks),
            dim3(kThreads),
            arguments,
            dynamic_shared_bytes,
            stream.stream()));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      workspace[1],
      workspace[12],
      workspace[13],
      workspace[14],
      workspace[15],
      workspace[16],
      workspace[17],
      workspace[18],
      workspace[24],
      workspace[26],
  };
}
