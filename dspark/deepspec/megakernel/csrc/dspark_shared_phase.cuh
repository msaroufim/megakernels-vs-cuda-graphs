// Self-contained shared-expert FP8 W13 phase bodies for the DeepSeek-V4
// DSpark megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// Frozen semantics (validation ledger #56/#67): five per-row FP32
// accumulators; per-128-element quantization blocks where inner products are
// accumulated first and the E8M0 activation/weight scales are applied through
// one outer fmaf per block; weight scales are shared per (128-output tile,
// 128-element block).
#pragma once

#include "dspark_tmem.cuh"

#include "dspark_w13_phase.cuh"
#include "dspark_batch.h"

namespace dspark_shared {

constexpr int kHidden = dspark_w13::kHidden;
constexpr int kQuantBlock = 128;
constexpr int kIntermediate = dspark_w13::kIntermediate;
constexpr int kDraftBlock = 5;
constexpr int kOutputTile = 128;
constexpr int kCombinedTiles = 2 * kIntermediate / kOutputTile;
constexpr int kMmaBlocks = kHidden / 32;         // K=32 MMA blocks per row
constexpr int kGroupBlocks = kQuantBlock / 32;   // MMA blocks per scale block
constexpr int kGroups = kHidden / kQuantBlock;   // scale blocks per row

struct Args {
  const cutlass::float_e4m3_t* input;
  const uint8_t* input_scales;
  __nv_bfloat16* shared_w13;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  int weight_base;
  int w1_slot;
  int w3_slot;
  int w1_scale_slot;
  int w3_scale_slot;
  uint32_t begin;
  uint32_t end;
};

struct ItemPlan {
  const cutlass::float_e4m3_t* weight;
  const uint8_t* weight_scales;
  int branch;
  int output_base;
  // Batched serving: the phase already carried a `batch *` factor, so one
  // item is one (batch element, combined tile) pair. The shared expert's
  // 16 MiB W1/W3 pair is re-read per element -- cheap next to retemplating
  // the atom, and 0 at batch 1.
  int element;
};

__device__ __forceinline__ ItemPlan item_plan(const Args& args, uint32_t item) {
  ItemPlan plan;
  const int combined_tile = static_cast<int>(item) % kCombinedTiles;
  plan.element = static_cast<int>(item) / kCombinedTiles;
  plan.branch = combined_tile / (kIntermediate / kOutputTile);
  plan.output_base =
      (combined_tile % (kIntermediate / kOutputTile)) * kOutputTile;
  const int weight_slot =
      plan.branch == 0 ? args.w1_slot : args.w3_slot;
  const int scale_slot =
      plan.branch == 0 ? args.w1_scale_slot : args.w3_scale_slot;
  plan.weight = reinterpret_cast<const cutlass::float_e4m3_t*>(
      args.weight_arena + args.weight_offsets[args.weight_base + weight_slot]);
  plan.weight_scales =
      args.weight_arena + args.weight_offsets[args.weight_base + scale_slot];
  return plan;
}

// Scalar body: exact retained #67/#68 semantics.
__device__ inline void execute_scalar(const Args& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    if (threadIdx.x >= kOutputTile) {
      continue;
    }
    const ItemPlan plan = item_plan(args, item);
    const int row_base = plan.element * kDraftBlock;
    const int output = plan.output_base + static_cast<int>(threadIdx.x);
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = 0; block < kGroups; ++block) {
      float inner[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      const int weight_base = output * kHidden + block * kQuantBlock;
#pragma unroll
      for (int offset = 0; offset < kQuantBlock; ++offset) {
        const float weight_value =
            static_cast<float>(plan.weight[weight_base + offset]);
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          inner[row] = fmaf(
              static_cast<float>(
                  args.input[(row_base + row) * kHidden
                         + block * kQuantBlock + offset]),
              weight_value,
              inner[row]);
        }
      }
      const float weight_scale = dspark_w13::decode_e8m0(
          plan.weight_scales[(output / kQuantBlock) * kGroups + block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = dspark_w13::decode_e8m0(
            args.input_scales[(row_base + row) * kGroups + block]);
        result[row] = fmaf(inner[row], input_scale * weight_scale, result[row]);
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.shared_w13[((row_base + row) * 2 + plan.branch) * kIntermediate
                      + output] = __float2bfloat16_rn(result[row]);
    }
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

using TcgenWeightType = cutlass::float_e4m3_t;
using TcgenInputType = cutlass::float_e4m3_t;
using TcgenTiledMma = decltype(cute::make_tiled_mma(
    cute::MMA_Traits<
        cute::SM100_MMA_F8F6F4_SS,
        TcgenWeightType,
        TcgenInputType,
        float,
        cute::C<128>,
        cute::C<8>,
        cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
        cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
        cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
        cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>{}));
using TcgenMmaShapeA = decltype(cute::partition_shape_A(
    TcgenTiledMma{},
    cute::make_shape(cute::Int<128>{}, cute::Int<32>{})));
using TcgenMmaShapeB = decltype(cute::partition_shape_B(
    TcgenTiledMma{},
    cute::make_shape(cute::Int<8>{}, cute::Int<32>{})));
using TcgenSmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<TcgenWeightType>{},
    TcgenMmaShapeA{}));
using TcgenSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<TcgenInputType>{},
    TcgenMmaShapeB{}));

// Warp-specialized pipeline (see dspark_w13_phase.cuh): an 8-deep cp.async
// weight-stage ring feeds one K=32 MMA per stage; four consecutive MMAs
// accumulate in one of four per-group TMEM slots (scaleC = 0,1,1,1), giving
// exactly one tensor-core inner product per 128-element quantization block,
// which then flows through the frozen outer FP32 fmaf scale chain. All five
// draft rows ride in tensor-core columns 0-4. Every barrier completes once
// per its ring round so all parity waits are single-lag by construction.
constexpr int kStages = 8;
constexpr int kAccumulatorSlots = 4;

struct TcgenSharedStorage {
  alignas(128)
      cute::ArrayEngine<
          TcgenWeightType, cute::cosize_v<TcgenSmemLayoutA>> a[kStages];
  alignas(128)
      cute::ArrayEngine<
          TcgenInputType, cute::cosize_v<TcgenSmemLayoutB>> b[kStages];
  alignas(16) float partial[2][128 * 8];
  alignas(16) cute::uint64_t stage_full[kStages];
  alignas(16) cute::uint64_t group_full[kAccumulatorSlots];
  alignas(16) cute::uint64_t group_free[kAccumulatorSlots];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <int kStagedSubtiles = 1>
__device__ inline void execute_tcgen(const Args& args) {
  static_assert(kStagedSubtiles == 1 || kStagedSubtiles == kGroupBlocks);
  // Four K32 tiles can share one producer handoff without changing the four
  // ordered MMAs or the per-K128 outer FP32 accumulation. The physical ring
  // and its group-completion reuse condition remain identical.
  using namespace cute;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<TcgenSmemLayoutB>);
  static_assert(
      sizeof(TcgenSharedStorage) <= dspark_w13::kSharedBytesBudget);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<TcgenSharedStorage*>(dynamic_shared_memory);
  TcgenTiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<128>{}, Int<8>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = threadIdx.x / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kAccumulatorSlots>(
        storage.group_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kAccumulatorSlots>(
        storage.group_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Rows 5..7 of every input stage stay zero; rows 0..4 are refilled per
  // block by cp.async. Generic-proxy stores need the async-proxy fence.
  for (int s = 0; s < kStages; ++s) {
    for (int i = threadIdx.x; i < kInputStageBytes;
         i += static_cast<int>(blockDim.x)) {
      reinterpret_cast<uint8_t*>(storage.b[s].begin())[i] = 0;
    }
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;
  uint64_t weight_descriptor[kStages];
  uint64_t input_descriptor[kStages];
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    weight_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.a[s].begin()),
            layout<0>(TcgenSmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(TcgenSmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int total_blocks =
      static_cast<int>(args.end - args.begin) * kMmaBlocks;

  if (warp >= 5) {
    // Staging crew. Stage s (block b) is reusable once the group holding the
    // MMA that consumed it (block b - kStages, group g - 2) has committed.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int first = 0; first < total_blocks; first += kStagedSubtiles) {
      const int s = first % kStages;
      if (first >= kStages) {
        const int consumed_group = (first - kStages) / kGroupBlocks;
        wait_barrier(
            storage.group_full[consumed_group % kAccumulatorSlots],
            (consumed_group / kAccumulatorSlots) & 1);
      }
      CUTE_UNROLL
      for (int subtile = 0; subtile < kStagedSubtiles; ++subtile) {
        const int b = first + subtile;
        const int slot = s + subtile;
        const ItemPlan plan = item_plan(
            args, args.begin + static_cast<uint32_t>(b / kMmaBlocks));
        const int block = b % kMmaBlocks;
        // A tile: 128 output rows x 32 e4m3 bytes, two 16-byte chunks per row.
        TcgenWeightType* stage_a = storage.a[slot].begin();
        for (int copy_index = lane; copy_index < kOutputTile * 2;
             copy_index += kStagingThreads) {
          const int output = copy_index >> 1;
          const int chunk = copy_index & 1;
          const cutlass::float_e4m3_t* source = plan.weight
              + static_cast<int64_t>(plan.output_base + output) * kHidden
              + block * 32 + chunk * 16;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              reinterpret_cast<uint8_t*>(stage_a) + output * 16 + chunk * 2048,
              source);
        }
        if (lane < 2 * kDraftBlock) {
          // Input rows 0..4: two 16-byte chunks each at atom bases 0 and 128.
          const int row = lane >> 1;
          const int chunk = lane & 1;
          const cutlass::float_e4m3_t* source =
              args.input + (plan.element * kDraftBlock + row) * kHidden
              + block * 32 + chunk * 16;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              reinterpret_cast<uint8_t*>(storage.b[slot].begin())
                  + chunk * 128 + row * 16,
              source);
        }
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    // MMA crew: scaleC=0 opens each scale group, then three chained
    // accumulating MMAs complete the 128-element inner product in TMEM.
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      const int group = b / kGroupBlocks;
      const int slot = group % kAccumulatorSlots;
      const int position = b % kGroupBlocks;
      if (b % kStagedSubtiles == 0) {
        wait_barrier(storage.stage_full[s], (b / kStages) & 1);
      }
      if (position == 0 && group >= kAccumulatorSlots) {
        wait_barrier(
            storage.group_free[slot],
            (group / kAccumulatorSlots - 1) & 1);
      }
      SM100_MMA_F8F6F4_SS::fma(
          weight_descriptor[s],
          input_descriptor[s],
          storage.tmem_base_ptr + slot * 8,
          position == 0 ? 0u : 1u,
          instruction_descriptor);
      if (position == kGroupBlocks - 1) {
        cutlass::arch::umma_arrive(&storage.group_full[slot]);
      }
    }
  } else {
    // Accumulator crew (warps 0-3): one drain per scale group into the
    // frozen five-row outer FP32 fmaf scale chain.
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int slot) {
      auto view = tmem_accumulator;
      view.data() = storage.tmem_base_ptr + slot * 8;
      return thread_copy.partition_S(view);
    };
    auto thread_tmem_0 = thread_source(0);
    auto thread_tmem_1 = thread_source(1);
    auto thread_tmem_2 = thread_source(2);
    auto thread_tmem_3 = thread_source(3);
    auto partial_layout = make_layout(
        make_shape(Int<128>{}, Int<8>{}),
        make_stride(Int<1>{}, Int<128>{}));
    auto shared_partial_low = make_tensor(
        make_smem_ptr(storage.partial[0]), partial_layout);
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));

    ItemPlan plan = {};
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int total_groups = total_blocks / kGroupBlocks;
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group % 2;
      const int block = group % kGroups;
      if (block == 0) {
        plan = item_plan(
            args, args.begin + static_cast<uint32_t>(group / kGroups));
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          result[row] = 0.0f;
        }
      }
      wait_barrier(
          storage.group_full[slot], (group / kAccumulatorSlots) & 1);
      if (slot == 0) {
        copy(tmem_to_register, thread_tmem_0, register_accumulator);
      } else if (slot == 1) {
        copy(tmem_to_register, thread_tmem_1, register_accumulator);
      } else if (slot == 2) {
        copy(tmem_to_register, thread_tmem_2, register_accumulator);
      } else {
        copy(tmem_to_register, thread_tmem_3, register_accumulator);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.group_free[slot]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const float weight_scale = dspark_w13::decode_e8m0(
          plan.weight_scales[
              ((plan.output_base + static_cast<int>(threadIdx.x))
               / kQuantBlock) * kGroups + block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = dspark_w13::decode_e8m0(
            args.input_scales[(plan.element * kDraftBlock + row) * kGroups
                              + block]);
        result[row] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row]);
      }
      if (block == kGroups - 1) {
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          args.shared_w13[
              ((plan.element * kDraftBlock + row) * 2 + plan.branch)
                  * kIntermediate
              + plan.output_base + threadIdx.x] = __float2bfloat16_rn(result[row]);
        }
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}
// ---------------------------------------------------------------------------
// N-WIDE FP8 MACHINERY (R4 batch-amortization campaign).
//
// Every FP8 dense band in the drafter (q_b, wo_b, q_a, draft_kv, shared_w13,
// shared_w2, main.projection) runs the pipeline above with the atom hard-wired
// to N=8, which holds at most 8 activation rows. At batch B the drafter has
// 5*B flat rows, so the bands that could not widen had to TILE OVER BATCH --
// re-streaming their whole weight matrix once per element.
//
// TcgenWideTraits<kN> is that same FP8 E4M3 x E4M3 K-major UMMA retemplated on
// the N mode, and execute_fp8_tcgen_wide is the (output_tile, split) body with
// a batch-INVARIANT item count that carries all 5*B rows on N. It is the FP8
// analogue of dspark_lm::execute_umma_bf16_batch, and the numerics argument is
// the same one: widening N adds independent output COLUMNS, so each output
// element keeps its exact K=32-block MMA sequence and its exact outer FP32
// fmaf scale chain, and results are bitwise identical to the N=8 body.
//
// The B SMEM layout for arbitrary N follows cute's layout algebra exactly as
// the LM band's does, with 16 e4m3 elements (not 8 bf16) per 16-byte atom row:
// layout<0>(SmemLayoutB) is (N,(16,2)):(16,(1,16N)), so element (row, k) sits
// at row*16 + k%16 + (k/16)*16N and one K=32 stage is N*32 bytes. N=8
// reproduces the frozen body's chunk*128 + row*16 formula exactly, which the
// static_asserts below pin down.
#if DSPARK_V4_BATCH > 1
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// TMEM columns for kAccumulatorSlots live N-wide accumulators: the allocator
// takes a power of two in [32, 512].
constexpr int tcgen_tmem_columns(int rows) {
  int columns = 32;
  while (columns < rows * kAccumulatorSlots) {
    columns *= 2;
  }
  return columns;
}

template <int kN>
struct TcgenWideTraits {
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Traits<
          cute::SM100_MMA_F8F6F4_SS,
          TcgenWeightType,
          TcgenInputType,
          float,
          cute::C<128>,
          cute::C<kN>,
          cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
          cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
          cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
          cute::integral_constant<cute::UMMA::ScaleIn,
                                  cute::UMMA::ScaleIn::One>>{}));
  using ShapeA = decltype(cute::partition_shape_A(
      TiledMma{}, cute::make_shape(cute::Int<128>{}, cute::Int<32>{})));
  using ShapeB = decltype(cute::partition_shape_B(
      TiledMma{}, cute::make_shape(cute::Int<kN>{}, cute::Int<32>{})));
  using SmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<TcgenWeightType>{}, ShapeA{}));
  using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<TcgenInputType>{}, ShapeB{}));
  static constexpr int kTmemColumns = tcgen_tmem_columns(kN);
};

template <int kN>
struct TcgenWideStorage {
  alignas(128) cute::ArrayEngine<
      TcgenWeightType,
      cute::cosize_v<typename TcgenWideTraits<kN>::SmemLayoutA>> a[kStages];
  alignas(128) cute::ArrayEngine<
      TcgenInputType,
      cute::cosize_v<typename TcgenWideTraits<kN>::SmemLayoutB>> b[kStages];
  alignas(16) float partial[2][128 * kN];
  // The running outer FP32 scale chain, one float per (drain lane, row).
  //
  // The frozen N=8 body keeps this in a `float result[kRows]` register array.
  // At N=40 that array is live across a whole item and it pushed the WHOLE
  // megakernel over the register cap: measured with ptxas -v, instantiating
  // the widened body anywhere took the single persistent kernel from 8 to 280
  // bytes of spill stores, which taxed every band in the program by ~0.24 ms
  // at batch 8 no matter which band was widened. Each drain thread owns its
  // own lane of this array (index row*128 + threadIdx.x), so there is no
  // sharing, no extra barrier, and the per-output fmaf order is untouched.
  alignas(16) float accumulate[128 * kN];
  alignas(16) cute::uint64_t stage_full[kStages];
  alignas(16) cute::uint64_t group_full[kAccumulatorSlots];
  alignas(16) cute::uint64_t group_free[kAccumulatorSlots];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

// Generic N-widened FP8 dense band. One item is one (output_tile, split)
// pair; the item count no longer carries a batch factor. Row indices are FLAT
// draft rows in [0, kRealRows), which is exactly how every (batch, block, ...)
// contiguous workspace region is laid out.
template <
    int kFeature,
    int kSplitCount,
    int kRows,
    int kRealRows,
    int kWidth,
    bool kBf16Out>
__device__ inline void execute_fp8_tcgen_wide(
    const cutlass::float_e4m3_t* input,
    const uint8_t* input_scales,
    const cutlass::float_e4m3_t* weight,
    const uint8_t* weight_scales,
    float* partials,
    __nv_bfloat16* bf16_output,
    uint32_t begin,
    uint32_t end) {
  using namespace cute;
  using Traits = TcgenWideTraits<kRows>;
  using Storage = TcgenWideStorage<kRows>;
  constexpr int kTmemColumns = Traits::kTmemColumns;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kSplitWidth = kFeature / kSplitCount;
  constexpr int kMmaBlocks = kSplitWidth / 32;
  constexpr int kGroupBlocks = 4;
  constexpr int kScaleBlocks = kFeature / kQuantBlock;
  constexpr int kBlocksPerSplit = kSplitWidth / kQuantBlock;
  constexpr int kGroupsPerItem = kMmaBlocks / kGroupBlocks;
  static_assert(kRows % 8 == 0 && kRows >= 8 && kRows <= 256,
                "tcgen05 f8 N-mode must be a multiple of 8 in [8, 256]");
  static_assert(kRealRows >= 1 && kRealRows <= kRows);
  static_assert(kSplitWidth % kQuantBlock == 0);
  static_assert(kMmaBlocks % kGroupBlocks == 0);
  static_assert(
      cute::cosize_v<typename Traits::SmemLayoutB> == kRows * 32,
      "one K=32 activation stage is exactly kRows*32 e4m3 elements");
  static_assert(sizeof(Storage) <= dspark_w13::kSharedBytesBudget,
                "widened FP8 staging must fit dynamic shared memory");

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<Storage*>(dynamic_shared_memory);
  typename Traits::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<128>{}, Int<kRows>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = threadIdx.x / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kAccumulatorSlots>(
        storage.group_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kAccumulatorSlots>(
        storage.group_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Every B stage is fully refilled by cp.async each block (pad rows included),
  // so the frozen body's zero prologue is unnecessary here.
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;
  uint64_t weight_descriptor[kStages];
  uint64_t input_descriptor[kStages];
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    weight_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.a[s].begin()),
            layout<0>(typename Traits::SmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(typename Traits::SmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int total_blocks = static_cast<int>(end - begin) * kMmaBlocks;

  if (warp >= 5) {
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      if (b >= kStages) {
        const int consumed_group = (b - kStages) / kGroupBlocks;
        wait_barrier(
            storage.group_full[consumed_group % kAccumulatorSlots],
            (consumed_group / kAccumulatorSlots) & 1);
      }
      const int item = static_cast<int>(begin) + b / kMmaBlocks;
      const int split = item % kSplitCount;
      const int output_tile = item / kSplitCount;
      const int block = b % kMmaBlocks;
      const int column_base = split * kSplitWidth + block * 32;
      cutlass::float_e4m3_t* stage_a = storage.a[s].begin();
      for (int copy_index = lane; copy_index < kOutputTile * 2;
           copy_index += kStagingThreads) {
        const int output = copy_index >> 1;
        const int chunk = copy_index & 1;
        const cutlass::float_e4m3_t* source = weight
            + static_cast<int64_t>(output_tile * kOutputTile + output)
                * kFeature
            + column_base + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(stage_a) + output * 16 + chunk * 2048,
            source);
      }
      for (int copy_index = lane; copy_index < 2 * kRows;
           copy_index += kStagingThreads) {
        const int row = copy_index >> 1;
        const int chunk = copy_index & 1;
        // Pad rows re-read the last real row: never stored, always in bounds.
        const int source_row = row < kRealRows ? row : kRealRows - 1;
        const cutlass::float_e4m3_t* source =
            input + static_cast<int64_t>(source_row) * kFeature
            + column_base + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(storage.b[s].begin())
                + chunk * 16 * kRows + row * 16,
            source);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      const int group = b / kGroupBlocks;
      const int slot = group % kAccumulatorSlots;
      const int position = b % kGroupBlocks;
      wait_barrier(storage.stage_full[s], (b / kStages) & 1);
      if (position == 0 && group >= kAccumulatorSlots) {
        wait_barrier(
            storage.group_free[slot],
            (group / kAccumulatorSlots - 1) & 1);
      }
      SM100_MMA_F8F6F4_SS::fma(
          weight_descriptor[s],
          input_descriptor[s],
          storage.tmem_base_ptr + slot * kRows,
          position == 0 ? 0u : 1u,
          instruction_descriptor);
      if (position == kGroupBlocks - 1) {
        cutlass::arch::umma_arrive(&storage.group_full[slot]);
      }
    }
  } else {
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int slot) {
      auto view = tmem_accumulator;
      view.data() = storage.tmem_base_ptr + slot * kRows;
      return thread_copy.partition_S(view);
    };
    auto thread_tmem_0 = thread_source(0);
    auto thread_tmem_1 = thread_source(1);
    auto thread_tmem_2 = thread_source(2);
    auto thread_tmem_3 = thread_source(3);
    auto partial_layout = make_layout(
        make_shape(Int<128>{}, Int<kRows>{}),
        make_stride(Int<1>{}, Int<128>{}));
    auto shared_partial_low = make_tensor(
        make_smem_ptr(storage.partial[0]), partial_layout);
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));

    int split = 0;
    int output_tile = 0;
    float* result = storage.accumulate + static_cast<int>(threadIdx.x);
    const int total_groups = total_blocks / kGroupBlocks;
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group % 2;
      const int block = group % kGroupsPerItem;
      if (block == 0) {
        const int item = static_cast<int>(begin) + group / kGroupsPerItem;
        split = item % kSplitCount;
        output_tile = item / kSplitCount;
#pragma unroll
        for (int row = 0; row < kRealRows; ++row) {
          result[row * 128] = 0.0f;
        }
      }
      wait_barrier(
          storage.group_full[slot], (group / kAccumulatorSlots) & 1);
      if (slot == 0) {
        copy(tmem_to_register, thread_tmem_0, register_accumulator);
      } else if (slot == 1) {
        copy(tmem_to_register, thread_tmem_1, register_accumulator);
      } else if (slot == 2) {
        copy(tmem_to_register, thread_tmem_2, register_accumulator);
      } else {
        copy(tmem_to_register, thread_tmem_3, register_accumulator);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.group_free[slot]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const int k_block = split * kBlocksPerSplit + block;
      const float weight_scale = dspark_w13::decode_e8m0(
          weight_scales[output_tile * kScaleBlocks + k_block]);
#pragma unroll
      for (int row = 0; row < kRealRows; ++row) {
        const float input_scale = dspark_w13::decode_e8m0(
            input_scales[row * kScaleBlocks + k_block]);
        result[row * 128] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row * 128]);
      }
      if (block == kGroupsPerItem - 1) {
#pragma unroll
        for (int row = 0; row < kRealRows; ++row) {
          if constexpr (kBf16Out) {
            bf16_output[
                static_cast<int64_t>(row) * kWidth
                + output_tile * kOutputTile + static_cast<int>(threadIdx.x)] =
                __float2bfloat16_rn(result[row * 128]);
          } else {
            partials[
                (static_cast<int64_t>(row) * kWidth
                 + output_tile * kOutputTile + static_cast<int>(threadIdx.x))
                    * kSplitCount + split] = result[row * 128];
          }
        }
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}


// N-WIDENED shared_w13 (R4 band 7). The generic body above cannot serve this
// band because its weight pointer is item-dependent: a combined tile selects
// either the W1 or the W3 arena slot. Everything else is identical, so this
// is execute_tcgen with the element decode removed (the item space is the
// batch-invariant kCombinedTiles) and the row loop widened onto the N mode.
template <int kRows, int kRealRows>
__device__ inline void execute_tcgen_wide(const Args& args) {
  using namespace cute;
  using Traits = TcgenWideTraits<kRows>;
  using Storage = TcgenWideStorage<kRows>;
  constexpr int kTmemColumns = Traits::kTmemColumns;
  constexpr int kStagingThreads = 3 * 32;
  static_assert(kRows % 8 == 0 && kRows >= 8 && kRows <= 256);
  static_assert(kRealRows >= 1 && kRealRows <= kRows);
  static_assert(
      cute::cosize_v<typename Traits::SmemLayoutB> == kRows * 32,
      "one K=32 activation stage is exactly kRows*32 e4m3 elements");
  static_assert(sizeof(Storage) <= dspark_w13::kSharedBytesBudget);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<Storage*>(dynamic_shared_memory);
  typename Traits::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<128>{}, Int<kRows>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = threadIdx.x / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kAccumulatorSlots>(
        storage.group_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kAccumulatorSlots>(
        storage.group_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;
  uint64_t weight_descriptor[kStages];
  uint64_t input_descriptor[kStages];
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    weight_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.a[s].begin()),
            layout<0>(typename Traits::SmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(typename Traits::SmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int total_blocks =
      static_cast<int>(args.end - args.begin) * kMmaBlocks;

  if (warp >= 5) {
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      if (b >= kStages) {
        const int consumed_group = (b - kStages) / kGroupBlocks;
        wait_barrier(
            storage.group_full[consumed_group % kAccumulatorSlots],
            (consumed_group / kAccumulatorSlots) & 1);
      }
      const ItemPlan plan = item_plan(
          args, args.begin + static_cast<uint32_t>(b / kMmaBlocks));
      const int block = b % kMmaBlocks;
      TcgenWeightType* stage_a = storage.a[s].begin();
      for (int copy_index = lane; copy_index < kOutputTile * 2;
           copy_index += kStagingThreads) {
        const int output = copy_index >> 1;
        const int chunk = copy_index & 1;
        const cutlass::float_e4m3_t* source = plan.weight
            + static_cast<int64_t>(plan.output_base + output) * kHidden
            + block * 32 + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(stage_a) + output * 16 + chunk * 2048,
            source);
      }
      for (int copy_index = lane; copy_index < 2 * kRows;
           copy_index += kStagingThreads) {
        const int row = copy_index >> 1;
        const int chunk = copy_index & 1;
        const int source_row = row < kRealRows ? row : kRealRows - 1;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(storage.b[s].begin())
                + chunk * 16 * kRows + row * 16,
            args.input + static_cast<int64_t>(source_row) * kHidden
                + block * 32 + chunk * 16);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      const int group = b / kGroupBlocks;
      const int slot = group % kAccumulatorSlots;
      const int position = b % kGroupBlocks;
      wait_barrier(storage.stage_full[s], (b / kStages) & 1);
      if (position == 0 && group >= kAccumulatorSlots) {
        wait_barrier(
            storage.group_free[slot],
            (group / kAccumulatorSlots - 1) & 1);
      }
      SM100_MMA_F8F6F4_SS::fma(
          weight_descriptor[s],
          input_descriptor[s],
          storage.tmem_base_ptr + slot * kRows,
          position == 0 ? 0u : 1u,
          instruction_descriptor);
      if (position == kGroupBlocks - 1) {
        cutlass::arch::umma_arrive(&storage.group_full[slot]);
      }
    }
  } else {
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int slot) {
      auto view = tmem_accumulator;
      view.data() = storage.tmem_base_ptr + slot * kRows;
      return thread_copy.partition_S(view);
    };
    auto thread_tmem_0 = thread_source(0);
    auto thread_tmem_1 = thread_source(1);
    auto thread_tmem_2 = thread_source(2);
    auto thread_tmem_3 = thread_source(3);
    auto partial_layout = make_layout(
        make_shape(Int<128>{}, Int<kRows>{}),
        make_stride(Int<1>{}, Int<128>{}));
    auto shared_partial_low = make_tensor(
        make_smem_ptr(storage.partial[0]), partial_layout);
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));

    ItemPlan plan = {};
    float* result = storage.accumulate + static_cast<int>(threadIdx.x);
    const int total_groups = total_blocks / kGroupBlocks;
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group % 2;
      const int block = group % kGroups;
      if (block == 0) {
        plan = item_plan(
            args, args.begin + static_cast<uint32_t>(group / kGroups));
#pragma unroll
        for (int row = 0; row < kRealRows; ++row) {
          result[row * 128] = 0.0f;
        }
      }
      wait_barrier(
          storage.group_full[slot], (group / kAccumulatorSlots) & 1);
      if (slot == 0) {
        copy(tmem_to_register, thread_tmem_0, register_accumulator);
      } else if (slot == 1) {
        copy(tmem_to_register, thread_tmem_1, register_accumulator);
      } else if (slot == 2) {
        copy(tmem_to_register, thread_tmem_2, register_accumulator);
      } else {
        copy(tmem_to_register, thread_tmem_3, register_accumulator);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.group_free[slot]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const float weight_scale = dspark_w13::decode_e8m0(
          plan.weight_scales[
              ((plan.output_base + static_cast<int>(threadIdx.x))
               / kQuantBlock) * kGroups + block]);
#pragma unroll
      for (int row = 0; row < kRealRows; ++row) {
        const float input_scale = dspark_w13::decode_e8m0(
            args.input_scales[row * kGroups + block]);
        result[row * 128] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row * 128]);
      }
      if (block == kGroups - 1) {
#pragma unroll
        for (int row = 0; row < kRealRows; ++row) {
          args.shared_w13[
              (static_cast<int64_t>(row) * 2 + plan.branch) * kIntermediate
              + plan.output_base + threadIdx.x] =
              __float2bfloat16_rn(result[row * 128]);
        }
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}

#endif  // __CUDA_ARCH__ >= 1000
#endif  // DSPARK_V4_BATCH > 1

#endif  // __CUDA_ARCH__ >= 1000

}  // namespace dspark_shared
