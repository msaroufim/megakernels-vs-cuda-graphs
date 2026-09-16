// Self-contained dense attention-projection phase bodies (q_b, wo_a, wo_b)
// for the DeepSeek-V4 DSpark megakernel, compiled by both the production
// kernel and the phase microbenchmark (see dspark_w13_phase.cuh for the
// pattern rationale).
//
// Three bands, each ~0.3-0.5 ms x 3 layers on the 14.3 ms critical path:
//   q_b   FP8 GEMV 1024 -> 32768, 5 rows; E4M3 weights with per-(128-output,
//         128-K) E8M0 block scales; item = one 128-output tile (256/layer).
//   wo_a  BF16 GEMV per (group, rank_tile, row): 8 groups x (4096 -> 1024)
//         with a per-(row, 128-rank) amax FP8 quantization epilogue;
//         item = (group, rank_tile, row) triple (320/layer, #102 retile).
//   wo_b  FP8 GEMV 8192 -> 4096 split-K=4, 5 rows, FP32 split partials;
//         item = (output_tile, split) pair (128/layer).
//
// The execute_*_reference bodies are verbatim extractions of the #run-105-era
// production bodies (pure code motion; stage./task. became args.): each is
// its own oracle in the bench lane.
#pragma once

#include "dspark_tmem.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

// TCGen05 candidate precedents: the FP8 E4M3 x E4M3 UMMA pipeline with
// per-128-block scale chaining (dspark_shared_phase.cuh) and the BF16
// M=128 x N=8 UMMA ring (dspark_lm_phase.cuh). Both headers are already
// compiled by every TU that compiles this one.
#include "dspark_shared_phase.cuh"
#include "dspark_lm_phase.cuh"
#include "dspark_batch.h"

namespace dspark_proj {

constexpr int kDraftBlock = 5;
// Batched serving (R3): q_b (33.5 MiB), wo_a (33.5 MiB) and wo_b (33.5 MiB)
// tile OVER BATCH -- their phases carry a `batch *` factor and one item is
// one (batch element, ...) pair, so element e's rows begin e * kDraftBlock
// further into every (batch, block, ...) workspace region. kBatch == 1 folds
// every added term away.
constexpr int kBatch = dspark_batch::kBatch;
// R4: the bands that N-WIDEN instead carry every flat draft row on the
// tensor-core N mode. tcgen05's N must be a multiple of 8, so batch 4 (20
// rows) runs a padded atom; pad columns re-read the last real row and are
// never stored, exactly as the frozen bodies pad 5 -> 8.
constexpr int kRealRows = dspark_batch::kRows;            // 5 * batch
constexpr int kWideRows = dspark_batch::kWideRows;
constexpr int kQuantBlock = 128;   // == kMainQuantBlock
constexpr int kOutputTile = 128;   // == kMainOutputTile
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;

// Bit-exact E8M0 decode, verbatim from dspark_v4_kernel.cu (#71/#72).
__device__ __forceinline__ float decode_e8m0(uint8_t bits) {
  const uint32_t exponent = bits;
  const uint32_t float_bits =
      exponent == 0 ? 0x00400000U : exponent << 23;
  return __uint_as_float(float_bits);
}

// ---------------------------------------------------------------------------
// q_b: query up-projection, FP8 1024 -> 64*512.
// ---------------------------------------------------------------------------

constexpr int kQbRank = 1024;
constexpr int kQbWidth = 64 * 512;
constexpr int kQbBlocks = kQbRank / kQuantBlock;  // 8 scale blocks per row
constexpr int kQbTiles = kQbWidth / kOutputTile;  // 256 items per element

struct QbArgs {
  // 5 x 1024 E4M3 activations + 5 x 8 E8M0 scales (q_lora quantized).
  const cutlass::float_e4m3_t* q_lora_quantized;
  const uint8_t* q_lora_scales;
  // 32768 x 1024 E4M3 weights, per-(128-output, 128-K) E8M0 block scales.
  const cutlass::float_e4m3_t* wqb;
  const uint8_t* wqb_scales;
  // 5 x 32768 BF16 output.
  __nv_bfloat16* query_projection;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_qb_projection from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_qb_reference(const QbArgs& args) {
  constexpr int kRank = kQbRank;
  constexpr int kQueryWidth = kQbWidth;
  constexpr int kBlocks = kRank / kQuantBlock;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int output_column = static_cast<int>(item) * kOutputTile + threadIdx.x;
    if (threadIdx.x >= kOutputTile || output_column >= kQueryWidth) {
      continue;
    }
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k_block = 0; k_block < kBlocks; ++k_block) {
      const int weight_base = output_column * kRank + k_block * kQuantBlock;
      float inner[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int offset = 0; offset < kQuantBlock; offset += 16) {
        const uint4 packed_weights = *reinterpret_cast<const uint4*>(
            args.wqb + weight_base + offset);
        const cutlass::float_e4m3_t* weight_values =
            reinterpret_cast<const cutlass::float_e4m3_t*>(&packed_weights);
#pragma unroll
        for (int vector_offset = 0; vector_offset < 16; ++vector_offset) {
          const float weight_value =
              static_cast<float>(weight_values[vector_offset]);
#pragma unroll
          for (int row = 0; row < kDraftBlock; ++row) {
            inner[row] = fmaf(
                static_cast<float>(args.q_lora_quantized[
                    row * kRank + k_block * kQuantBlock
                    + offset + vector_offset]),
                weight_value,
                inner[row]);
          }
        }
      }
      const float weight_scale =
          decode_e8m0(args.wqb_scales[(output_column / 128) * kBlocks + k_block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale =
            decode_e8m0(args.q_lora_scales[row * kBlocks + k_block]);
        result[row] = fmaf(inner[row], input_scale * weight_scale, result[row]);
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.query_projection[row * kQueryWidth + output_column] =
          __float2bfloat16_rn(result[row]);
    }
  }
}

// ---------------------------------------------------------------------------
// wo_a: grouped attention-output down-projection, BF16 8 x (4096 -> 1024),
// with per-(row, 128-rank) FP8 quantization epilogue.
// ---------------------------------------------------------------------------

constexpr int kWoaGroups = 8;
constexpr int kWoaGroupInput = 4096;
constexpr int kWoaOutputRank = 1024;
constexpr int kWoaRankTiles = kWoaOutputRank / kOutputTile;  // 8
constexpr int kWoaItemsPerGroup = kWoaRankTiles * kDraftBlock;
// Relaxed wo_a tiling: (group, rank_tile) pairs, all draft rows per item.
constexpr int kWoaItemsPerElement = kWoaGroups * kWoaRankTiles;

struct WoaArgs {
  // 5 x 8 x 4096 BF16 grouped attention rows (inverse-RoPE'd upstream).
  const __nv_bfloat16* attention;
  // 8 x 1024 x 4096 BF16 weights (group-major, rank rows of 4096).
  const __nv_bfloat16* wo_a;
  // 5 x 8 x 1024 BF16 lora rows, laid out (row * 8 + group) * 1024 + rank.
  __nv_bfloat16* output_lora;
  // FP8 quantized copy + per-(row, group, rank_tile) E8M0 scales (5 x 64).
  cutlass::float_e4m3_t* output_lora_quantized;
  uint8_t* output_lora_scales;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_output_a_projection from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_woa_reference(
    const WoaArgs& args, float* shared) {
  // Retiled (#102): one task item is a (group, rank_tile, row) triple —
  // 320 claimable tiles instead of 64. Each (rank, row) chain keeps its
  // original ascending-column FMA order, and the per-(row, rank_tile) amax
  // quantization group is exactly the original 128-rank set, so both the
  // BF16 lora rows and the FP8 scale/quantized outputs are bitwise
  // unchanged. Weight and attention rows stream through 16-byte loads.
  constexpr int kGroups = kWoaGroups;
  constexpr int kGroupInput = kWoaGroupInput;
  constexpr int kOutputRank = kWoaOutputRank;
  constexpr int kRankTiles = kOutputRank / kOutputTile;
  constexpr int kItemsPerGroup = kRankTiles * kDraftBlock;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int group = static_cast<int>(item) / kItemsPerGroup;
    const int remainder = static_cast<int>(item) % kItemsPerGroup;
    const int rank_tile = remainder / kDraftBlock;
    const int row = remainder % kDraftBlock;
    const int rank = rank_tile * kOutputTile + threadIdx.x;
    if (group >= kGroups) {
      continue;
    }
    if (threadIdx.x < kOutputTile) {
      float result = 0.0f;
      const int64_t weight_base =
          (static_cast<int64_t>(group) * kOutputRank + rank) * kGroupInput;
      const int input_base = (row * kGroups + group) * kGroupInput;
      for (int column = 0; column < kGroupInput; column += 8) {
        const uint4 packed_weights = *reinterpret_cast<const uint4*>(
            args.wo_a + weight_base + column);
        const uint4 packed_inputs = *reinterpret_cast<const uint4*>(
            args.attention + input_base + column);
        const __nv_bfloat16* weight_values =
            reinterpret_cast<const __nv_bfloat16*>(&packed_weights);
        const __nv_bfloat16* input_values =
            reinterpret_cast<const __nv_bfloat16*>(&packed_inputs);
#pragma unroll
        for (int vector_offset = 0; vector_offset < 8; ++vector_offset) {
          result = fmaf(
              __bfloat162float(input_values[vector_offset]),
              __bfloat162float(weight_values[vector_offset]),
              result);
        }
      }
      args.output_lora[(row * kGroups + group) * kOutputRank + rank] =
          __float2bfloat16_rn(result);
    }
    if (threadIdx.x < kOutputTile) {
      shared[threadIdx.x] = fabsf(__bfloat162float(
          args.output_lora[(row * kGroups + group) * kOutputRank + rank]));
    } else {
      shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + delta]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float amax = fmaxf(shared[0], 1.0e-4f);
      const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
      const int encoded = exponent + 127;
      args.output_lora_scales[
          row * (kGroups * kRankTiles) + group * kRankTiles + rank_tile] =
          static_cast<uint8_t>(encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
      shared[0] = ldexpf(1.0f, exponent);
    }
    __syncthreads();
    if (threadIdx.x < kOutputTile) {
      const int index = (row * kGroups + group) * kOutputRank + rank;
      args.output_lora_quantized[index] =
          convert(__bfloat162float(args.output_lora[index]) / shared[0]);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// wo_b: attention-output up-projection, FP8 8192 -> 4096, split-K = 4.
// ---------------------------------------------------------------------------

constexpr int kWobInput = 8192;
constexpr int kWobHidden = 4096;
constexpr int kWobSplits = 4;
constexpr int kWobBlocks = kWobInput / kQuantBlock;          // 64
constexpr int kWobBlocksPerSplit = kWobBlocks / kWobSplits;  // 16
constexpr int kWobTiles = kWobHidden / kOutputTile;           // 32

struct WobArgs {
  // 5 x 8192 E4M3 activations (wo_a's quantized lora rows) + 5 x 64 scales.
  const cutlass::float_e4m3_t* output_lora_quantized;
  const uint8_t* output_lora_scales;
  // 4096 x 8192 E4M3 weights, per-(128-output, 128-K) E8M0 block scales.
  const cutlass::float_e4m3_t* wo_b;
  const uint8_t* wo_b_scales;
  // FP32 split partials, (row * 4096 + output) * 4 + split.
  float* output_partials;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_output_b_projection from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_wob_reference(const WobArgs& args) {
  constexpr int kInput = kWobInput;
  constexpr int kSplits = kWobSplits;
  constexpr int kBlocks = kInput / kQuantBlock;
  constexpr int kBlocksPerSplit = kBlocks / kSplits;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int split = static_cast<int>(item) % kSplits;
    const int output_tile = static_cast<int>(item) / kSplits;
    const int output_column = output_tile * kOutputTile + threadIdx.x;
    if (threadIdx.x >= kOutputTile || output_column >= kWobHidden) {
      continue;
    }
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = 0; block < kBlocksPerSplit; ++block) {
      const int k_block = split * kBlocksPerSplit + block;
      const int weight_base = output_column * kInput + k_block * kQuantBlock;
      float inner[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int offset = 0; offset < kQuantBlock; ++offset) {
        const float weight_value =
            static_cast<float>(args.wo_b[weight_base + offset]);
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          inner[row] = fmaf(
              static_cast<float>(args.output_lora_quantized[
                  row * kInput + k_block * kQuantBlock + offset]),
              weight_value,
              inner[row]);
        }
      }
      const float weight_scale = decode_e8m0(
          args.wo_b_scales[(output_column / kQuantBlock) * kBlocks + k_block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = decode_e8m0(
            args.output_lora_scales[row * kBlocks + k_block]);
        result[row] = fmaf(inner[row], input_scale * weight_scale, result[row]);
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.output_partials[(row * kWobHidden + output_column) * kSplits + split] =
          result[row];
    }
  }
}

// ---------------------------------------------------------------------------
// CANDIDATE bodies (relaxed program stage 3, dense-projection iteration).
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// q_b TCGen05 candidate: the dspark_shared FP8 pipeline verbatim, re-pointed
// at the q_b operands. One item is still one 128-output tile, so the body is
// a DROP-IN for the existing 256-item phase tiling. An 8-deep cp.async ring
// stages 128x32 E4M3 weight slots; warp 4 issues one K=32 UMMA per slot with
// four consecutive MMAs chained into one of four TMEM slots
// (scaleC = 0,1,1,1), giving exactly one tensor-core inner product per
// 128-element quantization block; warps 0-3 drain each block through the
// frozen outer FP32 fmaf scale chain. Numerics: the K=32-tree order inside a
// block differs from the serial 128-FMA chain, so outputs can land 1 BF16
// ULP off on adversarial synthetic scales (bit-exact on the real checkpoint
// for the identical dspark_shared structure, oracle #91); the bench reports
// the drift.
__device__ inline void execute_qb_tcgen(const QbArgs& args) {
  using namespace cute;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kStages = dspark_shared::kStages;                    // 8
  constexpr int kAccumulatorSlots = dspark_shared::kAccumulatorSlots; // 4
  constexpr int kMmaBlocks = kQbRank / 32;   // 32 K=32 blocks per item
  constexpr int kGroupBlocks = 4;            // MMA blocks per scale block
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<dspark_shared::TcgenSmemLayoutB>);
  static_assert(
      sizeof(dspark_shared::TcgenSharedStorage) <= kSharedBytesBudget);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<dspark_shared::TcgenSharedStorage*>(
      dynamic_shared_memory);
  dspark_shared::TcgenTiledMma tiled_mma;
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
            layout<0>(dspark_shared::TcgenSmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(dspark_shared::TcgenSmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int total_blocks =
      static_cast<int>(args.end - args.begin) * kMmaBlocks;

  if (warp >= 5) {
    // Staging crew. Stage s (block b) is reusable once the group holding the
    // MMA that consumed it (block b - kStages) has committed.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      if (b >= kStages) {
        const int consumed_group = (b - kStages) / kGroupBlocks;
        wait_barrier(
            storage.group_full[consumed_group % kAccumulatorSlots],
            (consumed_group / kAccumulatorSlots) & 1);
      }
      const int raw_item = static_cast<int>(args.begin) + b / kMmaBlocks;
      const int item = raw_item % kQbTiles;
      const int element = raw_item / kQbTiles;
      const int block = b % kMmaBlocks;
      // A tile: 128 output rows x 32 e4m3 bytes, two 16-byte chunks per row.
      cutlass::float_e4m3_t* stage_a = storage.a[s].begin();
      for (int copy_index = lane; copy_index < kOutputTile * 2;
           copy_index += kStagingThreads) {
        const int output = copy_index >> 1;
        const int chunk = copy_index & 1;
        const cutlass::float_e4m3_t* source = args.wqb
            + (item * kOutputTile + output) * kQbRank + block * 32
            + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(stage_a) + output * 16 + chunk * 2048,
            source);
      }
      if (lane < 2 * kDraftBlock) {
        // Input rows 0..4: two 16-byte chunks each at atom bases 0 and 128.
        const int row = lane >> 1;
        const int chunk = lane & 1;
        const cutlass::float_e4m3_t* source =
            args.q_lora_quantized
            + (element * kDraftBlock + row) * kQbRank + block * 32
            + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(storage.b[s].begin())
                + chunk * 128 + row * 16,
            source);
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
      wait_barrier(storage.stage_full[s], (b / kStages) & 1);
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

    int item = 0;
    int element = 0;
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int total_groups = total_blocks / kGroupBlocks;
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group % 2;
      const int block = group % kQbBlocks;
      if (block == 0) {
        const int raw_item = static_cast<int>(args.begin) + group / kQbBlocks;
        item = raw_item % kQbTiles;
        element = raw_item / kQbTiles;
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
      const float weight_scale =
          decode_e8m0(args.wqb_scales[item * kQbBlocks + block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = decode_e8m0(
            args.q_lora_scales[(element * kDraftBlock + row) * kQbBlocks
                               + block]);
        result[row] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row]);
      }
      if (block == kQbBlocks - 1) {
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          args.query_projection[
              (element * kDraftBlock + row) * kQbWidth + item * kOutputTile
              + static_cast<int>(threadIdx.x)] =
              __float2bfloat16_rn(result[row]);
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

// DeepGEMM-style BLOCK_K=128 staging experiment for q_b. Seven-point
// same-process confirmation on GB300 measured a repeatable but tiny 1.5-us
// full-proposal improvement, so this remains opt-in rather than becoming the
// shipped body. The arithmetic is
// deliberately identical to execute_qb_tcgen: each scale group still issues
// the same four ordered K=32 UMMAs with scaleC=(0,1,1,1), followed by the
// same outer FP32 fmaf. The only change is that those four operand tiles share
// one producer barrier and one two-stage ring slot. This transfers the useful
// BLOCK_K=128 structure without importing the standalone DeepGEMM tactic's
// 230-KiB launch-wide shared-memory envelope into every megakernel phase.
struct QbK128SharedStorage {
  static constexpr int kStages = 2;
  static constexpr int kSubtiles = 4;
  alignas(128) cute::ArrayEngine<
      dspark_shared::TcgenWeightType,
      cute::cosize_v<dspark_shared::TcgenSmemLayoutA> * kSubtiles>
      a[kStages];
  alignas(128) cute::ArrayEngine<
      dspark_shared::TcgenInputType,
      cute::cosize_v<dspark_shared::TcgenSmemLayoutB> * kSubtiles>
      b[kStages];
  alignas(16) float partial[2][128 * 8];
  alignas(16) cute::uint64_t stage_full[kStages];
  alignas(16) cute::uint64_t group_full[dspark_shared::kAccumulatorSlots];
  alignas(16) cute::uint64_t group_free[dspark_shared::kAccumulatorSlots];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

__device__ inline void execute_qb_k128_ring(const QbArgs& args) {
  using namespace cute;
  constexpr int kStages = QbK128SharedStorage::kStages;
  constexpr int kSubtiles = QbK128SharedStorage::kSubtiles;
  constexpr int kAccumulatorSlots = dspark_shared::kAccumulatorSlots;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kWeightSubtileBytes =
      static_cast<int>(cute::cosize_v<dspark_shared::TcgenSmemLayoutA>);
  constexpr int kInputSubtileBytes =
      static_cast<int>(cute::cosize_v<dspark_shared::TcgenSmemLayoutB>);
  static_assert(kWeightSubtileBytes == kOutputTile * 32);
  static_assert(kInputSubtileBytes == 8 * 32);
  static_assert(sizeof(QbK128SharedStorage) <= kSharedBytesBudget);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<QbK128SharedStorage*>(dynamic_shared_memory);
  dspark_shared::TcgenTiledMma tiled_mma;
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
  for (int index = threadIdx.x;
       index < kStages * kSubtiles * kInputSubtileBytes;
       index += static_cast<int>(blockDim.x)) {
    reinterpret_cast<uint8_t*>(storage.b)[index] = 0;
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
            layout<0>(dspark_shared::TcgenSmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(dspark_shared::TcgenSmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);
  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_groups = item_count * kQbBlocks;

  if (warp >= 5) {
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int group = 0; group < total_groups; ++group) {
      const int stage = group % kStages;
      if (group >= kStages) {
        const int consumed = group - kStages;
        wait_barrier(
            storage.group_full[consumed % kAccumulatorSlots],
            (consumed / kAccumulatorSlots) & 1);
      }
      const int raw_item =
          static_cast<int>(args.begin) + group / kQbBlocks;
      const int item = raw_item % kQbTiles;
      const int element = raw_item / kQbTiles;
      const int scale_block = group % kQbBlocks;
      auto* stage_a = reinterpret_cast<uint8_t*>(storage.a[stage].begin());
      for (int packet = lane; packet < kSubtiles * kOutputTile * 2;
           packet += kStagingThreads) {
        const int subtile = packet / (kOutputTile * 2);
        const int within = packet % (kOutputTile * 2);
        const int output = within >> 1;
        const int half = within & 1;
        const cutlass::float_e4m3_t* source = args.wqb
            + (item * kOutputTile + output) * kQbRank
            + scale_block * kQuantBlock + subtile * 32 + half * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            stage_a + subtile * kWeightSubtileBytes
                + output * 16 + half * 2048,
            source);
      }
      auto* stage_b = reinterpret_cast<uint8_t*>(storage.b[stage].begin());
      for (int packet = lane; packet < kSubtiles * kDraftBlock * 2;
           packet += kStagingThreads) {
        const int subtile = packet / (kDraftBlock * 2);
        const int within = packet % (kDraftBlock * 2);
        const int row = within >> 1;
        const int half = within & 1;
        const cutlass::float_e4m3_t* source = args.q_lora_quantized
            + (element * kDraftBlock + row) * kQbRank
            + scale_block * kQuantBlock + subtile * 32 + half * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            stage_b + subtile * kInputSubtileBytes
                + half * 128 + row * 16,
            source);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(
          &storage.stage_full[stage]);
    }
  } else if (warp == 4) {
    for (int group = 0; group < total_groups; ++group) {
      const int stage = group % kStages;
      const int slot = group % kAccumulatorSlots;
      wait_barrier(storage.stage_full[stage], (group / kStages) & 1);
      if (group >= kAccumulatorSlots) {
        wait_barrier(
            storage.group_free[slot],
            (group / kAccumulatorSlots - 1) & 1);
      }
      CUTE_UNROLL
      for (int subtile = 0; subtile < kSubtiles; ++subtile) {
        SM100_MMA_F8F6F4_SS::fma(
            weight_descriptor[stage]
                + static_cast<uint64_t>(subtile * kWeightSubtileBytes / 16),
            input_descriptor[stage]
                + static_cast<uint64_t>(subtile * kInputSubtileBytes / 16),
            storage.tmem_base_ptr + slot * 8,
            subtile == 0 ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.group_full[slot]);
    }
  } else {
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

    int item = 0;
    int element = 0;
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group & 1;
      const int block = group % kQbBlocks;
      if (block == 0) {
        const int raw_item =
            static_cast<int>(args.begin) + group / kQbBlocks;
        item = raw_item % kQbTiles;
        element = raw_item / kQbTiles;
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
      const float weight_scale =
          decode_e8m0(args.wqb_scales[item * kQbBlocks + block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = decode_e8m0(
            args.q_lora_scales[(element * kDraftBlock + row) * kQbBlocks
                               + block]);
        result[row] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row]);
      }
      if (block == kQbBlocks - 1) {
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          args.query_projection[
              (element * kDraftBlock + row) * kQbWidth + item * kOutputTile
              + static_cast<int>(threadIdx.x)] =
              __float2bfloat16_rn(result[row]);
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

// wo_b TCGen05 candidate: same dspark_shared FP8 pipeline; one item is one
// (output_tile, split) pair (drop-in for the 128-item phase tiling), 16
// scale groups of 4 K=32 MMAs per item, and the epilogue stores raw FP32
// split partials (no BF16 rounding).
__device__ inline void execute_wob_tcgen(const WobArgs& args) {
  using namespace cute;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kStages = dspark_shared::kStages;
  constexpr int kAccumulatorSlots = dspark_shared::kAccumulatorSlots;
  constexpr int kSplitK = kWobInput / kWobSplits;      // 2048 columns
  constexpr int kMmaBlocks = kSplitK / 32;             // 64 blocks per item
  constexpr int kGroupBlocks = 4;
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<dspark_shared::TcgenSmemLayoutB>);
  static_assert(
      sizeof(dspark_shared::TcgenSharedStorage) <= kSharedBytesBudget);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<dspark_shared::TcgenSharedStorage*>(
      dynamic_shared_memory);
  dspark_shared::TcgenTiledMma tiled_mma;
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
            layout<0>(dspark_shared::TcgenSmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(dspark_shared::TcgenSmemLayoutB{}))).desc_;
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
      const int item = static_cast<int>(args.begin) + b / kMmaBlocks;
      const int split = item % kWobSplits;
      const int raw_tile = item / kWobSplits;
      const int output_tile = raw_tile % kWobTiles;
      const int element = raw_tile / kWobTiles;
      const int block = b % kMmaBlocks;
      const int column_base = split * kSplitK + block * 32;
      cutlass::float_e4m3_t* stage_a = storage.a[s].begin();
      for (int copy_index = lane; copy_index < kOutputTile * 2;
           copy_index += kStagingThreads) {
        const int output = copy_index >> 1;
        const int chunk = copy_index & 1;
        const cutlass::float_e4m3_t* source = args.wo_b
            + (output_tile * kOutputTile + output) * kWobInput
            + column_base + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(stage_a) + output * 16 + chunk * 2048,
            source);
      }
      if (lane < 2 * kDraftBlock) {
        const int row = lane >> 1;
        const int chunk = lane & 1;
        const cutlass::float_e4m3_t* source =
            args.output_lora_quantized
            + (element * kDraftBlock + row) * kWobInput + column_base
            + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(storage.b[s].begin())
                + chunk * 128 + row * 16,
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
          storage.tmem_base_ptr + slot * 8,
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

    int split = 0;
    int output_tile = 0;
    int element = 0;
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    constexpr int kGroupsPerItem = kMmaBlocks / kGroupBlocks;  // 16
    const int total_groups = total_blocks / kGroupBlocks;
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group % 2;
      const int block = group % kGroupsPerItem;
      if (block == 0) {
        const int item =
            static_cast<int>(args.begin) + group / kGroupsPerItem;
        split = item % kWobSplits;
        const int raw_tile = item / kWobSplits;
        output_tile = raw_tile % kWobTiles;
        element = raw_tile / kWobTiles;
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
      const int k_block = split * kWobBlocksPerSplit + block;
      const float weight_scale = decode_e8m0(
          args.wo_b_scales[output_tile * kWobBlocks + k_block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = decode_e8m0(
            args.output_lora_scales[(element * kDraftBlock + row) * kWobBlocks
                                    + k_block]);
        result[row] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row]);
      }
      if (block == kGroupsPerItem - 1) {
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          args.output_partials[
              ((element * kDraftBlock + row) * kWobHidden
               + output_tile * kOutputTile
               + static_cast<int>(threadIdx.x)) * kWobSplits + split] =
              result[row];
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

// wo_a TCGen05 candidate: the dspark_lm BF16 UMMA ring with the activation
// (B) operand carried IN the ring instead of staged once — B changes with
// each item's group, and re-reading 5 x 64 columns per slot costs ~4% of the
// A stream. One item is one (group, rank_tile) pair covering all five draft
// rows (the production wiring must re-tile the phase 320 -> 64 items,
// item = group * 8 + rank_tile). Per slot warp 4 issues 4 chained
// M=128 x N=8 x K=16 BF16 UMMAs (scaleC=0 opens an item, 1 chains across all
// 256 K-slices); warps 0-3 drain the 128x8 TMEM accumulator once per item
// and run the reference's quantization epilogue per draft row (order-free
// fmaxf amax tree, then the exact exponent/clamp/ldexpf chain). Numerics vs
// the serial 4096-FMA reference chain: LM-class accumulation-order drift
// (#iteration-3: mean 4.4e-6 at logit scale ~1); the bench reports it.
struct WoaUmmaSharedStorage {
  alignas(128) __nv_bfloat16
      a_ring[dspark_lm::kUmmaStages][dspark_lm::kUmmaSlotElements];  // 64 KB
  alignas(128) __nv_bfloat16
      b_ring[dspark_lm::kUmmaStages][8 * dspark_lm::kUmmaSlotK];     // 4 KB
  alignas(16) float partial[2][kOutputTile * 8];                     // 8 KB
  alignas(16) float quant_scratch[8];
  alignas(16) cute::uint64_t stage_full[dspark_lm::kUmmaStages];
  alignas(16) cute::uint64_t slot_free[dspark_lm::kUmmaStages];
  alignas(16) cute::uint64_t accumulator_full[1];
  alignas(16) cute::uint64_t accumulator_free[1];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <bool kBulkStaging = false>
__device__ inline void execute_woa_umma(const WoaArgs& args) {
  using namespace cute;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kTmemColumns = 32;
  constexpr int kUmmaStages = dspark_lm::kUmmaStages;          // 4
  constexpr int kUmmaK = dspark_lm::kUmmaK;                    // 16
  constexpr int kUmmaSubTiles = dspark_lm::kUmmaSubTiles;      // 4
  constexpr int kUmmaSubTileShift = dspark_lm::kUmmaSubTileShift;
  constexpr int kUmmaSlotK = dspark_lm::kUmmaSlotK;            // 64
  constexpr int kSlotsPerItem = kWoaGroupInput / kUmmaSlotK;   // 64
  static_assert(
      sizeof(WoaUmmaSharedStorage) <= kSharedBytesBudget,
      "wo_a UMMA staging rings must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<WoaUmmaSharedStorage*>(dynamic_shared_memory);
  dspark_lm::LmUmmaTiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<kOutputTile>{}, Int<8>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = static_cast<int>(threadIdx.x) / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(
        storage.stage_full, kBulkStaging ? 33 : kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(storage.slot_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Rows 5..7 of every B ring slot stay zero for the whole invocation;
  // cp.async refills only rows 0..4 per slot. Chunk c of a slot holds K
  // columns [c*16, c*16+16) at byte base c*256 (halves at +0/+128, rows at
  // +row*16), exactly the dspark_lm padded-B layout.
  for (int index = static_cast<int>(threadIdx.x);
       index < kUmmaStages * 8 * kUmmaSlotK;
       index += static_cast<int>(blockDim.x)) {
    reinterpret_cast<__nv_bfloat16*>(storage.b_ring)[index] =
        __float2bfloat16(0.0f);
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

  uint64_t a_descriptor[kUmmaStages];
  uint64_t b_descriptor[kUmmaStages];
  CUTE_UNROLL
  for (int s = 0; s < kUmmaStages; ++s) {
    a_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[s])),
            layout<0>(dspark_lm::LmUmmaSmemLayoutA{}))).desc_;
    b_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_ring[s])),
            layout<0>(dspark_lm::LmUmmaSmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_slots = item_count * kSlotsPerItem;

  if (kBulkStaging && warp >= 5) {
    // One TMA producer streams prepacked weights; a separate warp supplies
    // the small, row-major activation operand. Both copies contribute to
    // stage_full, preserving the existing four K16 MMAs and quantization.
    if (threadIdx.x == 160 || warp == 6) {
      for (int c = 0; c < total_slots; ++c) {
        const int s = c % kUmmaStages;
        if (c >= kUmmaStages) {
          wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
        }
        const int raw_item = static_cast<int>(args.begin) + c / kSlotsPerItem;
        const int item = raw_item % kWoaItemsPerElement;
        const int element = raw_item / kWoaItemsPerElement;
        const int group = item / kWoaRankTiles;
        const int k_base = (c % kSlotsPerItem) * kUmmaSlotK;
        if (threadIdx.x == 160) {
          constexpr int bytes = dspark_lm::kUmmaSlotElements * sizeof(__nv_bfloat16);
          const auto* source = args.wo_a
              + (static_cast<int64_t>(item) * kSlotsPerItem + c % kSlotsPerItem)
                  * dspark_lm::kUmmaSlotElements;
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
              &storage.stage_full[s], bytes);
          cute::SM90_BULK_COPY_G2S::copy(
              source, &storage.stage_full[s], storage.a_ring[s], bytes);
        } else {
          const int lane = static_cast<int>(threadIdx.x) % 32;
          auto* slot = reinterpret_cast<unsigned char*>(storage.b_ring[s]);
          for (int index = lane; index < 8 * kUmmaSubTiles * 2; index += 32) {
            const unsigned packet = static_cast<unsigned>(index);
            const int row = static_cast<int>(((packet >> 5) << 2) | ((packet >> 1) & 3u));
            if (row < kDraftBlock) {
              const int half = static_cast<int>(packet & 1u);
              const int j = static_cast<int>((packet >> 3) & 3u);
              cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
                  slot + j * 256 + half * 128 + row * 16,
                  args.attention + ((element * kDraftBlock + row) * kWoaGroups + group)
                      * kWoaGroupInput + k_base + j * kUmmaK + half * 8);
            }
          }
          cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
        }
      }
    }
  } else if (warp >= 5) {
    // Staging crew: 1024 A packets + 40 B packets per 128x64 slot.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      if (c >= kUmmaStages) {
        wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
      }
      const int raw_item = static_cast<int>(args.begin) + c / kSlotsPerItem;
      const int item = raw_item % kWoaItemsPerElement;
      const int element = raw_item / kWoaItemsPerElement;
      const int group = item / kWoaRankTiles;
      const int rank_tile = item % kWoaRankTiles;
      const int k_base = (c % kSlotsPerItem) * kUmmaSlotK;
      unsigned char* slot = reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      // Bank-conflict-free lane->packet map (see execute_umma_bf16 in
      // dspark_lm_phase.cuh): m = index >> 3 would put a whole quarter-warp
      // at SMEM offsets 2048 apart, i.e. in the same four banks -- an 8-way
      // store conflict. Interleaving two bits of m below j spreads the
      // quarter-warp over four rows while keeping each row's two 16-byte
      // halves adjacent, so the global 32-byte sectors are unchanged.
      // Bitwise by construction (pure lane re-assignment).
      static_assert(kOutputTile % 4 == 0, "quarter-warp spreads 4 rows");
      for (int index = lane; index < kOutputTile * kUmmaSubTiles * 2;
           index += kStagingThreads) {
        const unsigned packet = static_cast<unsigned>(index);
        const int half = static_cast<int>(packet & 1u);
        const int j = static_cast<int>(
            (packet >> 3) & static_cast<unsigned>(kUmmaSubTiles - 1));
        const int m = static_cast<int>(
            ((packet >> (3 + kUmmaSubTileShift)) << 2) | ((packet >> 1) & 3u));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            slot + j * 4096 + half * 2048 + m * 16,
            args.wo_a
                + (static_cast<int64_t>(group) * kWoaOutputRank
                   + rank_tile * kOutputTile + m) * kWoaGroupInput
                + k_base + j * kUmmaK + half * 8);
      }
      unsigned char* b_slot =
          reinterpret_cast<unsigned char*>(storage.b_ring[s]);
      // Same bank-conflict-free decomposition as the A ring above: eight
      // consecutive lanes on ONE row would land 128 B apart, i.e. in the same
      // four banks. The walk covers the padded 8-row space with rows
      // >= kDraftBlock predicated off, so the same 40 packets reach the same
      // SMEM addresses -- bitwise by construction.
      static_assert(kUmmaSubTiles == 4, "b-ring map assumes 4 K16 chunks");
      for (int index = lane; index < 8 * kUmmaSubTiles * 2;
           index += kStagingThreads) {
        const unsigned packet = static_cast<unsigned>(index);
        const int row = static_cast<int>(
            ((packet >> 5) << 2) | ((packet >> 1) & 3u));
        if (row >= kDraftBlock) {
          continue;
        }
        const int half = static_cast<int>(packet & 1u);
        const int j = static_cast<int>((packet >> 3) & 3u);
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            b_slot + j * 256 + half * 128 + row * 16,
            args.attention
                + ((element * kDraftBlock + row) * kWoaGroups + group)
                      * kWoaGroupInput
                + k_base + j * kUmmaK + half * 8);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    // MMA crew: four chained K=16 UMMAs per slot into ONE TMEM accumulator;
    // scaleC=0 opens each item, 1 chains.
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      const int slot_in_item = c % kSlotsPerItem;
      const int item_local = c / kSlotsPerItem;
      wait_barrier(storage.stage_full[s], (c / kUmmaStages) & 1);
      if (slot_in_item == 0 && item_local > 0) {
        wait_barrier(storage.accumulator_free[0], (item_local - 1) & 1);
      }
      CUTE_UNROLL
      for (int j = 0; j < kUmmaSubTiles; ++j) {
        dspark_lm::LmUmmaAtom::fma(
            a_descriptor[s] + static_cast<uint64_t>(j) * 256,   // 4 KB >> 4
            b_descriptor[s] + static_cast<uint64_t>(j) * 16,    // 256 B >> 4
            storage.tmem_base_ptr,
            (slot_in_item == 0 && j == 0) ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kSlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[0]);
      }
    }
  } else {
    // Drain crew (warps 0-3): one TMEM drain per item, then the reference's
    // per-(row, rank_tile) FP8 quantization epilogue.
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto slot_view = tmem_accumulator;
    slot_view.data() = storage.tmem_base_ptr;
    auto thread_tmem = thread_copy.partition_S(slot_view);
    auto partial_layout = make_layout(
        make_shape(Int<kOutputTile>{}, Int<8>{}),
        make_stride(Int<1>{}, Int<kOutputTile>{}));
    auto shared_partial_low = make_tensor(
        make_smem_ptr(storage.partial[0]), partial_layout);
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));
    const int lane = static_cast<int>(threadIdx.x) % 32;

    for (int i = 0; i < item_count; ++i) {
      const int parity = i & 1;
      wait_barrier(storage.accumulator_full[0], parity);
      copy(tmem_to_register, thread_tmem, register_accumulator);
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[0]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const int raw_item = static_cast<int>(args.begin) + i;
      const int item = raw_item % kWoaItemsPerElement;
      const int element = raw_item / kWoaItemsPerElement;
      const int group = item / kWoaRankTiles;
      const int rank_tile = item % kWoaRankTiles;
      const int rank = rank_tile * kOutputTile + static_cast<int>(threadIdx.x);
      for (int row = 0; row < kDraftBlock; ++row) {
        const int flat_row = element * kDraftBlock + row;
        const int index =
            (flat_row * kWoaGroups + group) * kWoaOutputRank + rank;
        const __nv_bfloat16 rounded = __float2bfloat16_rn(
            storage.partial[parity][
                row * kOutputTile + static_cast<int>(threadIdx.x)]);
        args.output_lora[index] = rounded;
        // Order-free per-(row, rank_tile) amax: warp fmaxf butterflies, then
        // one cross-warp fold (same selected value as the reference's
        // stride-halving shared-memory tree).
        float amax_lane = fabsf(__bfloat162float(rounded));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          amax_lane = fmaxf(
              amax_lane, __shfl_xor_sync(0xffffffffu, amax_lane, offset));
        }
        if (lane == 0) {
          storage.quant_scratch[warp] = amax_lane;
        }
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
        if (threadIdx.x == 0) {
          const float amax = fmaxf(
              fmaxf(
                  fmaxf(storage.quant_scratch[0], storage.quant_scratch[1]),
                  fmaxf(storage.quant_scratch[2], storage.quant_scratch[3])),
              1.0e-4f);
          const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
          const int encoded = exponent + 127;
          args.output_lora_scales[
              flat_row * (kWoaGroups * kWoaRankTiles) + group * kWoaRankTiles
              + rank_tile] =
              static_cast<uint8_t>(
                  encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
          storage.quant_scratch[4] = ldexpf(1.0f, exponent);
        }
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
        args.output_lora_quantized[index] =
            convert(__bfloat162float(rounded) / storage.quant_scratch[4]);
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
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

// ---------------------------------------------------------------------------
// N-WIDENED wo_a (R4 batch-amortization campaign, band 1).
//
// The body above tiles wo_a OVER BATCH: one item is one (element, group,
// rank_tile) triple, so the 64 MiB wo_a weight matrix is re-streamed once per
// batch element. Every element reads the SAME weights, so those re-reads are
// pure waste.
//
// This variant carries the batch on the tensor-core N mode exactly the way
// dspark_lm::execute_umma_bf16_batch does for the LM head: the item space
// goes back to the batch-invariant 64 (group, rank_tile) pairs, the UMMA atom
// widens from N=8 to N = round-up-8 of 5*batch, and each 128x64 weight slot
// now carries the activation window for ALL flat draft rows over the same K
// span. The weight stream is read exactly once per proposal.
//
// Numerics: every output element (flat_row, group, rank) still accumulates
// over 256 chained K=16 UMMAs in ascending K with its FP32 accumulator
// resident in TMEM (scaleC=0 opens the item, 1 chains). Widening N only adds
// independent output COLUMNS -- no per-output accumulation order changes --
// so results are bitwise identical to the N=8 body row for row. The
// per-(row, rank_tile) amax epilogue is unchanged as well: it reduces over
// the same 128 ranks with the same order-free fmaxf tree.
#if DSPARK_V4_BATCH > 1
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

template <int kRows>
struct WoaUmmaBatchStorage {
  alignas(128) __nv_bfloat16
      a_ring[dspark_lm::kUmmaStages][dspark_lm::kUmmaSlotElements];
  alignas(128) __nv_bfloat16
      b_ring[dspark_lm::kUmmaStages]
            [dspark_lm::kUmmaSubTiles * kRows * dspark_lm::kUmmaK];
  alignas(16) float partial[2][kOutputTile * kRows];
  alignas(16) float quant_scratch[8];
  alignas(16) cute::uint64_t stage_full[dspark_lm::kUmmaStages];
  alignas(16) cute::uint64_t slot_free[dspark_lm::kUmmaStages];
  alignas(16) cute::uint64_t accumulator_full[1];
  alignas(16) cute::uint64_t accumulator_free[1];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <int kRows, int kRealRows>
__device__ inline void execute_woa_umma_batch(const WoaArgs& args) {
  using namespace cute;
  using Traits = dspark_lm::LmBatchTraits<kRows>;
  using Atom = typename Traits::Atom;
  using Storage = WoaUmmaBatchStorage<kRows>;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kTmemColumns = Traits::kTmemColumns;
  constexpr int kUmmaStages = dspark_lm::kUmmaStages;          // 4
  constexpr int kUmmaK = dspark_lm::kUmmaK;                    // 16
  constexpr int kUmmaSubTiles = dspark_lm::kUmmaSubTiles;      // 4
  constexpr int kUmmaSubTileShift = dspark_lm::kUmmaSubTileShift;
  constexpr int kUmmaSlotK = dspark_lm::kUmmaSlotK;            // 64
  constexpr int kSlotsPerItem = kWoaGroupInput / kUmmaSlotK;   // 64
  constexpr int kBChunkElements = kRows * kUmmaK;
  constexpr uint64_t kBChunkUnits = static_cast<uint64_t>(kRows) * 2;
  static_assert(kRows % 8 == 0 && kRows >= 8 && kRows <= 256,
                "tcgen05 f16 N-mode must be a multiple of 8 in [8, 256]");
  static_assert(kRealRows >= 1 && kRealRows <= kRows,
                "the widened body must cover the real draft rows and no more");
  static_assert(cute::cosize_v<typename Traits::BChunkLayout> == kRows * kUmmaK,
                "one K16 activation chunk is exactly kRows*16 elements");
  static_assert(sizeof(Storage) <= kSharedBytesBudget,
                "widened wo_a staging rings must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<Storage*>(dynamic_shared_memory);
  typename Traits::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<kOutputTile>{}, Int<kRows>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = static_cast<int>(threadIdx.x) / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(storage.slot_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Every B ring slot is fully refilled by cp.async each round (pad columns
  // included), so no zero prologue is needed.
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

  uint64_t a_descriptor[kUmmaStages];
  uint64_t b_descriptor[kUmmaStages];
  CUTE_UNROLL
  for (int s = 0; s < kUmmaStages; ++s) {
    a_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[s])),
            layout<0>(typename Traits::SmemLayoutA{}))).desc_;
    b_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_ring[s])),
            layout<0>(typename Traits::SmemLayoutB{}))).desc_;
  }
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_slots = item_count * kSlotsPerItem;

  if (warp >= 5) {
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      if (c >= kUmmaStages) {
        wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
      }
      const int item = static_cast<int>(args.begin) + c / kSlotsPerItem;
      const int group = item / kWoaRankTiles;
      const int rank_tile = item % kWoaRankTiles;
      const int k_base = (c % kSlotsPerItem) * kUmmaSlotK;
      unsigned char* slot = reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      // Bank-conflict-free lane->packet map (see execute_umma_bf16 in
      // dspark_lm_phase.cuh): m = index >> 3 would put a whole quarter-warp
      // at SMEM offsets 2048 apart, i.e. in the same four banks -- an 8-way
      // store conflict. Interleaving two bits of m below j spreads the
      // quarter-warp over four rows while keeping each row's two 16-byte
      // halves adjacent, so the global 32-byte sectors are unchanged.
      // Bitwise by construction (pure lane re-assignment).
      static_assert(kOutputTile % 4 == 0, "quarter-warp spreads 4 rows");
      for (int index = lane; index < kOutputTile * kUmmaSubTiles * 2;
           index += kStagingThreads) {
        const unsigned packet = static_cast<unsigned>(index);
        const int half = static_cast<int>(packet & 1u);
        const int j = static_cast<int>(
            (packet >> 3) & static_cast<unsigned>(kUmmaSubTiles - 1));
        const int m = static_cast<int>(
            ((packet >> (3 + kUmmaSubTileShift)) << 2) | ((packet >> 1) & 3u));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            slot + j * 4096 + half * 2048 + m * 16,
            args.wo_a
                + (static_cast<int64_t>(group) * kWoaOutputRank
                   + rank_tile * kOutputTile + m) * kWoaGroupInput
                + k_base + j * kUmmaK + half * 8);
      }
      __nv_bfloat16* b_slot = storage.b_ring[s];
      for (int index = lane; index < kUmmaSubTiles * kRows * 2;
           index += kStagingThreads) {
        const int j = index / (kRows * 2);
        const int rest = index - j * (kRows * 2);
        const int row = rest >> 1;
        const int half = rest & 1;
        // Pad columns (row >= kRealRows) re-read the last real row: their
        // outputs are never stored, and this keeps every load in bounds.
        const int source_row = row < kRealRows ? row : kRealRows - 1;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            b_slot + j * kBChunkElements + row * 8 + half * 8 * kRows,
            args.attention
                + (static_cast<int64_t>(source_row) * kWoaGroups + group)
                      * kWoaGroupInput
                + k_base + j * kUmmaK + half * 8);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      const int slot_in_item = c % kSlotsPerItem;
      const int item_local = c / kSlotsPerItem;
      wait_barrier(storage.stage_full[s], (c / kUmmaStages) & 1);
      if (slot_in_item == 0 && item_local > 0) {
        wait_barrier(storage.accumulator_free[0], (item_local - 1) & 1);
      }
      CUTE_UNROLL
      for (int j = 0; j < kUmmaSubTiles; ++j) {
        Atom::fma(
            a_descriptor[s] + static_cast<uint64_t>(j) * 256,   // 4 KB >> 4
            b_descriptor[s] + static_cast<uint64_t>(j) * kBChunkUnits,
            storage.tmem_base_ptr,
            (slot_in_item == 0 && j == 0) ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kSlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[0]);
      }
    }
  } else {
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto slot_view = tmem_accumulator;
    slot_view.data() = storage.tmem_base_ptr;
    auto thread_tmem = thread_copy.partition_S(slot_view);
    auto partial_layout = make_layout(
        make_shape(Int<kOutputTile>{}, Int<kRows>{}),
        make_stride(Int<1>{}, Int<kOutputTile>{}));
    auto shared_partial_low = make_tensor(
        make_smem_ptr(storage.partial[0]), partial_layout);
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));
    const int lane = static_cast<int>(threadIdx.x) % 32;

    for (int i = 0; i < item_count; ++i) {
      const int parity = i & 1;
      wait_barrier(storage.accumulator_full[0], parity);
      copy(tmem_to_register, thread_tmem, register_accumulator);
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[0]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const int item = static_cast<int>(args.begin) + i;
      const int group = item / kWoaRankTiles;
      const int rank_tile = item % kWoaRankTiles;
      const int rank = rank_tile * kOutputTile + static_cast<int>(threadIdx.x);
      for (int flat_row = 0; flat_row < kRealRows; ++flat_row) {
        const int index =
            (flat_row * kWoaGroups + group) * kWoaOutputRank + rank;
        const __nv_bfloat16 rounded = __float2bfloat16_rn(
            storage.partial[parity][
                flat_row * kOutputTile + static_cast<int>(threadIdx.x)]);
        args.output_lora[index] = rounded;
        float amax_lane = fabsf(__bfloat162float(rounded));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          amax_lane = fmaxf(
              amax_lane, __shfl_xor_sync(0xffffffffu, amax_lane, offset));
        }
        if (lane == 0) {
          storage.quant_scratch[warp] = amax_lane;
        }
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
        if (threadIdx.x == 0) {
          const float amax = fmaxf(
              fmaxf(
                  fmaxf(storage.quant_scratch[0], storage.quant_scratch[1]),
                  fmaxf(storage.quant_scratch[2], storage.quant_scratch[3])),
              1.0e-4f);
          const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
          const int encoded = exponent + 127;
          args.output_lora_scales[
              flat_row * (kWoaGroups * kWoaRankTiles) + group * kWoaRankTiles
              + rank_tile] =
              static_cast<uint8_t>(
                  encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
          storage.quant_scratch[4] = ldexpf(1.0f, exponent);
        }
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
        args.output_lora_quantized[index] =
            convert(__bfloat162float(rounded) / storage.quant_scratch[4]);
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
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

}  // namespace dspark_proj
