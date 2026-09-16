// Self-contained layer-0 sparse-attention phase bodies for the DeepSeek-V4
// DSpark megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// Work item shape (frozen by the phase program): one (draft row, head) pair,
// 320 items per layer. Each item runs online softmax over 3 blocks x 64 keys
// (128-slot target KV window + 5 draft rows, remainder masked), a learned
// per-head attention sink joining the softmax denominator with no value
// vector, and RoPE on the trailing 64 dims applied AFTER the bf16 rounding
// of the attention output.
//
// execute_reference is the verbatim #run-102-era production body (legacy
// nvcuda::wmma 16x16x16 BF16): it is its own oracle in the bench lane.
// execute_batched is the relaxed-program stage-3 candidate (see below).
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cfloat>
#include <cstdint>

#include "dspark_attn_mma8.cuh"
#include "dspark_batch.h"
#include "dspark_v4_tp2_ablate.cuh"

namespace dspark_attn {

constexpr int kDraftBlock = 5;
// Batched serving: `queries`, `draft_kv` and the outputs are flat over the
// (batch, block, ...) workspace, so `row` already walks the batch. Only the
// per-element KV window needs an explicit stride: kv_cache is
// (layers, batch, window, kv_width) and this layer's pointer addresses
// element 0. All elements share start_pos in this specialization.
constexpr int kBatch = dspark_batch::kBatch;
constexpr int kTargetKvElementStride = 128 * 512;
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;
// R10: both call sites launch 256 threads/CTA (generated::kThreads in the
// megakernel, kBenchThreads in the phase bench; each TU static_asserts the
// match). blockDim.x is a RUNTIME special register, so every loop strided by
// it -- the KV/Q cp.async staging, the accumulator rescale, the softmax
// elementwise passes, the epilogue -- has a trip count nvcc cannot see and
// therefore cannot unroll, and the PV output-tile loop's `warps` stride was
// runtime for the same reason. Substituting the constant changes no address,
// no operand and no accumulation order.
constexpr int kAttnThreads = 256;
constexpr int kAttnWarps = kAttnThreads / 32;

struct Args {
  // 5 rows x 64 heads x 512 dims, bf16, already RoPE'd/normalized.
  const __nv_bfloat16* queries;
  // This layer's 128-slot KV window (the caller applies the layer offset).
  const __nv_bfloat16* target_kv;
  // 5 draft rows x 512 dims.
  const __nv_bfloat16* draft_kv;
  // FP32 scratch, 5 x 64 x 512 (the reference keeps its running PV here).
  float* accumulator;
  // Pre-RoPE bf16 attention output, 5 x 64 x 512.
  __nv_bfloat16* raw_output;
  // Post-RoPE bf16 attention output, 5 x 64 x 512.
  __nv_bfloat16* output;
  // 64 learned per-head sink logits.
  const float* attention_sink;
  // Interleaved cos/sin pairs; row r reads pairs at (1 + r) * 64.
  const float* rope_cos_sin;
  int start_pos;
  uint32_t begin;
  uint32_t end;
};

__device__ __forceinline__ float load_sparse_kv(
    const Args& args,
    int topk_ordinal,
    int column,
    int target_count) {
  if (topk_ordinal < target_count) {
    return __bfloat162float(args.target_kv[topk_ordinal * 512 + column]);
  }
  const int draft_row = topk_ordinal - target_count;
  if (draft_row < kDraftBlock) {
    return __bfloat162float(args.draft_kv[draft_row * 512 + column]);
  }
  return 0.0f;
}

// Verbatim extraction of execute_layer0_sparse_attention from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_reference(const Args& args, float* shared) {
  constexpr int kHeads = 64;
  constexpr int kHeadDim = 512;
  constexpr int kRopeDim = 64;
  constexpr int kSparseBlock = 64;
  constexpr int kSparseBlocks = 3;
  constexpr float kSoftmaxScale = 0.04419417382415922f;
  constexpr int kWmmaTile = 16;
  constexpr int kScoreWarps = kSparseBlock / kWmmaTile;
  constexpr int kOutputTiles = kHeadDim / kWmmaTile;
  __nv_bfloat16* tensor_shared = reinterpret_cast<__nv_bfloat16*>(shared);
  __nv_bfloat16* query_matrix = tensor_shared;
  __nv_bfloat16* kv_matrix = query_matrix + kWmmaTile * kHeadDim;
  // Score fragments live beyond the query/KV staging range so a fast warp
  // cannot overwrite the replicated query while another warp is still
  // consuming it.
  float* score_tiles = shared + 20480;
  float* scores = shared + 1024;
  __nv_bfloat16* weights = reinterpret_cast<__nv_bfloat16*>(shared + 1088);
  __nv_bfloat16* weight_matrix = reinterpret_cast<__nv_bfloat16*>(shared + 1632);
  float* warp_output = shared + 2144;
  // Online-softmax state survives all three KV blocks, so it must sit beyond
  // both the query/KV staging range and the score-fragment range.
  float* state = shared + 21504;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kHeads;
    const int head = static_cast<int>(item) % kHeads;
    const int query_base = (row * kHeads + head) * kHeadDim;
    const int target_count = args.start_pos + 1 < 128 ? args.start_pos + 1 : 128;
    const int topk = target_count + kDraftBlock;
    const int output_base = query_base;
    for (int column = threadIdx.x; column < kHeadDim; column += blockDim.x) {
      args.accumulator[output_base + column] = 0.0f;
    }
    if (threadIdx.x == 0) {
      state[0] = -FLT_MAX;
      state[1] = 0.0f;
    }
    __syncthreads();

    for (int block = 0; block < kSparseBlocks; ++block) {
      for (int index = threadIdx.x; index < kWmmaTile * kHeadDim;
           index += blockDim.x) {
        query_matrix[index] = args.queries[query_base + index % kHeadDim];
      }
      for (int index = threadIdx.x; index < kSparseBlock * kHeadDim;
           index += blockDim.x) {
        const int ordinal = block * kSparseBlock + index / kHeadDim;
        const int column = index % kHeadDim;
        kv_matrix[index] = __float2bfloat16_rn(
            ordinal < topk
                ? load_sparse_kv(args, ordinal, column, target_count)
                : 0.0f);
      }
      __syncthreads();

      const int warp = threadIdx.x / warpSize;
      if (warp < kScoreWarps) {
        using namespace nvcuda;
        wmma::fragment<
            wmma::matrix_a,
            kWmmaTile,
            kWmmaTile,
            kWmmaTile,
            __nv_bfloat16,
            wmma::row_major>
            query_fragment;
        wmma::fragment<
            wmma::matrix_b,
            kWmmaTile,
            kWmmaTile,
            kWmmaTile,
            __nv_bfloat16,
            wmma::col_major>
            kv_fragment;
        wmma::fragment<
            wmma::accumulator,
            kWmmaTile,
            kWmmaTile,
            kWmmaTile,
            float>
            score_fragment;
        wmma::fill_fragment(score_fragment, 0.0f);
        for (int k_tile = 0; k_tile < kHeadDim / kWmmaTile; ++k_tile) {
          wmma::load_matrix_sync(
              query_fragment,
              query_matrix + k_tile * kWmmaTile,
              kHeadDim);
          wmma::load_matrix_sync(
              kv_fragment,
              kv_matrix + warp * kWmmaTile * kHeadDim + k_tile * kWmmaTile,
              kHeadDim);
          wmma::mma_sync(
              score_fragment,
              query_fragment,
              kv_fragment,
              score_fragment);
        }
        wmma::store_matrix_sync(
            score_tiles + warp * kWmmaTile * kWmmaTile,
            score_fragment,
            kWmmaTile,
            wmma::mem_row_major);
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        for (int position = 0; position < kSparseBlock; ++position) {
          const int score_warp = position / kWmmaTile;
          const int score_column = position % kWmmaTile;
          const int ordinal = block * kSparseBlock + position;
          scores[position] = ordinal < topk
              ? score_tiles[score_warp * kWmmaTile * kWmmaTile + score_column]
                  * kSoftmaxScale
              : -FLT_MAX;
        }
        const float previous_max = state[0];
        float maximum = previous_max;
        for (int position = 0; position < kSparseBlock; ++position) {
          maximum = fmaxf(maximum, scores[position]);
        }
        const float rescale = expf(previous_max - maximum);
        for (int position = 0; position < kSparseBlock; ++position) {
          const float weight =
              scores[position] == -FLT_MAX ? 0.0f : expf(scores[position] - maximum);
          weights[position] = __float2bfloat16_rn(weight);
          scores[position] = weight;
        }
        for (int stride = kSparseBlock / 2; stride > 0; stride /= 2) {
          for (int position = 0; position < stride; ++position) {
            scores[position] += scores[position + stride];
          }
        }
        state[0] = maximum;
        state[1] = state[1] * rescale + scores[0];
        state[2] = rescale;
      }
      __syncthreads();

      for (int index = threadIdx.x; index < kWmmaTile * kSparseBlock;
           index += blockDim.x) {
        weight_matrix[index] = weights[index % kSparseBlock];
      }
      __syncthreads();

      if (warp < kScoreWarps) {
        using namespace nvcuda;
        for (int output_tile = warp; output_tile < kOutputTiles;
             output_tile += kScoreWarps) {
          wmma::fragment<
              wmma::matrix_a,
              kWmmaTile,
              kWmmaTile,
              kWmmaTile,
              __nv_bfloat16,
              wmma::row_major>
              weight_fragment;
          wmma::fragment<
              wmma::matrix_b,
              kWmmaTile,
              kWmmaTile,
              kWmmaTile,
              __nv_bfloat16,
              wmma::row_major>
              value_fragment;
          wmma::fragment<
              wmma::accumulator,
              kWmmaTile,
              kWmmaTile,
              kWmmaTile,
              float>
              output_fragment;
          float* warp_tile = warp_output + warp * kWmmaTile * kWmmaTile;
          const int lane = threadIdx.x % warpSize;
          for (int index = lane; index < kWmmaTile * kWmmaTile;
               index += warpSize) {
            warp_tile[index] =
                args.accumulator[
                    output_base + output_tile * kWmmaTile + index % kWmmaTile]
                * state[2];
          }
          __syncwarp();
          wmma::load_matrix_sync(
              output_fragment,
              warp_tile,
              kWmmaTile,
              wmma::mem_row_major);
          for (int k_tile = 0; k_tile < kSparseBlock / kWmmaTile; ++k_tile) {
            wmma::load_matrix_sync(
                weight_fragment,
                weight_matrix + k_tile * kWmmaTile,
                kSparseBlock);
            wmma::load_matrix_sync(
                value_fragment,
                kv_matrix + k_tile * kWmmaTile * kHeadDim
                    + output_tile * kWmmaTile,
                kHeadDim);
            wmma::mma_sync(
                output_fragment,
                weight_fragment,
                value_fragment,
                output_fragment);
          }
          wmma::store_matrix_sync(
              warp_tile,
              output_fragment,
              kWmmaTile,
              wmma::mem_row_major);
          __syncwarp();
          if (lane < kWmmaTile) {
            args.accumulator[output_base + output_tile * kWmmaTile + lane] =
                warp_tile[lane];
          }
          __syncwarp();
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float denominator =
          state[1] + expf(args.attention_sink[head] - state[0]);
      state[3] = 1.0f / denominator;
    }
    __syncthreads();
    args.raw_output[output_base + threadIdx.x] =
        __float2bfloat16_rn(args.accumulator[output_base + threadIdx.x] * state[3]);
    args.raw_output[output_base + threadIdx.x + blockDim.x] =
        __float2bfloat16_rn(
            args.accumulator[output_base + threadIdx.x + blockDim.x] * state[3]);
    args.output[output_base + threadIdx.x] = args.raw_output[output_base + threadIdx.x];
    args.output[output_base + threadIdx.x + blockDim.x] =
        args.raw_output[output_base + threadIdx.x + blockDim.x];
    __syncthreads();

    if (threadIdx.x < kRopeDim / 2) {
      const int pair = threadIdx.x;
      const int first = output_base + kHeadDim - kRopeDim + 2 * pair;
      // Batched: RoPE slot is the row's draft position (shared start_pos).
      const int rope_base = (1 + row % kDraftBlock) * kRopeDim;
      const float x0 = __bfloat162float(args.output[first]);
      const float x1 = __bfloat162float(args.output[first + 1]);
      const float cosine = args.rope_cos_sin[rope_base + pair * 2];
      const float sine = args.rope_cos_sin[rope_base + pair * 2 + 1];
      args.output[first] = __float2bfloat16_rn(x0 * cosine + x1 * sine);
      args.output[first + 1] = __float2bfloat16_rn(x1 * cosine - x0 * sine);
    }
    __syncthreads();
  }
}

// CANDIDATE body (relaxed program stage 3, attention iteration): batch
// kGroupHeads heads of one draft row per item. The reference burns its
// entire 16-row wmma A operand on ONE head (the query is replicated 16x)
// and restages the shared 64x512 KV block once per (row, head) item, so 64
// heads re-read the same KV window 64 times through scalar branchy loads.
// Here the 16 A rows carry 16 REAL queries: per item the QK^T and PV mma
// volume serves kGroupHeads heads instead of one, KV is staged once per
// group through uint4 packets (a bf16 round-trips __bfloat162float exactly,
// so the vectorized copy is value-identical to the reference staging), the
// running PV accumulator lives in shared memory instead of global scratch,
// PV output tiles spread over all 8 warps, and the per-block online-softmax
// scan runs one thread per head instead of one thread total.
//
// Numerics: each score/PV element is the same chained 16x16x16 wmma
// reduction tree over the same operand rows the reference used (an mma
// output element depends only on its own A row / B column), and the
// per-head softmax scan preserves the reference's op order (including the
// stride-halving weight-sum tree), so drift vs the reference is expected to
// be zero-to-ULP; the bench lane measures and reports it.
//
// Item shape: item = row * (64 / kGroupHeads) + head_group; the bench (and
// any production wiring) must size the phase to 320 / kGroupHeads items.
//
// kParallelSoftmax replaces the one-thread-per-head softmax scan with
// block-wide passes that are still bitwise-identical to the reference:
// the running max is an order-free fmaxf reduction (tree order cannot
// change the selected value), the exp/weight pass is elementwise, and the
// weight sum replays the reference's exact stride-halving tree with the
// per-stride additions spread across threads. It also skips KV blocks that
// are entirely beyond topk, which the reference walks through as exact
// no-ops (maximum stays, rescale == expf(0) == 1, all weights 0).
//
// kProbe is a BENCH-ONLY timing attribution mask (same pattern as
// kStagingProbe in dspark_lm_phase.cuh). GARBAGE NUMERICS; it exists only to
// price each stream of the body out of the item's serial latency. Production
// instantiates kProbe == 0, whose codegen is unchanged (every probe branch is
// `if constexpr`). Bit 8 (kProbeInit) is the attribution BASELINE: it zeroes
// the shared scratch once per item so that a skipped producer never leaves a
// consumer reading uninitialised shared memory, and every other probe mode
// sets it too, so a delta is exactly one stream.
namespace probe {
constexpr int kNoKvLoad = 1;      // KV global read -> constant (shared store kept)
constexpr int kNoQLoad = 2;       // query global read -> zero (shared store kept)
constexpr int kNoQk = 4;          // skip the QK^T wmma stage
constexpr int kNoSoftmax = 8;     // skip the online-softmax scan
constexpr int kNoRescale = 16;    // skip the accumulator rescale pass
constexpr int kNoPv = 32;         // skip the PV wmma stage
constexpr int kNoEpilogue = 64;   // skip the output store + RoPE
constexpr int kNoAccInit = 128;   // skip zeroing the running accumulator
constexpr int kInit = 256;        // baseline: zero shared scratch per item
constexpr int kNoKvStage = 512;   // skip KV staging entirely (load AND store)
constexpr int kNoQStage = 1024;   // skip query staging entirely
}  // namespace probe

// kOpt selects BODY-EFFICIENCY levers. Both move DATA only -- never an
// arithmetic operand, an operand order, or a reduction tree -- so every
// combination must stay BITWISE-identical to execute_reference, which the
// bench lane gates.
//
//   kPadShared  breaks shared-memory bank conflicts. Every tile in this body
//               is stored at its natural leading dimension, and every one of
//               those strides is a multiple of 128 B (kv/q rows 1024 B, acc
//               rows 2048 B, weight/score rows 128/256 B). ldmatrix reads 8
//               rows x 16 B at a time, so with stride % 128 == 0 all 8 rows
//               land in the same four banks: an 8-way conflict on EVERY
//               wmma::load_matrix_sync. Padding each stride to
//               stride % 128 == 16 B spreads the 8 rows over all 32 banks.
//   kCpAsync    stages KV and queries with cp.async instead of LDG->STS.
//               The scalar loop is a chain of 16 dependent global round
//               trips per thread per KV block (the trip count is
//               blockDim.x-dependent, so ptxas cannot unroll it away);
//               cp.async issues them all and waits once.
//   kHoistWeights lifts the PV weight fragments out of the output-tile loop.
namespace opt {
constexpr int kMma8 = 8;  // Eight live head columns in explicit MMA fragments.
constexpr int kCpAsync = 1;
constexpr int kPadShared = 2;
constexpr int kHoistWeights = 4;
}  // namespace opt

// 16-byte async global->shared copy. `valid == false` uses the src-size
// operand to zero-fill the destination, which is exactly what the scalar
// staging loop does for positions past topk.
__device__ __forceinline__ void cp_async_16_zfill(
    void* smem,
    const void* gmem,
    bool valid) {
  const unsigned address =
      static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int source_bytes = valid ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address),
      "l"(gmem),
      "r"(source_bytes));
}

template <
    int kGroupHeads,
    bool kParallelSoftmax = false,
    int kProbe = 0,
    int kOpt = 0>
__device__ inline void execute_batched(const Args& args, float* shared) {
#if !defined(DSPARK_ATTN_PROBE)
  static_assert(
      kProbe == 0,
      "attention attribution probes require -DDSPARK_ATTN_PROBE");
#endif
  constexpr bool kProbeNoKvLoad = (kProbe & probe::kNoKvLoad) != 0;
  constexpr bool kProbeNoQLoad = (kProbe & probe::kNoQLoad) != 0;
  constexpr bool kProbeNoQk = (kProbe & probe::kNoQk) != 0;
  constexpr bool kProbeNoSoftmax = (kProbe & probe::kNoSoftmax) != 0;
  constexpr bool kProbeNoRescale = (kProbe & probe::kNoRescale) != 0;
  constexpr bool kProbeNoPv = (kProbe & probe::kNoPv) != 0;
  constexpr bool kProbeNoEpilogue = (kProbe & probe::kNoEpilogue) != 0;
  constexpr bool kProbeNoAccInit = (kProbe & probe::kNoAccInit) != 0;
  constexpr bool kProbeInit = (kProbe & probe::kInit) != 0;
  constexpr bool kProbeNoKvStage = (kProbe & probe::kNoKvStage) != 0;
  constexpr bool kProbeNoQStage = (kProbe & probe::kNoQStage) != 0;
  constexpr int kHeads = 64;
  constexpr int kHeadDim = 512;
  constexpr int kRopeDim = 64;
  constexpr int kSparseBlock = 64;
  constexpr int kSparseBlocks = 3;
  constexpr float kSoftmaxScale = 0.04419417382415922f;
  constexpr int kWmmaTile = 16;
  constexpr int kGroupsPerRow = kHeads / kGroupHeads;
  static_assert(
      kGroupHeads > 0 && kGroupHeads <= kWmmaTile
          && kHeads % kGroupHeads == 0,
      "head group must fill at most one wmma A tile");
  constexpr bool kOptCpAsync = (kOpt & opt::kCpAsync) != 0;
  constexpr bool kOptPad = (kOpt & opt::kPadShared) != 0;
  constexpr bool kOptHoistWeights = (kOpt & opt::kHoistWeights) != 0;

  // Shared leading dimensions. Unpadded they are the natural tile widths;
  // padded they satisfy (row bytes) % 128 == 16 so an 8-row ldmatrix group
  // spreads over all 32 banks, while staying legal wmma strides (multiple of
  // 8 elements for bf16, 4 for f32) and keeping every row 16-byte aligned.
  constexpr int kQStride = kOptPad ? kHeadDim + 8 : kHeadDim;      // bf16
  constexpr int kKvStride = kOptPad ? kHeadDim + 8 : kHeadDim;     // bf16
  constexpr int kWeightStride =
      kOptPad ? kSparseBlock + 8 : kSparseBlock;                   // bf16
  constexpr int kAccStride = kOptPad ? kHeadDim + 4 : kHeadDim;    // f32
  constexpr int kScoreStride =
      kOptPad ? kSparseBlock + 4 : kSparseBlock;                   // f32
  static_assert(kQStride % 8 == 0 && kKvStride % 8 == 0, "bf16 wmma ldm");
  static_assert(kWeightStride % 8 == 0, "bf16 wmma ldm");
  static_assert(kAccStride % 4 == 0 && kScoreStride % 4 == 0, "f32 wmma ldm");

  // Shared layout (float offsets from `shared`):
  //   q_tile        16 x kQStride      bf16  (padded head rows zeroed)
  //   kv_matrix     64 x kKvStride     bf16  (one KV block, staged per block)
  //   weight_matrix 16 x kWeightStride bf16  (per-head softmax weights)
  //   acc           16 x kAccStride    f32   (running PV accumulator)
  //   scores        16 x kScoreStride  f32
  //   row state     4  x 16            f32   (max, sum, rescale, inv_denom)
  __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(shared);
  __nv_bfloat16* kv_matrix = q_tile + kWmmaTile * kQStride;
  __nv_bfloat16* weight_matrix = kv_matrix + kSparseBlock * kKvStride;
  constexpr int kTensorFloats =
      (kWmmaTile * kQStride + kSparseBlock * kKvStride
       + kWmmaTile * kWeightStride) / 2;
  static_assert(
      (kWmmaTile * kQStride + kSparseBlock * kKvStride
       + kWmmaTile * kWeightStride) % 2 == 0,
      "tensor staging must be an even number of bf16 elements");
  static_assert(
      (kWmmaTile * kQStride) % 8 == 0
          && (kWmmaTile * kQStride + kSparseBlock * kKvStride) % 8 == 0,
      "each shared tile must start 16-byte aligned");
  float* acc = shared + kTensorFloats;
  float* scores = acc + kWmmaTile * kAccStride;
  float* row_max = scores + kWmmaTile * kScoreStride;
  float* row_sum = row_max + kWmmaTile;
  float* row_rescale = row_sum + kWmmaTile;
  float* row_inv_denominator = row_rescale + kWmmaTile;
  // Parallel-softmax scratch (allocated unconditionally; 4.2 KB).
  float* new_max = row_inv_denominator + kWmmaTile;
  float* max_scratch = new_max + kWmmaTile;
  constexpr int kSharedFloats = kTensorFloats + kWmmaTile * kAccStride
      + 2 * kWmmaTile * kScoreStride + 5 * kWmmaTile;
  static_assert(
      kSharedFloats * static_cast<int>(sizeof(float)) <= kSharedBytesBudget,
      "batched attention staging must fit dynamic shared memory");

  const int warp = static_cast<int>(threadIdx.x) / 32;
  constexpr int warps = kAttnWarps;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kGroupsPerRow;
    const int element = row / kDraftBlock;
    const __nv_bfloat16* element_target_kv =
        args.target_kv + element * kTargetKvElementStride;
    const __nv_bfloat16* element_draft_kv =
        args.draft_kv + element * kDraftBlock * 512;
    const int head_base =
        (static_cast<int>(item) % kGroupsPerRow) * kGroupHeads;
#if defined(DSPARK_V4_TP2_ABLATE)
    // Head-parallel ownership. The item axis interleaves rows and head
    // groups (item = row * groups_per_row + group), so ownership is a
    // per-item test rather than a range clip. Every intermediate the band
    // touches is head-disjoint (MAIN-BAND-MAP), so skipping the peer's heads
    // leaves this rank's heads untouched.
    if (!dspark_tp2_ablate::owns_head(head_base)) {
      continue;
    }
#endif
    const int target_count = args.start_pos + 1 < 128 ? args.start_pos + 1 : 128;
    const int topk = target_count + kDraftBlock;

    if constexpr (kProbeInit) {
      // Attribution baseline: every consumer sees defined shared memory even
      // when its producer is probed out.
      #pragma unroll 8
      for (int index = threadIdx.x; index < kTensorFloats; index += kAttnThreads) {
        shared[index] = 0.0f;
      }
      #pragma unroll
      for (int index = threadIdx.x; index < kWmmaTile * kScoreStride;
           index += kAttnThreads) {
        scores[index] = 0.0f;
        max_scratch[index] = 0.0f;
      }
      if (threadIdx.x < kWmmaTile) {
        row_rescale[threadIdx.x] = 1.0f;
        new_max[threadIdx.x] = 0.0f;
        row_inv_denominator[threadIdx.x] = 1.0f;
      }
    }
    // Stage the group's queries once per item (16-byte packets; padded A
    // rows zeroed so their score rows are finite garbage that softmax
    // masking then ignores).
    if constexpr (!kProbeNoQStage) {
      #pragma unroll
      for (int packet = threadIdx.x;
           packet < kWmmaTile * (kHeadDim / 8);
           packet += kAttnThreads) {
        const int head = packet / (kHeadDim / 8);
        const int column = (packet % (kHeadDim / 8)) * 8;
        if constexpr (kOptCpAsync) {
          // Same bytes, same destination; only the issue mechanism differs.
          // A dead A row zero-fills via the src-size operand, so its source
          // address is never read and only has to be in bounds.
          const bool live = head < kGroupHeads && !kProbeNoQLoad;
          cp_async_16_zfill(
              q_tile + head * kQStride + column,
              args.queries
                  + (row * kHeads + (live ? head_base + head : 0)) * kHeadDim
                  + column,
              live);
        } else {
          uint4 value = make_uint4(0u, 0u, 0u, 0u);
          if (head < kGroupHeads && !kProbeNoQLoad) {
            value = *reinterpret_cast<const uint4*>(
                args.queries
                + (row * kHeads + head_base + head) * kHeadDim + column);
          }
          *reinterpret_cast<uint4*>(q_tile + head * kQStride + column) = value;
        }
      }
      if constexpr (kOptCpAsync) {
        // Committed here, awaited by the first KV block's wait_group 0 --
        // q_tile is not read until after that wait and its __syncthreads.
        asm volatile("cp.async.commit_group;\n");
      }
    }
    if constexpr (!kProbeNoAccInit) {
      #pragma unroll 4
      for (int index = threadIdx.x; index < kWmmaTile * kAccStride;
           index += kAttnThreads) {
        acc[index] = 0.0f;
      }
    }
    if (threadIdx.x < kWmmaTile) {
      row_max[threadIdx.x] = -FLT_MAX;
      row_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int block = 0; block < kSparseBlocks; ++block) {
      if constexpr (kParallelSoftmax) {
        // A block entirely beyond topk is an exact no-op in the reference
        // (maximum unchanged, rescale expf(0) == 1, weights all zero).
        if (block * kSparseBlock >= topk) {
          continue;
        }
      }
      // KV staging, shared by the whole head group: 64 rows x 64 uint4
      // packets, masked rows zero-filled.
      if constexpr (!kProbeNoKvStage) {
        #pragma unroll 8
        for (int packet = threadIdx.x;
             packet < kSparseBlock * (kHeadDim / 8);
             packet += kAttnThreads) {
          const int position = packet / (kHeadDim / 8);
          const int column = (packet % (kHeadDim / 8)) * 8;
          const int ordinal = block * kSparseBlock + position;
          if constexpr (kOptCpAsync) {
            // Identical bytes to the scalar staging (a masked position
            // zero-fills through the src-size operand, exactly as the scalar
            // path stores make_uint4(0)); only the issue mechanism differs,
            // so the wmma operands are unchanged and the body stays bitwise.
            const bool from_target = ordinal < target_count;
            const bool live = ordinal < topk && !kProbeNoKvLoad;
            const __nv_bfloat16* source = from_target
                ? element_target_kv + ordinal * kHeadDim + column
                : element_draft_kv
                    + (live ? ordinal - target_count : 0) * kHeadDim + column;
            cp_async_16_zfill(
                kv_matrix + position * kKvStride + column, source, live);
          } else {
            // Probe: the substitute payload depends on `block` so the store
            // cannot be hoisted out of the KV-block loop, which would
            // silently credit the load probe with two thirds of the
            // shared-store cost.
            uint4 value = kProbeNoKvLoad
                ? make_uint4(static_cast<unsigned>(block) + 1u, 0u, 0u, 0u)
                : make_uint4(0u, 0u, 0u, 0u);
            if (!kProbeNoKvLoad) {
              if (ordinal < target_count) {
                value = *reinterpret_cast<const uint4*>(
                    element_target_kv + ordinal * kHeadDim + column);
              } else if (ordinal < topk) {
                value = *reinterpret_cast<const uint4*>(
                    element_draft_kv + (ordinal - target_count) * kHeadDim
                    + column);
              }
            }
            *reinterpret_cast<uint4*>(kv_matrix + position * kKvStride + column)
                = value;
          }
        }
        if constexpr (kOptCpAsync) {
          asm volatile("cp.async.commit_group;\n");
          asm volatile("cp.async.wait_group 0;\n");
        }
      }
      __syncthreads();

      if constexpr ((kOpt & opt::kMma8) != 0) {
        static_assert(kGroupHeads == 8);
        if (warp < kSparseBlock / kWmmaTile && !kProbeNoQk) {
          float d[4] = {0.f, 0.f, 0.f, 0.f};
          for (int kt = 0; kt < kHeadDim / 16; ++kt) {
            uint32_t a[4], b[2];
            dspark_attn_mma8_detail::ld_a(
                a, kv_matrix + warp * 16 * kKvStride + kt * 16, kKvStride);
            dspark_attn_mma8_detail::ld_b(b, q_tile + kt * 16, kQStride);
            dspark_attn_mma8_detail::mma8(d, a, b);
          }
          const int lane = threadIdx.x & 31;
      #pragma unroll
          for (int r = 0; r < 4; ++r) {
            const int pos = warp * 16 + lane / 4 + (r / 2) * 8;
            const int head = (lane % 4) * 2 + (r % 2);
            scores[head * kScoreStride + pos] = d[r];
          }
        }
      } else {
        // QK^T: warps 0-3 each own one 16-key score tile for all 16 A rows.
        if (warp < kSparseBlock / kWmmaTile && !kProbeNoQk) {
          using namespace nvcuda;
          wmma::fragment<
              wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, __nv_bfloat16,
              wmma::row_major>
              query_fragment;
          wmma::fragment<
              wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, __nv_bfloat16,
              wmma::col_major>
              kv_fragment;
          wmma::fragment<
              wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float>
              score_fragment;
          wmma::fill_fragment(score_fragment, 0.0f);
          for (int k_tile = 0; k_tile < kHeadDim / kWmmaTile; ++k_tile) {
            wmma::load_matrix_sync(
                query_fragment, q_tile + k_tile * kWmmaTile, kQStride);
            wmma::load_matrix_sync(
                kv_fragment,
                kv_matrix + warp * kWmmaTile * kKvStride + k_tile * kWmmaTile,
                kKvStride);
            wmma::mma_sync(
                score_fragment, query_fragment, kv_fragment, score_fragment);
          }
          wmma::store_matrix_sync(
              scores + warp * kWmmaTile,
              score_fragment,
              kScoreStride,
              wmma::mem_row_major);
        }
      }
      __syncthreads();

      if constexpr (kProbeNoSoftmax) {
        // Probe: keep the rescale factor finite so the accumulator pass and
        // the PV chain stay on fast arithmetic paths.
        if (threadIdx.x < kWmmaTile) {
          row_rescale[threadIdx.x] = 1.0f;
        }
      } else if constexpr (kParallelSoftmax) {
        // Exact64 tree, transported in warp registers. All32 lanes execute
        // each shuffle; only lanes below the current stride consume/update.
        // Head warp and warp+8 retain all16 rows, including padded PV rows.
        static_assert(kSparseBlock == 64 && kWmmaTile == 16);
        const int tree_lane = threadIdx.x & 31;
        const int tree_warp = threadIdx.x >> 5;
        #pragma unroll
        for (int head_half = 0; head_half < 2; ++head_half) {
          const int head = tree_warp + head_half * 8;
          const int slot = head * kScoreStride + tree_lane;
          const int ordinal = block * kSparseBlock + tree_lane;
          const float value0 = (head < kGroupHeads && ordinal < topk)
              ? scores[slot] * kSoftmaxScale : -FLT_MAX;
          const float value1 = (head < kGroupHeads && ordinal + 32 < topk)
              ? scores[slot + 32] * kSoftmaxScale : -FLT_MAX;
          float maximum_tree = fmaxf(value0, value1); // stride32
          #pragma unroll
          for (int stride = 16; stride > 0; stride /= 2) {
            const float peer = __shfl_down_sync(0xffffffffu, maximum_tree, stride);
            if (tree_lane < stride) maximum_tree = fmaxf(maximum_tree, peer);
          }
          float maximum = 0.0f;
          float rescale = 0.0f;
          if (tree_lane == 0) {
            const float previous_max = row_max[head];
            maximum = fmaxf(previous_max, maximum_tree);
            rescale = expf(previous_max - maximum);
            row_rescale[head] = rescale;
            new_max[head] = maximum;
          }
          maximum = __shfl_sync(0xffffffffu, maximum, 0);
          // Identical sentinel/exp/BF16 boundaries and unchanged online state.
          const float weight0 = value0 == -FLT_MAX ? 0.0f : expf(value0 - maximum);
          const float weight1 = value1 == -FLT_MAX ? 0.0f : expf(value1 - maximum);
          weight_matrix[head * kWeightStride + tree_lane] = __float2bfloat16_rn(weight0);
          weight_matrix[head * kWeightStride + tree_lane + 32] = __float2bfloat16_rn(weight1);
          float sum_tree = weight0 + weight1; // stride32, same left/right operands
          #pragma unroll
          for (int stride = 16; stride > 0; stride /= 2) {
            const float peer = __shfl_down_sync(0xffffffffu, sum_tree, stride);
            if (tree_lane < stride) sum_tree += peer;
          }
          if (tree_lane == 0) {
            row_sum[head] = row_sum[head] * rescale + sum_tree;
            row_max[head] = maximum;
          }
        }
        // The unchanged CTA join after this branch publishes every weight row
        // and row-state value before rescale/PV reads shared memory.
      } else
      // Online softmax, one thread per head row; op order per head matches
      // the reference thread-0 scan (including the stride-halving sum tree).
      if (threadIdx.x < kWmmaTile) {
        const int head = threadIdx.x;
        float* score_row = scores + head * kScoreStride;
        const bool live = head < kGroupHeads;
        for (int position = 0; position < kSparseBlock; ++position) {
          const int ordinal = block * kSparseBlock + position;
          score_row[position] = (live && ordinal < topk)
              ? score_row[position] * kSoftmaxScale
              : -FLT_MAX;
        }
        const float previous_max = row_max[head];
        float maximum = previous_max;
        for (int position = 0; position < kSparseBlock; ++position) {
          maximum = fmaxf(maximum, score_row[position]);
        }
        const float rescale = expf(previous_max - maximum);
        for (int position = 0; position < kSparseBlock; ++position) {
          const float weight = score_row[position] == -FLT_MAX
              ? 0.0f
              : expf(score_row[position] - maximum);
          weight_matrix[head * kWeightStride + position] =
              __float2bfloat16_rn(weight);
          score_row[position] = weight;
        }
        for (int stride = kSparseBlock / 2; stride > 0; stride /= 2) {
          for (int position = 0; position < stride; ++position) {
            score_row[position] += score_row[position + stride];
          }
        }
        row_max[head] = maximum;
        row_sum[head] = row_sum[head] * rescale + score_row[0];
        row_rescale[head] = rescale;
      }
      __syncthreads();

      // Rescale the running accumulator by each head's factor (the
      // reference fuses this multiply into its C-fragment load; same value,
      // done in-place here because the accumulator is shared-resident).
      if constexpr (!kProbeNoRescale) {
        #pragma unroll 4
        for (int index = threadIdx.x; index < kWmmaTile * kHeadDim;
             index += kAttnThreads) {
          const int head = index / kHeadDim;
          acc[head * kAccStride + index % kHeadDim] *= row_rescale[head];
        }
      }
      __syncthreads();

      if constexpr ((kOpt & opt::kMma8) != 0) {
        if constexpr (!kProbeNoPv) {
          const int lane = threadIdx.x & 31;
          uint32_t weights[4][2];
      #pragma unroll
          for (int kt = 0; kt < 4; ++kt)
            dspark_attn_mma8_detail::ld_b(weights[kt], weight_matrix + kt * 16,
                                          kWeightStride);
          for (int tile = warp; tile < kHeadDim / 16; tile += warps) {
            float d[4];
      #pragma unroll
            for (int r = 0; r < 4; ++r) {
              int feature = tile * 16 + lane / 4 + (r / 2) * 8;
              int head = (lane % 4) * 2 + (r % 2);
              d[r] = acc[head * kAccStride + feature];
            }
      #pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
              uint32_t a[4];
              dspark_attn_mma8_detail::ld_at(
                  a, kv_matrix + kt * 16 * kKvStride + tile * 16, kKvStride);
              dspark_attn_mma8_detail::mma8(d, a, weights[kt]);
            }
      #pragma unroll
            for (int r = 0; r < 4; ++r) {
              int feature = tile * 16 + lane / 4 + (r / 2) * 8;
              int head = (lane % 4) * 2 + (r % 2);
              acc[head * kAccStride + feature] = d[r];
            }
          }
        }
      } else {
        // PV: 32 output tiles across all warps, C chained in shared memory.
        if constexpr (!kProbeNoPv) {
          using namespace nvcuda;
          // The weight fragments do not depend on output_tile. kOptHoistWeights
          // lifts them out of the tile loop -- same operand values in the same
          // mma order, 12 fewer shared-memory fragment loads per warp per KV
          // block -- at the cost of 3 extra live bf16 A fragments (12 regs) in
          // a megakernel whose budget every band shares, so it is its own bit.
          constexpr int kPvKTiles = kSparseBlock / kWmmaTile;
          wmma::fragment<
              wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, __nv_bfloat16,
              wmma::row_major>
              weight_fragment[kOptHoistWeights ? kPvKTiles : 1];
          if constexpr (kOptHoistWeights) {
            for (int k_tile = 0; k_tile < kPvKTiles; ++k_tile) {
              wmma::load_matrix_sync(
                  weight_fragment[k_tile],
                  weight_matrix + k_tile * kWmmaTile,
                  kWeightStride);
            }
          }
          for (int output_tile = warp; output_tile < kHeadDim / kWmmaTile;
               output_tile += warps) {
            wmma::fragment<
                wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, __nv_bfloat16,
                wmma::row_major>
                value_fragment;
            wmma::fragment<
                wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float>
                output_fragment;
            wmma::load_matrix_sync(
                output_fragment,
                acc + output_tile * kWmmaTile,
                kAccStride,
                wmma::mem_row_major);
            for (int k_tile = 0; k_tile < kPvKTiles; ++k_tile) {
              if constexpr (!kOptHoistWeights) {
                wmma::load_matrix_sync(
                    weight_fragment[0],
                    weight_matrix + k_tile * kWmmaTile,
                    kWeightStride);
              }
              wmma::load_matrix_sync(
                  value_fragment,
                  kv_matrix + k_tile * kWmmaTile * kKvStride
                      + output_tile * kWmmaTile,
                  kKvStride);
              wmma::mma_sync(
                  output_fragment,
                  weight_fragment[kOptHoistWeights ? k_tile : 0],
                  value_fragment,
                  output_fragment);
            }
            wmma::store_matrix_sync(
                acc + output_tile * kWmmaTile,
                output_fragment,
                kAccStride,
                wmma::mem_row_major);
          }
        }
      }
      __syncthreads();
    }

    if constexpr (kProbeNoEpilogue) {
      __syncthreads();
      continue;
    }
    // The sink joins each head's denominator with no value vector.
    if (threadIdx.x < kGroupHeads) {
      const int head = threadIdx.x;
      row_inv_denominator[head] = 1.0f
          / (row_sum[head]
             + expf(args.attention_sink[head_base + head] - row_max[head]));
    }
    __syncthreads();
    #pragma unroll 8
    for (int index = threadIdx.x; index < kGroupHeads * kHeadDim;
         index += kAttnThreads) {
      const int head = index / kHeadDim;
      const int column = index % kHeadDim;
      const int output_index =
          (row * kHeads + head_base + head) * kHeadDim + column;
      const __nv_bfloat16 value = __float2bfloat16_rn(
          acc[head * kAccStride + column] * row_inv_denominator[head]);
      args.raw_output[output_index] = value;
      args.output[output_index] = value;
    }
    __syncthreads();

    // RoPE on the trailing 64 dims of every head in the group, rotating the
    // bf16-rounded output exactly as the reference does.
    #pragma unroll
    for (int index = threadIdx.x;
         index < kGroupHeads * (kRopeDim / 2);
         index += kAttnThreads) {
      const int head = index / (kRopeDim / 2);
      const int pair = index % (kRopeDim / 2);
      const int first = (row * kHeads + head_base + head) * kHeadDim
          + kHeadDim - kRopeDim + 2 * pair;
      // Batched: RoPE slot is the row's draft position (shared start_pos).
      const int rope_base = (1 + row % kDraftBlock) * kRopeDim;
      const float x0 = __bfloat162float(args.output[first]);
      const float x1 = __bfloat162float(args.output[first + 1]);
      const float cosine = args.rope_cos_sin[rope_base + pair * 2];
      const float sine = args.rope_cos_sin[rope_base + pair * 2 + 1];
      args.output[first] = __float2bfloat16_rn(x0 * cosine + x1 * sine);
      args.output[first + 1] = __float2bfloat16_rn(x1 * cosine - x0 * sine);
    }
    __syncthreads();
  }
}

}  // namespace dspark_attn
