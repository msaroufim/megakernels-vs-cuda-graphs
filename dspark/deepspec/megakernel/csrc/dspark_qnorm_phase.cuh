// Self-contained query-norm+RoPE phase bodies for the DeepSeek-V4 DSpark
// megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// Two bands (relaxed-drafter stage 3, iteration 9):
//   q_norm_rope   per-(row, head) RMSNorm over 512 dims with BF16 rounding
//                 at every scalar step (the square, the mean, mean+eps and
//                 the stored inverse RMS each round through BF16) + RoPE on
//                 the last 64 dims. 320 items (5 rows x 64 heads), claim
//                 chunk 2 in production. ~0.22 ms x 3 layers on the
//                 instrumented critical path — the last significant band
//                 never rewritten.
//   draft_kv_norm draft-KV finalize sibling: split-partial fold + RMSNorm
//                 (FP32 mean, unlike the query band) + RoPE + per-64-group
//                 FP8 fake-quant. 5 items; diagnostic-first per the
//                 ~0.05 ms bar.
//
// The execute_*_reference bodies are verbatim extractions of the production
// bodies from dspark_v4_kernel.cu (pure code motion; stage./task. became
// args.): each is its own oracle in the bench lane.
//
// Candidate rationale (execute_qnorm_batched<G>): the reference spends one
// whole 256-thread CTA on ONE head — 512 loaded values give each thread two
// elements, then an 8-level __syncthreads halving tree plus three more
// block barriers serialize the item (~12 block barriers per 512-dim head).
// The batched body gives each head ONE WARP and folds G heads into one item
// (retile 320 -> 320/G): the block reduction tree is replayed EXACTLY —
// each lane computes its 8-leaf subtree (deltas 128/64/32 of the reference
// tree) in the reference's combine order, and the remaining 5 levels
// (deltas 16..1) map 1:1 onto __shfl_down_sync steps — so every FP32 add
// keeps its operand pair and order and the swap is BITWISE, like iter-4's
// batched attention. No __syncthreads anywhere; one __syncwarp between the
// normalize stores and the RoPE pair reads.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

#include "dspark_batch.h"
#include "dspark_v4_tp2_ablate.cuh"

namespace dspark_qnorm {

constexpr int kDraftBlock = 5;
constexpr int kHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr float kNormEpsilon = 1.0e-6f;    // == kMainNormEpsilon
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;
// R10: both call sites launch 256 threads/CTA (generated::kThreads in the
// megakernel, kBenchThreads in the phase bench). blockDim.x is a runtime
// special register, so a loop strided or bounded by it has a trip count
// nvcc cannot see and cannot unroll. The constant is the same value, so
// every address and every accumulation order is unchanged.
constexpr int kQNormThreads = 256;

// ---------------------------------------------------------------------------
// q_norm_rope: per-(row, head) BF16-stepped RMSNorm + RoPE on the tail dims.
// ---------------------------------------------------------------------------

struct QNormArgs {
  // 5 x 64 x 512 BF16 query projections (q_b output).
  const __nv_bfloat16* query_projection;
  // 5 x 64 BF16 stored inverse RMS values.
  __nv_bfloat16* query_inverse_rms;
  // 5 x 64 x 512 BF16 normalized + roped queries.
  __nv_bfloat16* queries;
  // (1 + kDraftBlock) x 64 interleaved FP32 cos/sin; row uses slot 1 + row.
  const float* rope_cos_sin;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_query_norm_rope from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_qnorm_reference(
    const QNormArgs& args,
    float* shared) {
  constexpr int kHeads = 64;
  constexpr int kHeadDim = 512;
  constexpr int kRopeDim = 64;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kHeads;
    const int head = static_cast<int>(item) % kHeads;
    const int base = (row * kHeads + head) * kHeadDim;
    float local_square = 0.0f;
    #pragma unroll
    for (int column = threadIdx.x; column < kHeadDim; column += kQNormThreads) {
      const float value = __bfloat162float(args.query_projection[base + column]);
      const float squared = __bfloat162float(__float2bfloat16_rn(value * value));
      local_square += squared;
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    #pragma unroll
    for (int delta = kQNormThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] += shared[threadIdx.x + delta];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float mean =
          __bfloat162float(__float2bfloat16_rn(shared[0] / kHeadDim));
      const float with_epsilon =
          __bfloat162float(__float2bfloat16_rn(mean + kNormEpsilon));
      args.query_inverse_rms[row * kHeads + head] =
          __float2bfloat16_rn(rsqrtf(with_epsilon));
      shared[0] = __bfloat162float(args.query_inverse_rms[row * kHeads + head]);
    }
    __syncthreads();
    const float inverse_rms = shared[0];
    #pragma unroll
    for (int column = threadIdx.x; column < kHeadDim; column += kQNormThreads) {
      const float value = __bfloat162float(args.query_projection[base + column]);
      args.queries[base + column] = __float2bfloat16_rn(value * inverse_rms);
    }
    __syncthreads();
    if (threadIdx.x < kRopeDim / 2) {
      const int pair = threadIdx.x;
      const int first = base + kHeadDim - kRopeDim + 2 * pair;
      // Batched: RoPE slot is the row's draft position (shared start_pos).
      const int rope_base = (1 + row % kDraftBlock) * kRopeDim;
      const float x0 = __bfloat162float(args.queries[first]);
      const float x1 = __bfloat162float(args.queries[first + 1]);
      const float cosine = args.rope_cos_sin[rope_base + pair * 2];
      const float sine = args.rope_cos_sin[rope_base + pair * 2 + 1];
      args.queries[first] = __float2bfloat16_rn(x0 * cosine - x1 * sine);
      args.queries[first + 1] = __float2bfloat16_rn(x1 * cosine + x0 * sine);
    }
    __syncthreads();
  }
}

// One warp per head, kGroup heads per item (retile 320 -> 320 / kGroup;
// item -> row = item / (64 / kGroup), head = (item % (64 / kGroup)) * kGroup
// + warp). BITWISE vs the reference by construction:
//   - The reference's 256-leaf halving tree (leaf t = (0 + sq(t)) +
//     sq(t + 256), then shared[i] += shared[i + delta] for delta = 128..1)
//     is replayed exactly: lane l owns leaves {l + 32j, j = 0..7}, computes
//     the delta = 128/64/32 levels of its subtree in the reference's combine
//     order, and the delta = 16..1 levels are __shfl_down_sync adds with the
//     identical operand pairing. Every FP32 add keeps its operands and
//     order, so lane 0's sum is bit-identical to shared[0].
//   - Every BF16 rounding (square, mean, mean + eps, stored inverse RMS,
//     normalize, RoPE) is the reference's op-for-op scalar chain.
// The RoPE pair reads cross lanes within the warp (column parity vs lane
// index), hence the single __syncwarp after the normalize stores. No block
// barriers: warps proceed fully independently.
template <int kGroup>
__device__ inline void execute_qnorm_batched(const QNormArgs& args) {
  static_assert(kGroup >= 1 && kGroup <= 8, "one warp per head, 256 threads");
  static_assert(kHeads % kGroup == 0, "head groups must tile the head count");
  constexpr int kGroupsPerRow = kHeads / kGroup;
  constexpr int kLaneColumns = kHeadDim / 32;  // 16 columns per lane
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  if (warp >= kGroup) {
    return;
  }
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kGroupsPerRow;
    const int head =
        (static_cast<int>(item) % kGroupsPerRow) * kGroup + warp;
#if defined(DSPARK_V4_TP2_ABLATE)
    if (!dspark_tp2_ablate::owns_head(head)) {
      continue;
    }
#endif
    const int base = (row * kHeads + head) * kHeadDim;
    // Lane l loads columns {l + 32k, k = 0..15}: coalesced, and exactly the
    // operand set of its 8 reduction-tree leaves (leaf p uses columns p and
    // p + 256, i.e. k = j and k = j + 8 for p = l + 32j).
    float value[kLaneColumns];
    float squared[kLaneColumns];
#pragma unroll
    for (int k = 0; k < kLaneColumns; ++k) {
      value[k] = __bfloat162float(args.query_projection[base + lane + 32 * k]);
      squared[k] =
          __bfloat162float(__float2bfloat16_rn(value[k] * value[k]));
    }
    // Reference leaf t: local_square = (0 + sq(t)) + sq(t + 256).
    float leaf[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      leaf[j] = (0.0f + squared[j]) + squared[j + 8];
    }
    // Reference tree levels delta = 128 / 64 / 32 for this lane's subtree:
    //   a_i = leaf(i) + leaf(i + 128); b_i = a_i + a_{i+64};
    //   c_i = b_i + b_{i+32}   (indices in units of 32 -> j offsets 4/2/1).
    const float a0 = leaf[0] + leaf[4];
    const float a1 = leaf[1] + leaf[5];
    const float a2 = leaf[2] + leaf[6];
    const float a3 = leaf[3] + leaf[7];
    const float b0 = a0 + a2;
    const float b1 = a1 + a3;
    float sum = b0 + b1;
    // Reference tree levels delta = 16..1: shared[i] += shared[i + delta].
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    float inverse_rms = 0.0f;
    if (lane == 0) {
      const float mean =
          __bfloat162float(__float2bfloat16_rn(sum / kHeadDim));
      const float with_epsilon =
          __bfloat162float(__float2bfloat16_rn(mean + kNormEpsilon));
      const __nv_bfloat16 stored = __float2bfloat16_rn(rsqrtf(with_epsilon));
      args.query_inverse_rms[row * kHeads + head] = stored;
      inverse_rms = __bfloat162float(stored);
    }
    inverse_rms = __shfl_sync(0xffffffffu, inverse_rms, 0);
#pragma unroll
    for (int k = 0; k < kLaneColumns; ++k) {
      args.queries[base + lane + 32 * k] =
          __float2bfloat16_rn(value[k] * inverse_rms);
    }
    __syncwarp();
    // 32 RoPE pairs, one per lane (the reference's threadIdx.x < 32 branch).
    const int pair = lane;
    const int first = base + kHeadDim - kRopeDim + 2 * pair;
    // Batched serving: rows are flat over (batch, block), and every batch
    // element shares start_pos in this specialization, so the RoPE slot is
    // the row's DRAFT position, not its flat index.
    const int rope_base = (1 + row % kDraftBlock) * kRopeDim;
    const float x0 = __bfloat162float(args.queries[first]);
    const float x1 = __bfloat162float(args.queries[first + 1]);
    const float cosine = args.rope_cos_sin[rope_base + pair * 2];
    const float sine = args.rope_cos_sin[rope_base + pair * 2 + 1];
    args.queries[first] = __float2bfloat16_rn(x0 * cosine - x1 * sine);
    args.queries[first + 1] = __float2bfloat16_rn(x1 * cosine + x0 * sine);
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// draft_kv_norm_rope_quant: draft-KV finalize (5 items). Diagnostic-first.
// ---------------------------------------------------------------------------

struct DraftKvNormArgs {
  // (row * 512 + column) * 4 + split FP32 split partials.
  const float* draft_kv_partials;
  // 512 FP32 norm weights.
  const float* kv_norm_weight;
  // 5 x 512 BF16 finalized draft KV rows.
  __nv_bfloat16* draft_kv;
  // (1 + kDraftBlock) x 64 interleaved FP32 cos/sin; row uses slot 1 + row.
  const float* rope_cos_sin;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_draft_kv_finalize from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_draft_kv_norm_reference(
    const DraftKvNormArgs& args,
    float* shared) {
  constexpr int kSplits = 4;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item);
    const int base = row * 512;
    float local_square = 0.0f;
    #pragma unroll
    for (int column = threadIdx.x; column < 512; column += kQNormThreads) {
      float value = 0.0f;
#pragma unroll
      for (int split = 0; split < kSplits; ++split) {
        value += args.draft_kv_partials[(row * 512 + column) * kSplits + split];
      }
      const float rounded = __bfloat162float(__float2bfloat16_rn(value));
      shared[256 + column] = rounded;
      local_square = fmaf(rounded, rounded, local_square);
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    #pragma unroll
    for (int delta = kQNormThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] += shared[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms = rsqrtf(shared[0] / 512.0f + kNormEpsilon);
    #pragma unroll
    for (int column = threadIdx.x; column < 512; column += kQNormThreads) {
      args.draft_kv[base + column] = __float2bfloat16_rn(
          shared[256 + column] * inverse_rms * args.kv_norm_weight[column]);
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      const int pair = threadIdx.x;
      const int first = base + 448 + 2 * pair;
      // Batched: RoPE slot is the row's draft position (shared start_pos).
      const int rope_base = (1 + row % kDraftBlock) * 64;
      const float x0 = __bfloat162float(args.draft_kv[first]);
      const float x1 = __bfloat162float(args.draft_kv[first + 1]);
      const float cosine = args.rope_cos_sin[rope_base + pair * 2];
      const float sine = args.rope_cos_sin[rope_base + pair * 2 + 1];
      args.draft_kv[first] = __float2bfloat16_rn(x0 * cosine - x1 * sine);
      args.draft_kv[first + 1] = __float2bfloat16_rn(x1 * cosine + x0 * sine);
    }
    __syncthreads();

    cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    // All normalized/RoPE stores are published by the unchanged preceding join.
    // Each of seven warps owns one disjoint 64-element quantization group.
    if (warp < 7) {
      const int index = base + warp * 64 + lane;
      const float value0 = __bfloat162float(args.draft_kv[index]);
      const float value1 = __bfloat162float(args.draft_kv[index + 32]);
      // The original deltas128/64 combine nonnegative magnitudes with zero.
      // Delta32 and deltas16..1 retain the same operand pairs and order.
      float maximum = fmaxf(fabsf(value0), fabsf(value1));
      #pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        const float other = __shfl_down_sync(0xffffffffu, maximum, delta);
        if (lane < delta) maximum = fmaxf(maximum, other);
      }
      float scale = 0.0f;
      if (lane == 0) {
        const float amax = fmaxf(maximum, 1.0e-4f);
        const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
        scale = ldexpf(1.0f, exponent);
      }
      scale = __shfl_sync(0xffffffffu, scale, 0);
      const float quantized0 = static_cast<float>(convert(value0 / scale)) * scale;
      const float quantized1 = static_cast<float>(convert(value1 / scale)) * scale;
      args.draft_kv[index] = __float2bfloat16_rn(quantized0);
      args.draft_kv[index + 32] = __float2bfloat16_rn(quantized1);
    }
    // Preserve whole-CTA publication and shared-arena lifetime for the next row.
    __syncthreads();
  }
}

}  // namespace dspark_qnorm
