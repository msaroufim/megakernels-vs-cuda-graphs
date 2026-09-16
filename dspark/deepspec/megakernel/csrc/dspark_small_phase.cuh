// Self-contained small dense phase bodies (q_a, main_kv, draft_kv, HC
// projections) for the DeepSeek-V4 DSpark megakernel, compiled by both the
// production kernel and the phase microbenchmark (see dspark_w13_phase.cuh
// for the pattern rationale).
//
// Four bands, each ~0.15-0.25 ms x 3 layers on the critical path:
//   q_a       FP8 GEMM 4096 -> 1024, split-K = 4, 5 rows, FP32 split
//             partials; item = (output_tile, split) pair (32/layer).
//   main_kv   FP8 GEMM 4096 -> 512, split-K = 4, ONE row (the main token),
//             FP32 split partials; item = (output_tile, split) (16/layer).
//   draft_kv  FP8 GEMM 4096 -> 512, split-K = 4, 5 rows, FP32 split
//             partials; item = (output_tile, split) pair (16/layer).
//   hc        attn/ffn HC projection: 120 (row, mix) items per phase, each
//             ONE lane replaying a 16,384-step serial FP32 chain from
//             shared-staged operands (the #92 body), scaled by the row's
//             inverse RMS.
//
// The execute_*_reference bodies are verbatim extractions of the production
// bodies from dspark_v4_kernel.cu (pure code motion; stage./task. became
// args., and the main_kv layer indirection is resolved by the wrapper into
// pre-offset pointers): each is its own oracle in the bench lane.
#pragma once

#include "dspark_tmem.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

// TCGen05 candidate precedent: the FP8 E4M3 x E4M3 UMMA pipeline with
// per-128-block scale chaining (dspark_shared_phase.cuh) and its iter-5
// clones in dspark_proj_phase.cuh. q_a and the kv projections are the same
// (output_tile, split) split-K pattern as wo_b at smaller N.
#include "dspark_shared_phase.cuh"
#include "dspark_batch.h"

namespace dspark_small {

constexpr int kDraftBlock = 5;
constexpr int kHidden = 4096;      // == kMainHidden
constexpr int kQuantBlock = 128;   // == kMainQuantBlock
constexpr int kOutputTile = 128;   // == kMainOutputTile
constexpr int kSplits = 4;
constexpr int kBlocks = kHidden / kQuantBlock;           // 32
constexpr int kBlocksPerSplit = kBlocks / kSplits;       // 8
constexpr int kQaRank = 1024;
constexpr int kKvWidth = 512;
constexpr int kHcStreams = 4;
constexpr int kHcMixes = 24;
constexpr int kHcFlattened = kHcStreams * kHidden;       // 16384
// Both HC call sites launch 256 threads/CTA (generated::kThreads in the
// megakernel, kBenchThreads in the phase bench; each TU static_asserts the
// match). Spelling it as a constant instead of reading blockDim.x is the
// whole point: blockDim.x is a runtime special register, so the 16,384-wide
// strided loop had an unknown trip count and nvcc could not unroll it --
// one load, one full memory-latency stall, 64 times over. See the
// kHcPerThread bodies below.
constexpr int kHcThreads = 256;
constexpr int kHcPerThread = kHcFlattened / kHcThreads;  // 64
constexpr float kNormEpsilon = 1.0e-6f;                  // == kMainNormEpsilon
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;

// Bit-exact E8M0 decode, verbatim from dspark_v4_kernel.cu (#71/#72).
__device__ __forceinline__ float decode_e8m0(uint8_t bits) {
  const uint32_t exponent = bits;
  const uint32_t float_bits =
      exponent == 0 ? 0x00400000U : exponent << 23;
  return __uint_as_float(float_bits);
}

// ---------------------------------------------------------------------------
// q_a: query down-projection, FP8 4096 -> 1024, split-K = 4, 5 rows.
// ---------------------------------------------------------------------------

struct QaArgs {
  // 5 x 4096 E4M3 activations + 5 x 32 E8M0 scales.
  const cutlass::float_e4m3_t* input_quantized;
  const uint8_t* input_scales;
  // 1024 x 4096 E4M3 weights, per-(128-output, 128-K) E8M0 block scales.
  const cutlass::float_e4m3_t* wqa;
  const uint8_t* wqa_scales;
  // FP32 split partials, (row * 1024 + output) * 4 + split.
  float* qa_partials;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_qa_projection from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_qa_reference(const QaArgs& args) {
  constexpr int kRank = kQaRank;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int split = static_cast<int>(item) % kSplits;
    const int output_tile = static_cast<int>(item) / kSplits;
    const int output_column = output_tile * kOutputTile + threadIdx.x;
    if (threadIdx.x >= kOutputTile || output_column >= kRank) {
      continue;
    }
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = 0; block < kBlocksPerSplit; ++block) {
      const int k_block = split * kBlocksPerSplit + block;
      const int weight_base =
          output_column * kHidden + k_block * kQuantBlock;
      float inner[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int offset = 0; offset < kQuantBlock; ++offset) {
        const float weight_value =
            static_cast<float>(args.wqa[weight_base + offset]);
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          inner[row] = fmaf(
              static_cast<float>(args.input_quantized[
                  row * kHidden + k_block * kQuantBlock + offset]),
              weight_value,
              inner[row]);
        }
      }
      const float weight_scale =
          decode_e8m0(args.wqa_scales[(output_column / 128) * 32 + k_block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale =
            decode_e8m0(args.input_scales[row * 32 + k_block]);
        result[row] = fmaf(
            inner[row], input_scale * weight_scale, result[row]);
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.qa_partials[(row * kRank + output_column) * kSplits + split] =
          result[row];
    }
  }
}

// ---------------------------------------------------------------------------
// main_kv: main-token KV projection, FP8 4096 -> 512, split-K = 4, ONE row.
// The wrapper resolves the per-layer weight/scale/partials pointers, so the
// body indexes partials at (output_column * 4 + split) directly.
// ---------------------------------------------------------------------------

struct MainKvArgs {
  // 1 x 4096 E4M3 activations + 32 E8M0 scales (single main-token row).
  const cutlass::float_e4m3_t* quantized;
  const uint8_t* activation_scales;
  // 512 x 4096 E4M3 weights (layer-resolved), per-(128-out, 128-K) scales.
  const cutlass::float_e4m3_t* weight;
  const uint8_t* weight_scales;
  // FP32 split partials, layer-resolved base: output * 4 + split.
  float* partials;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_main_kv_projection from
// dspark_v4_kernel.cu (pure code motion; stage. became args. and the
// [layer] indirections became the wrapper's pre-offset pointers).
__device__ inline void execute_main_kv_reference(const MainKvArgs& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int split = static_cast<int>(item) % kSplits;
    const int output_tile = static_cast<int>(item) / kSplits;
    const int output_column = output_tile * kOutputTile + threadIdx.x;
    if (threadIdx.x < kOutputTile && output_column < 512) {
      float result = 0.0f;
      for (int block = 0; block < kBlocksPerSplit; ++block) {
        const int k_block = split * kBlocksPerSplit + block;
        const int input_base = k_block * kQuantBlock;
        const int weight_base = output_column * kHidden + input_base;
        float inner = 0.0f;
#pragma unroll
        for (int offset = 0; offset < kQuantBlock; ++offset) {
          inner = fmaf(
              static_cast<float>(args.quantized[input_base + offset]),
              static_cast<float>(args.weight[weight_base + offset]),
              inner);
        }
        const float input_scale = decode_e8m0(args.activation_scales[k_block]);
        const float weight_scale = decode_e8m0(
            args.weight_scales[(output_column / kQuantBlock) * 32 + k_block]);
        result = fmaf(inner, input_scale * weight_scale, result);
      }
      args.partials[output_column * kSplits + split] = result;
    }
  }
}

// ---------------------------------------------------------------------------
// draft_kv: draft-row KV projection, FP8 4096 -> 512, split-K = 4, 5 rows.
// ---------------------------------------------------------------------------

struct DraftKvArgs {
  // 5 x 4096 E4M3 activations + 5 x 32 E8M0 scales.
  const cutlass::float_e4m3_t* input_quantized;
  const uint8_t* input_scales;
  // 512 x 4096 E4M3 weights, per-(128-output, 128-K) E8M0 block scales.
  const cutlass::float_e4m3_t* wkv;
  const uint8_t* wkv_scales;
  // FP32 split partials, (row * 512 + output) * 4 + split.
  float* draft_kv_partials;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_draft_kv_projection from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_draft_kv_reference(const DraftKvArgs& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int split = static_cast<int>(item) % kSplits;
    const int output_tile = static_cast<int>(item) / kSplits;
    const int output_column = output_tile * kOutputTile + threadIdx.x;
    if (threadIdx.x >= kOutputTile || output_column >= 512) {
      continue;
    }
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = 0; block < kBlocksPerSplit; ++block) {
      const int k_block = split * kBlocksPerSplit + block;
      const int weight_base =
          output_column * kHidden + k_block * kQuantBlock;
      float inner[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int offset = 0; offset < kQuantBlock; ++offset) {
        const float weight_value =
            static_cast<float>(args.wkv[weight_base + offset]);
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          inner[row] = fmaf(
              static_cast<float>(args.input_quantized[
                  row * kHidden + k_block * kQuantBlock + offset]),
              weight_value,
              inner[row]);
        }
      }
      const float weight_scale =
          decode_e8m0(args.wkv_scales[(output_column / 128) * 32 + k_block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale =
            decode_e8m0(args.input_scales[row * 32 + k_block]);
        result[row] = fmaf(
            inner[row], input_scale * weight_scale, result[row]);
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.draft_kv_partials[(row * 512 + output_column) * kSplits + split] =
          result[row];
    }
  }
}

// ---------------------------------------------------------------------------
// hc: attn/ffn HC projection, 24 mixes of 4 x 4096 flattened streams.
// ---------------------------------------------------------------------------

struct HcArgs {
  // 5 x 16384 BF16 flattened streams (kHcStreams x kMainHidden per row).
  const __nv_bfloat16* streams;
  // 24 x 16384 FP32 mixing weights.
  const float* fn_weight;
  // 5 x 24 FP32 outputs.
  float* mixes;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_attn_hc_projection from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.; the
// kDynamicSharedBytes static_assert moved here with the same value).
__device__ inline void execute_hc_reference(const HcArgs& args, float* shared) {
  constexpr int kMixes = kHcMixes;
  constexpr int kFlattened = kHcFlattened;
  static_assert(
      (256 + kFlattened) * sizeof(float) + kFlattened * sizeof(__nv_bfloat16)
          <= kSharedBytesBudget,
      "HC projection staging must fit dynamic shared memory");
  // One task per (row, mix) output so the 120 chains spread across the grid
  // (the retained single-task body serialized them on one CTA at ~1.16 ms).
  // Numerics are preserved bitwise: the inverse-RMS keeps its 256-lane
  // strided + tree order, and the 16,384-step dot keeps its original single
  // serial FMA chain, replayed by one lane out of shared-staged operands.
  float* reduction_scratch = shared;
  float* staged_weights = shared + 256;
  __nv_bfloat16* staged_streams =
      reinterpret_cast<__nv_bfloat16*>(staged_weights + kFlattened);
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kMixes;
    const int mix = static_cast<int>(item) % kMixes;
    const int row_base = row * kFlattened;
    const int weight_base = mix * kFlattened;
    for (int column = threadIdx.x; column < kFlattened;
         column += blockDim.x) {
      staged_weights[column] = args.fn_weight[weight_base + column];
      staged_streams[column] = args.streams[row_base + column];
    }
    float local_square = 0.0f;
    for (int column = threadIdx.x; column < kFlattened; column += blockDim.x) {
      const float value = __bfloat162float(args.streams[row_base + column]);
      local_square = fmaf(value, value, local_square);
    }
    reduction_scratch[threadIdx.x] = local_square;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction_scratch[threadIdx.x] += reduction_scratch[threadIdx.x + delta];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float inverse_rms =
          rsqrtf(reduction_scratch[0] / kFlattened + kNormEpsilon);
      float result = 0.0f;
      for (int column = 0; column < kFlattened; ++column) {
        result = fmaf(
            __bfloat162float(staged_streams[column]),
            staged_weights[column],
            result);
      }
      args.mixes[row * kMixes + mix] = result * inverse_rms;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// CANDIDATE bodies (relaxed program stage 3, small-band iteration).
// ---------------------------------------------------------------------------

// hc parallel-tree candidate (relaxed: the serial-order constraint is gone).
// DROP-IN for the existing 120-item phase tiling. The inverse-RMS replays
// the reference's exact strided + halving-tree order (bitwise identical);
// the 16,384-step serial dot becomes 256 strided per-thread FP32 fmaf
// chains folded by the same halving tree. Operands stream straight from
// GMEM (each element is consumed exactly once per item, so SMEM staging
// buys nothing). Drift vs the serial chain is FP32 accumulation-order only.
__device__ inline void execute_hc_parallel(const HcArgs& args, float* shared) {
  constexpr int kMixes = kHcMixes;
  constexpr int kFlattened = kHcFlattened;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kMixes;
    const int mix = static_cast<int>(item) % kMixes;
    const int row_base = row * kFlattened;
    const int weight_base = mix * kFlattened;
    // R1 leg 6, intra-node retile: the RMS pass and the weighted dot pass walk
    // the SAME 16,384 ascending columns, so they ride one loop instead of two.
    // Each thread still accumulates local_square and local_dot over exactly
    // the ascending column sequence it owned before -- neither chain is
    // reassociated, only interleaved -- but the pass now has three independent
    // loads in flight per column instead of one, and the band pays one
    // dependent-load traversal of the row rather than two back to back. The
    // rms result is not consumed until after the dot's tree, so nothing
    // downstream observes the reordering.
    float local_square = 0.0f;
    float local_dot = 0.0f;
    for (int column = threadIdx.x; column < kFlattened; column += blockDim.x) {
      const float value = __bfloat162float(args.streams[row_base + column]);
      local_square = fmaf(value, value, local_square);
      local_dot = fmaf(value, args.fn_weight[weight_base + column], local_dot);
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] += shared[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(shared[0] / kFlattened + kNormEpsilon);
    __syncthreads();
    shared[threadIdx.x] = local_dot;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] += shared[threadIdx.x + delta];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      args.mixes[row * kMixes + mix] = shared[0] * inverse_rms;
    }
    __syncthreads();
  }
}

// hc row-batched candidate: item = row (retile 120 -> 5 per phase). The
// row's streams stage once through SMEM (32 KB BF16) and the inverse-RMS is
// computed once (same bitwise replay from GMEM as the reference); the 24
// mixes then stream their FP32 weight rows from GMEM through the same
// parallel chains + halving tree. Removes the 24x stream re-read and the
// per-item RMS recompute at the cost of 24x fewer claimable items.
__device__ inline void execute_hc_rowbatch(const HcArgs& args, float* shared) {
  constexpr int kMixes = kHcMixes;
  constexpr int kFlattened = kHcFlattened;
  static_assert(
      256 * sizeof(float) + kFlattened * sizeof(__nv_bfloat16)
          <= kSharedBytesBudget,
      "HC row-batch staging must fit dynamic shared memory");
  float* reduction_scratch = shared;
  __nv_bfloat16* staged_streams =
      reinterpret_cast<__nv_bfloat16*>(shared + 256);
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item);
    const int row_base = row * kFlattened;
    for (int column = threadIdx.x; column < kFlattened;
         column += blockDim.x) {
      staged_streams[column] = args.streams[row_base + column];
    }
    float local_square = 0.0f;
    for (int column = threadIdx.x; column < kFlattened; column += blockDim.x) {
      const float value = __bfloat162float(args.streams[row_base + column]);
      local_square = fmaf(value, value, local_square);
    }
    reduction_scratch[threadIdx.x] = local_square;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        reduction_scratch[threadIdx.x] += reduction_scratch[threadIdx.x + delta];
      }
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(reduction_scratch[0] / kFlattened + kNormEpsilon);
    __syncthreads();
    for (int mix = 0; mix < kMixes; ++mix) {
      const int weight_base = mix * kFlattened;
      float local_dot = 0.0f;
      for (int column = threadIdx.x; column < kFlattened;
           column += blockDim.x) {
        local_dot = fmaf(
            __bfloat162float(staged_streams[column]),
            args.fn_weight[weight_base + column],
            local_dot);
      }
      reduction_scratch[threadIdx.x] = local_dot;
      __syncthreads();
      for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
        if (threadIdx.x < delta) {
          reduction_scratch[threadIdx.x] +=
              reduction_scratch[threadIdx.x + delta];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        args.mixes[row * kMixes + mix] =
            reduction_scratch[0] * inverse_rms;
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// hc memory-level-parallelism candidates (R7).
//
// ncu on the shipped execute_hc_parallel body (hc_bench_kernel, 17.152 us,
// 459 PC samples) reported long_scoreboard 65.6%, barrier 2.8%, MIO throttle
// 0.0%, shared bank conflicts 0.0/wavefront, L2 sector actual/ideal 1.000,
// DRAM 1.29% of peak. Access patterns are already perfect and bandwidth is
// nearly unused: the band is not moving too many bytes, it is moving them
// ONE AT A TIME. 16,384 / 256 = 64 loop trips x ~full L2 latency with a
// single load outstanding is ~17 us on the nose, which is the whole measured
// band.
//
// The trip count was unknown to the compiler only because the loop bound was
// blockDim.x, a runtime special register. Every body below fixes the count
// at kHcPerThread so nvcc can unroll and keep kUnroll loads in flight.
//
// BITWISE CONTRACT for all of them: thread t still accumulates exactly the
// ascending column sequence t, t + 256, ..., t + 16128 into one fmaf chain,
// and the 8-level halving tree over 256 lanes is unchanged. Unrolling,
// staging a row through SMEM, and moving a mix to a different CTA relocate
// DATA and ISSUE ORDER, never arithmetic order, so every candidate here must
// be bitwise-identical to execute_hc_parallel.
// ---------------------------------------------------------------------------

// Candidate 1 -- pure issue-rate fix, no data movement change at all.
// Identical loads to execute_hc_parallel (streams BF16 + fn_weight FP32
// straight from GMEM, same addresses, same order), identical arithmetic,
// only the loop is now a compile-time 64 trips unrolled kUnroll deep. This
// exists to separate "the compiler could not unroll" from every structural
// change below.
template <int kUnroll>
__device__ inline void execute_hc_unroll(const HcArgs& args, float* shared) {
  constexpr int kMixes = kHcMixes;
  constexpr int kFlattened = kHcFlattened;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kMixes;
    const int mix = static_cast<int>(item) % kMixes;
    const __nv_bfloat16* stream_row = args.streams + row * kFlattened;
    const float* weight_row = args.fn_weight + mix * kFlattened;
    const int lane = static_cast<int>(threadIdx.x);
    float local_square = 0.0f;
    float local_dot = 0.0f;
#pragma unroll kUnroll
    for (int step = 0; step < kHcPerThread; ++step) {
      const int column = lane + step * kHcThreads;
      const float value = __bfloat162float(stream_row[column]);
      local_square = fmaf(value, value, local_square);
      local_dot = fmaf(value, weight_row[column], local_dot);
    }
    shared[lane] = local_square;
    __syncthreads();
#pragma unroll
    for (int delta = kHcThreads / 2; delta > 0; delta /= 2) {
      if (lane < delta) {
        shared[lane] += shared[lane + delta];
      }
      __syncthreads();
    }
    const float inverse_rms = rsqrtf(shared[0] / kFlattened + kNormEpsilon);
    __syncthreads();
    shared[lane] = local_dot;
    __syncthreads();
#pragma unroll
    for (int delta = kHcThreads / 2; delta > 0; delta /= 2) {
      if (lane < delta) {
        shared[lane] += shared[lane + delta];
      }
      __syncthreads();
    }
    if (lane == 0) {
      args.mixes[row * kMixes + mix] = shared[0] * inverse_rms;
    }
    __syncthreads();
  }
}

// Candidate 2 -- kGroup mixes per CTA over one SMEM-cached stream row.
// item = (row, mix group); the phase tiling goes 120 -> 120/kGroup.
//
// Three effects stack, and the bench measures the curve so they can be
// separated: (a) the 32 KiB stream row is fetched once per item instead of
// once per mix, as 16-byte vector packets whose only ordering requirement is
// "all of them land before the barrier" -- maximal MLP by construction;
// (b) the dot pass carries kGroup independent accumulators, so kGroup *
// kUnroll weight loads are outstanding at once instead of one; (c) the
// inverse-RMS is computed once per item instead of once per mix, out of SMEM.
// Against that, the grid loses CTAs (kGroup = 2 keeps 60, kGroup = 24
// collapses to 5), so the isolated span is a real tradeoff and only the
// sweep settles it.
//
// kGroup == 1 is a legal instantiation and is NOT a no-op: it is candidate 1
// plus the SMEM row cache, which is how the staging cost gets priced on its
// own.
template <int kGroup, bool kCpAsync = false, int kUnrollOverride = 0>
__device__ inline void execute_hc_mixgroup(const HcArgs& args, float* shared) {
  constexpr int kMixes = kHcMixes;
  constexpr int kFlattened = kHcFlattened;
  static_assert(kMixes % kGroup == 0, "mix group must divide the 24 mixes");
  constexpr int kGroups = kMixes / kGroup;
  // Keep total loads in flight near 8-12 so the register window stays small
  // enough not to tax the megakernel's 252-register budget.
  constexpr int kUnroll =
      kUnrollOverride > 0
          ? kUnrollOverride
          : (kGroup <= 2 ? 8 / kGroup
                         : (kGroup <= 4 ? 12 / kGroup : (kGroup <= 8 ? 2 : 1)));
  // Scratch holds the square tree plus one tree per mix, all folded by a
  // single 8-level barrier pass (independent trees, unchanged order).
  constexpr int kScratchFloats = (kGroup + 1) * kHcThreads;
  constexpr int kPackets = kFlattened * sizeof(__nv_bfloat16) / 16;  // 2048
  constexpr int kPacketsPerThread = kPackets / kHcThreads;           // 8
  static_assert(
      kScratchFloats * sizeof(float) + kFlattened * sizeof(__nv_bfloat16)
          <= kSharedBytesBudget,
      "HC mix-group staging must fit dynamic shared memory");
  float* reduction_scratch = shared;
  __nv_bfloat16* staged_streams =
      reinterpret_cast<__nv_bfloat16*>(shared + kScratchFloats);
  const int lane = static_cast<int>(threadIdx.x);
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kGroups;
    const int group = static_cast<int>(item) % kGroups;
    // (a) cache the row. Pure data movement -- the BF16 bit patterns are
    // copied verbatim, so every value the chains below consume is the one
    // execute_hc_parallel would have loaded from GMEM.
    {
      const uint4* source =
          reinterpret_cast<const uint4*>(args.streams + row * kFlattened);
      uint4* destination = reinterpret_cast<uint4*>(staged_streams);
      if (kCpAsync) {
#pragma unroll
        for (int packet = 0; packet < kPacketsPerThread; ++packet) {
          const int index = lane + packet * kHcThreads;
          const unsigned address = static_cast<unsigned>(
              __cvta_generic_to_shared(destination + index));
          asm volatile(
              "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address),
              "l"(source + index));
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
      } else {
#pragma unroll
        for (int packet = 0; packet < kPacketsPerThread; ++packet) {
          const int index = lane + packet * kHcThreads;
          destination[index] = source[index];
        }
      }
    }
    __syncthreads();
    // One base pointer plus compile-time slot offsets: a kGroup-wide array of
    // pointers would burn 2 * kGroup registers for nothing (kGroup = 24 alone
    // would be 48), and slot * kFlattened * 4 <= 1.5 MB fits the LDG
    // immediate-offset field.
    const float* weight_base = args.fn_weight + group * kGroup * kFlattened;
    float local_square = 0.0f;
    float local_dot[kGroup];
#pragma unroll
    for (int slot = 0; slot < kGroup; ++slot) {
      local_dot[slot] = 0.0f;
    }
#pragma unroll kUnroll
    for (int step = 0; step < kHcPerThread; ++step) {
      const int column = lane + step * kHcThreads;
      const float value = __bfloat162float(staged_streams[column]);
      local_square = fmaf(value, value, local_square);
#pragma unroll
      for (int slot = 0; slot < kGroup; ++slot) {
        local_dot[slot] = fmaf(
            value, weight_base[slot * kFlattened + column], local_dot[slot]);
      }
    }
    // reduction_scratch is disjoint from staged_streams and is written before
    // it is read, so the item-trailing barrier is the only one needed here.
    reduction_scratch[lane] = local_square;
#pragma unroll
    for (int slot = 0; slot < kGroup; ++slot) {
      reduction_scratch[(slot + 1) * kHcThreads + lane] = local_dot[slot];
    }
    __syncthreads();
    // One barrier pass folds all kGroup + 1 independent halving trees; each
    // tree sees the same operands in the same order as the per-mix trees it
    // replaces.
#pragma unroll
    for (int delta = kHcThreads / 2; delta > 0; delta /= 2) {
      if (lane < delta) {
#pragma unroll
        for (int slot = 0; slot < kGroup + 1; ++slot) {
          reduction_scratch[slot * kHcThreads + lane] +=
              reduction_scratch[slot * kHcThreads + lane + delta];
        }
      }
      __syncthreads();
    }
    if (lane < kGroup) {
      const float inverse_rms =
          rsqrtf(reduction_scratch[0] / kFlattened + kNormEpsilon);
      args.mixes[row * kMixes + group * kGroup + lane] =
          reduction_scratch[(lane + 1) * kHcThreads] * inverse_rms;
    }
    __syncthreads();
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// Split-K TCGen05 candidate for q_a / main_kv / draft_kv: the dspark_shared
// FP8 pipeline (via its iter-5 wo_b clone in dspark_proj_phase.cuh),
// re-pointed at a 4096-K split-K = 4 operand set. One item is still one
// (output_tile, split) pair, so the body is a DROP-IN for the existing
// phase tilings (32 / 16 / 16 items). An 8-deep cp.async ring stages
// 128x32 E4M3 weight slots; warp 4 issues one K=32 UMMA per slot with four
// consecutive MMAs chained into one of four TMEM slots (scaleC = 0,1,1,1) =
// one tensor-core inner product per 128-element quantization block; warps
// 0-3 drain each block through the frozen outer FP32 fmaf scale chain and
// store raw FP32 split partials (no BF16 rounding). kRows = 1 (main_kv)
// leaves B columns 1..7 zero. Numerics: K=32-tree order inside a block vs
// the serial 128-FMA chain -> low-bit FP32 reorder noise only (the wo_b
// precedent measured rel_mean 1.3e-7); the bench reports the drift.
// kBatchElements > 1 tiles the item space OVER BATCH: one item is one
// (batch element, output tile, split) triple, element-major. Every workspace
// region is (batch, block, ...) contiguous, so element e's rows begin
// e * kRows further on. kBatchElements == 1 folds every added term away and
// compiles the frozen body verbatim.
template <int kWidth, int kRows, int kBatchElements = 1>
__device__ inline void execute_splitk_tcgen(
    const cutlass::float_e4m3_t* input,
    const uint8_t* input_scales,
    const cutlass::float_e4m3_t* weight,
    const uint8_t* weight_scales,
    float* partials,
    uint32_t begin,
    uint32_t end) {
  using namespace cute;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kStages = dspark_shared::kStages;                     // 8
  constexpr int kAccumulatorSlots = dspark_shared::kAccumulatorSlots; // 4
  constexpr int kSplitK = kHidden / kSplits;           // 1024 columns
  constexpr int kMmaBlocks = kSplitK / 32;             // 32 blocks per item
  constexpr int kGroupBlocks = 4;
  constexpr int kOutputTiles = kWidth / kOutputTile;
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<dspark_shared::TcgenSmemLayoutB>);
  static_assert(kRows >= 1 && kRows <= 8);
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
      const int split = item % kSplits;
      int output_tile = item / kSplits;
      int element = 0;
      if constexpr (kBatchElements > 1) {
        element = output_tile / kOutputTiles;
        output_tile -= element * kOutputTiles;
      }
      const cutlass::float_e4m3_t* element_input =
          input + static_cast<int64_t>(element) * kRows * kHidden;
      const int block = b % kMmaBlocks;
      const int column_base = split * kSplitK + block * 32;
      cutlass::float_e4m3_t* stage_a = storage.a[s].begin();
      for (int copy_index = lane; copy_index < kOutputTile * 2;
           copy_index += kStagingThreads) {
        const int output = copy_index >> 1;
        const int chunk = copy_index & 1;
        const cutlass::float_e4m3_t* source = weight
            + (output_tile * kOutputTile + output) * kHidden
            + column_base + chunk * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(stage_a) + output * 16 + chunk * 2048,
            source);
      }
      if (lane < 2 * kRows) {
        const int row = lane >> 1;
        const int chunk = lane & 1;
        const cutlass::float_e4m3_t* source =
            element_input + row * kHidden + column_base + chunk * 16;
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
    float result[kRows];
    constexpr int kGroupsPerItem = kMmaBlocks / kGroupBlocks;  // 8
    const int total_groups = total_blocks / kGroupBlocks;
    for (int group = 0; group < total_groups; ++group) {
      const int slot = group % kAccumulatorSlots;
      const int parity = group % 2;
      const int block = group % kGroupsPerItem;
      if (block == 0) {
        const int item =
            static_cast<int>(begin) + group / kGroupsPerItem;
        split = item % kSplits;
        output_tile = item / kSplits;
        if constexpr (kBatchElements > 1) {
          element = output_tile / kOutputTiles;
          output_tile -= element * kOutputTiles;
        }
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
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
      const int k_block = split * kBlocksPerSplit + block;
      const float weight_scale = decode_e8m0(
          weight_scales[output_tile * kBlocks + k_block]);
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        const float input_scale = decode_e8m0(
            input_scales[(element * kRows + row) * kBlocks + k_block]);
        result[row] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row]);
      }
      if (block == kGroupsPerItem - 1) {
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          partials[
              ((element * kRows + row) * kWidth + output_tile * kOutputTile
               + static_cast<int>(threadIdx.x)) * kSplits + split] =
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

// R4 band 4: batch 1 keeps the frozen body; batch > 1 N-widens onto the
// batch-invariant 32-item (output tile, split) tiling.
__device__ inline void execute_qa_tcgen(const QaArgs& args) {
#if DSPARK_V4_BATCH == 1
  execute_splitk_tcgen<kQaRank, kDraftBlock, dspark_batch::kBatch>(
      args.input_quantized,
      args.input_scales,
      args.wqa,
      args.wqa_scales,
      args.qa_partials,
      args.begin,
      args.end);
#else
  dspark_shared::execute_fp8_tcgen_wide<
      kHidden, kSplits, dspark_batch::kWideRows, dspark_batch::kRows,
      kQaRank, false>(
      args.input_quantized,
      args.input_scales,
      args.wqa,
      args.wqa_scales,
      args.qa_partials,
      nullptr,
      args.begin,
      args.end);
#endif
}

// The main-KV projection has ONE row per batch element and a batch-invariant
// item count, so the batch rides the UMMA N mode inside the frozen 8 columns.
__device__ inline void execute_main_kv_tcgen(const MainKvArgs& args) {
  execute_splitk_tcgen<kKvWidth, dspark_batch::kBatch>(
      args.quantized,
      args.activation_scales,
      args.weight,
      args.weight_scales,
      args.partials,
      args.begin,
      args.end);
}

// R4 band 5: same split as q_a, on the batch-invariant 16-item tiling.
__device__ inline void execute_draft_kv_tcgen(const DraftKvArgs& args) {
#if DSPARK_V4_BATCH == 1
  execute_splitk_tcgen<kKvWidth, kDraftBlock, dspark_batch::kBatch>(
      args.input_quantized,
      args.input_scales,
      args.wkv,
      args.wkv_scales,
      args.draft_kv_partials,
      args.begin,
      args.end);
#else
  dspark_shared::execute_fp8_tcgen_wide<
      kHidden, kSplits, dspark_batch::kWideRows, dspark_batch::kRows,
      kKvWidth, false>(
      args.input_quantized,
      args.input_scales,
      args.wkv,
      args.wkv_scales,
      args.draft_kv_partials,
      nullptr,
      args.begin,
      args.end);
#endif
}

#endif  // __CUDA_ARCH__ >= 1000

}  // namespace dspark_small
