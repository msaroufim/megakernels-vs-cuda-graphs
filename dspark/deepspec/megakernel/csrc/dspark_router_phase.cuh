// Router score band (`layer_N.router_sqrtsoftplus_top6`) for the DeepSeek-V4
// DSpark megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// The band computes, for every (expert, row) pair of one draft layer,
//   score = sqrt(softplus(dot(gate[expert, :], input[row, :])))
// over kRouterHidden = 4,096 BF16 columns, 256 experts and 5 block rows:
// 1,280 independent dot products per batch element per layer.
//
// TWO SHAPES LIVE HERE.
//
//   execute_router_scores_split<..., kSplit = 1, kRows = 1>
//     is the SHIPPED #101 body: one (expert, row) pair per thread, one
//     strictly ascending 4,096-step FMA chain per thread. At the shipped
//     tiling (kChainsPerItem = 40, i.e. 8 experts x 5 rows per work item)
//     it reproduces the pre-R12 body's thread->(expert, row) map and its
//     FMA order exactly, so it is BITWISE the frozen contract body and is
//     the numerical oracle for everything else in this file.
//
//   kSplit > 1 K-SPLITS one chain across kSplit threads and finishes with a
//     warp-shuffle tree. THIS REASSOCIATES THE FP32 SUM and is therefore a
//     RELAXED-BUILD-ONLY body, admitted through the same acceptance gate
//     that admitted the other relaxed stage-1..3 bodies. The contract build
//     never instantiates kSplit > 1.
//
// WHY K-SPLIT. The chain is 4,096 DEPENDENT FMAs, so no item-count retile
// can shorten it -- that much a previous leg established. What it missed is
// that only 40 of the CTA's 256 threads own a chain, and each thread has at
// most kUnroll * 2 loads in flight, so the band runs at ~40 x 8 outstanding
// requests per CTA on 32 of 148 CTAs: it is memory-level-parallelism
// starvation, not arithmetic latency (4,096 dependent FMAs is ~9 us at
// clock; the band measures ~55 us per layer). Splitting the chain raises
// both the live-thread count and the CTA count (fewer chains per item means
// more items) and leaves each thread a 4096/kSplit walk plus a log2(kSplit)
// shuffle tree.
//
// LANE PARTITION. Lane l of a chain owns the 8-element packed groups
// l, l + kSplit, l + 2*kSplit, ... so the kSplit lanes of one chain touch
// CONSECUTIVE 16 B groups at every step: one chain's step is a single
// contiguous 16*kSplit B window instead of kSplit scattered ones. Every
// loop bound is a compile-time constant (R10: a `blockDim.x` stride is a
// runtime special register and blocks unrolling).
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <math.h>

namespace dspark_router {

#if defined(DSPARK_V4_RELAXED_DAG) && DSPARK_V4_BATCH == 1 \
    && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 \
    && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
#define DSPARK_ROUTER_HAS_BF16_SOURCE_FMA 1
// Blackwell accepts the packed BF16 operands directly while retaining one
// FP32 rounding per FMA. Keep the original operand and accumulation order.
__device__ __forceinline__ float router_mixed_bf16_fma(
    __nv_bfloat16 a, __nv_bfloat16 b, float accumulator) {
  const unsigned short a_bits = __bfloat16_as_ushort(a);
  const unsigned short b_bits = __bfloat16_as_ushort(b);
  asm("fma.rn.f32.bf16 %0, %1, %2, %0;"
      : "+f"(accumulator) : "h"(a_bits), "h"(b_bits));
  return accumulator;
}
#endif

// The megakernel's fixed launch width. Every body here indexes threads with
// compile-time constants; the wrapper static_asserts this against
// generated::kThreads.
constexpr int kRouterThreads = 256;
constexpr int kRouterHidden = 4096;
constexpr int kRouterExperts = 256;
constexpr int kRouterBlock = 5;
// BF16 elements in one 16-byte aligned load (#84 pattern).
constexpr int kRouterPacked = 8;
constexpr int kRouterGroups = kRouterHidden / kRouterPacked;

struct RouterScoreArgs {
  const __nv_bfloat16* gate;
  const __nv_bfloat16* input;
  float* scores;
  uint32_t begin;
  uint32_t end;
};

// Chains one work item owns, given kChainsPerItem chains of kRows rows each.
// One batch element carries kRouterExperts * (kRouterBlock / kRows) chains.
template <int kRows>
struct RouterChainSpace {
  static constexpr int kRowGroups = kRouterBlock / kRows;
  static constexpr int kChainsPerElement = kRouterExperts * kRowGroups;
};

// kChainsPerItem : chains (each = one expert x kRows rows) per work item.
// kSplit         : threads cooperating on one chain (1 = the frozen shape).
// kRows          : rows one chain carries; 5 shares each gate load across the
//                  whole block (10 -> 6 loads per element per expert) at the
//                  cost of kRows live accumulators.
// kUnroll        : packed groups issued per lane per step.
template <int kChainsPerItem, int kSplit, int kRows, int kUnroll>
__device__ inline void execute_router_scores_split(
    const RouterScoreArgs& args) {
  using Space = RouterChainSpace<kRows>;
  constexpr int kThreadsUsed = kChainsPerItem * kSplit;
  constexpr int kSteps = kRouterGroups / (kSplit * kUnroll);
  static_assert(kRows == 1 || kRows == kRouterBlock, "kRows is 1 or the block");
  static_assert(kThreadsUsed <= kRouterThreads, "one CTA per work item");
  static_assert(kSplit >= 1 && kSplit <= 32, "the tree is intra-warp");
  static_assert((kSplit & (kSplit - 1)) == 0, "kSplit is a power of two");
  // A shuffle tree needs every lane of its warp resident: with a whole
  // number of warps busy each warp is either fully active or fully idle, so
  // the full mask is exact. kSplit == 1 has no tree and no such constraint.
  static_assert(kSplit == 1 || kThreadsUsed % 32 == 0, "whole warps");
  static_assert(kRouterGroups % (kSplit * kUnroll) == 0, "even lane walk");
  static_assert(
      Space::kChainsPerElement % kChainsPerItem == 0,
      "a work item may not straddle two batch elements");

  const int thread = static_cast<int>(threadIdx.x);
  if (thread >= kThreadsUsed) {
    return;
  }
  const int chain = thread / kSplit;
  const int lane = thread % kSplit;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int index = static_cast<int>(item) * kChainsPerItem + chain;
    const int element = index / Space::kChainsPerElement;
    const int local = index - element * Space::kChainsPerElement;
    const int expert = local / Space::kRowGroups;
    const int row_group = local - expert * Space::kRowGroups;
    const int row_base = element * kRouterBlock + row_group * kRows;

    const __nv_bfloat16* gate_row = args.gate + expert * kRouterHidden;
    const __nv_bfloat16* input_rows = args.input + row_base * kRouterHidden;

    float logit[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      logit[r] = 0.0f;
    }
    for (int step = 0; step < kSteps; ++step) {
      uint4 packed_gate[kUnroll];
      uint4 packed_input[kUnroll][kRows];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int column =
            ((step * kUnroll + u) * kSplit + lane) * kRouterPacked;
        packed_gate[u] =
            *reinterpret_cast<const uint4*>(gate_row + column);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          packed_input[u][r] = *reinterpret_cast<const uint4*>(
              input_rows + r * kRouterHidden + column);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const __nv_bfloat16* gate_values =
            reinterpret_cast<const __nv_bfloat16*>(&packed_gate[u]);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          const __nv_bfloat16* input_values =
              reinterpret_cast<const __nv_bfloat16*>(&packed_input[u][r]);
#pragma unroll
          for (int offset = 0; offset < kRouterPacked; ++offset) {
#ifdef DSPARK_ROUTER_HAS_BF16_SOURCE_FMA
            if constexpr (kChainsPerItem == 8 && kSplit == 32
                          && kRows == 1 && kUnroll == 4) {
              logit[r] = router_mixed_bf16_fma(
                  input_values[offset], gate_values[offset], logit[r]);
            } else
#endif
            {
              logit[r] = fmaf(
                  __bfloat162float(input_values[offset]),
                  __bfloat162float(gate_values[offset]),
                  logit[r]);
            }
          }
        }
      }
    }
    // Intra-chain tree. kSplit == 1 compiles this away entirely, which is
    // what makes that instantiation bitwise to the frozen body.
#pragma unroll
    for (int delta = kSplit / 2; delta > 0; delta /= 2) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        logit[r] += __shfl_down_sync(0xffffffffu, logit[r], delta, kSplit);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const float softplus =
            logit[r] > 20.0f ? logit[r] : log1pf(expf(logit[r]));
        args.scores[(row_base + r) * kRouterExperts + expert] =
            sqrtf(softplus);
      }
    }
  }
}

}  // namespace dspark_router

#ifdef DSPARK_ROUTER_HAS_BF16_SOURCE_FMA
#undef DSPARK_ROUTER_HAS_BF16_SOURCE_FMA
#endif
