// TP2 band-ownership ablation: the per-rank workload of a two-GPU math-level
// tensor-parallel drafter, measured on ONE GPU with no cross-rank traffic.
//
// WHY THIS EXISTS.  The R2 math-TP campaign was sized on a per-wave model and
// the model is what falsified it: at batch 1 the split bands are one wave
// deep, so halving the items removes nothing.  At batch 8 several bands are
// genuinely multi-wave, but "genuinely" is a claim about a dynamically
// claimed 148-worker megakernel, and the only honest way to settle it is to
// run one rank's half of the work and time it.  This header makes each split
// band skip the work-items the OTHER rank would own.  The output is
// numerically wrong by construction -- that is the point; the measurement is
// the per-rank latency floor a real TP2 build could reach, before comm.
//
// The two-rank latency is then (ablated wall) + (measured exchange cost),
// and the exchange cost is already known from the qualified layout-v4
// all-reduce primitive (10.496 us median publish->consume, PLAN.md S1).
// A band that does not move here cannot be made to pay by building the
// harness, so this is the gate that decides what is worth building.
//
// Inert unless DSPARK_V4_TP2_ABLATE is defined; the shipped builds never see
// it and stay byte-identical.
#pragma once

#if defined(DSPARK_V4_TP2_ABLATE)

#include <cstdint>

namespace dspark_tp2_ablate {

// Band selector bits. Runtime-settable so ONE build measures every subset:
// a recompile per configuration would cost more than the GPU time.
enum BandBits : uint32_t {
  kBandRouted = 1u << 0,     // fp4_routed_w13 / routed_swiglu / fp4_routed_w2
  kBandAttention = 1u << 1,  // q_b / q_norm_rope / sparse_attention / wo_a
  kBandVocab = 1u << 2,      // head.lm_row_0 / tail_*.markov_w2
};

// Rank id (0 or 1) and the active band mask. Both are plain device globals
// written once from the host before a proposal; no proposal-time thread
// writes them, so no atomics are needed.
__device__ uint32_t g_rank = 0;
__device__ uint32_t g_bands = 0;

__device__ __forceinline__ bool band_active(uint32_t bit) {
  return (g_bands & bit) != 0;
}

// EXPERT-PARALLEL ownership: rank r owns experts 128r..128r+127.
// Splitting on the EXPERT (not the route row) is what keeps the
// chunk_leader/chunk_members grouping intact -- every route row that shares
// an expert lands on the same rank, so a member is never separated from its
// leader.  A route-row split would silently produce zeros (EXPERT-BAND-MAP
// hazard 1).
__device__ __forceinline__ bool owns_expert(int expert) {
  if (!band_active(kBandRouted)) {
    return true;
  }
  return (expert < 128) == (g_rank == 0);
}

// HEAD-PARALLEL ownership: rank r owns heads 32r..32r+31 of the 64.
__device__ __forceinline__ bool owns_head(int head) {
  if (!band_active(kBandAttention)) {
    return true;
  }
  return (head < 32) == (g_rank == 0);
}

// VOCAB ownership on 128-column tiles, rebalanced 505/505 (the shipped
// static-TP2 tail is 404/606; at batch 8 both ranks carry equal load
// elsewhere, so the skew has nothing left to compensate for).
constexpr int kVocabTiles = 1010;
constexpr int kVocabSplit = kVocabTiles / 2;  // 505

__device__ __forceinline__ bool owns_vocab_tile(int tile) {
  if (!band_active(kBandVocab)) {
    return true;
  }
  return (tile < kVocabSplit) == (g_rank == 0);
}

// Several split bands have an item axis on which ownership is a contiguous
// prefix/suffix -- q_b tiles (head = item / 4), wo_a (item = group * 8 +
// rank_tile, so groups 0..3 are heads 0..31), and the vocab tiles. For those
// the ablation clips the CLAIMED range instead of testing each item, which
// keeps the frozen body untouched.
__device__ __forceinline__ void clip_half(
    uint32_t total, uint32_t bit, uint32_t& begin, uint32_t& end) {
  if (!band_active(bit)) {
    return;
  }
  const uint32_t split = total / 2;
  const uint32_t lo = (g_rank == 0) ? 0u : split;
  const uint32_t hi = (g_rank == 0) ? split : total;
  begin = begin > lo ? begin : lo;
  end = end < hi ? end : hi;
  if (end < begin) {
    end = begin;
  }
}

}  // namespace dspark_tp2_ablate

#endif  // DSPARK_V4_TP2_ABLATE
