// Self-contained Markov-W2 tail phase bodies for the DeepSeek-V4 DSpark
// megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// The band: for draft step N over the FULL 129,280-entry vocabulary,
//   markov_logits[v]  = sum_{c=0..255} emb[N][c] * W2[v][c]   (FP32 ascending)
//   corrected[N][v]   = base_logits[N][v] + markov_logits[v]
//   argmax over v, published through the #102 packed uint64 atomicMax.
// M = 129,280, N = 1, K = 256 — a pure GEMV at 1.0 FLOP/byte. The relaxed DAG
// fuses the correction + argmax into this one 1,010-tile phase; a work item is
// one 128-output vocab tile, and the five steps are STRICTLY SERIAL
// (normalize_N gates gather_{N+1}), so per-step latency IS the critical path.
//
// execute_fused_reference is a VERBATIM extraction of the production
// execute_tail_markov_w2_fused body (pure code motion; stage./task. became
// args.) and is this lane's oracle.
#pragma once

#include "dspark_tmem.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "dspark_batch.h"

namespace dspark_markov {

constexpr int kVocab = 129280;
constexpr int kRank = 256;        // K extent (markov embedding rank)
constexpr int kOutputTile = 128;  // vocab rows per work item
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;

struct Args {
  // Weight streams. bf16 is the relaxed-slot copy (checkpoint-exact: the FP32
  // arena entry is itself a widened BF16 tensor); when it is null the FP32
  // stream is used and the UMMA body is not available.
  const float* markov_w2 = nullptr;
  const __nv_bfloat16* markov_w2_bf16 = nullptr;
  // Pre-offset to this step: markov_embeddings + step * kRank.
  const __nv_bfloat16* embedding = nullptr;
  // Pre-offset to this step: base_logits / corrected_logits + step * kVocab.
  const float* base_logits = nullptr;
  float* corrected_logits = nullptr;
  float* markov_logits = nullptr;
  unsigned long long* argmax_slot = nullptr;
  uint32_t begin = 0;
  uint32_t end = 0;
  // Batched serving (R3): the pointers above address batch element 0 and the
  // strides below reach the rest. The 66 MiB W2 stream is read ONCE per step
  // for the whole batch -- the rows ride the UMMA N mode, which the frozen
  // body already allocates 8 of and fills 1 of. Zero strides reproduce the
  // batch-1 body exactly.
  int embedding_stride = 0;         // markov_embeddings: block * kRank
  int64_t logits_stride = 0;        // base/corrected logits: block * kVocab
  int64_t markov_logits_stride = 0; // markov_logits_row: kVocab
  int argmax_slot_stride = 0;       // uint64 words between per-element slots
};

// Packed (value, lowest-index-tiebreak) encoding for the fused tail argmax
// (#102), copied verbatim from dspark_v4_kernel.cu (which keeps its own copy
// for the other tail bands). The float bits map monotonically onto uint32 and
// the complemented index occupies the low word, so a single uint64 atomicMax
// reproduces the two-stage (max value, then lowest index) reduction bit-exactly
// under ANY arrival order. LOAD-BEARING: the TP2 cross-rank merge decodes this
// exact encoding.
__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ unsigned long long pack_argmax(
    float value, int index) {
  return (static_cast<unsigned long long>(ordered_float_bits(value)) << 32)
      | static_cast<unsigned long long>(
          0xFFFFFFFFu - static_cast<uint32_t>(index));
}

// ORACLE (mode 0): verbatim extraction of execute_tail_markov_w2_fused.
// One thread per vocab row, 512-byte lane stride, kTile 128 under blockDim
// 256 so four of eight warps only reach barriers. Inactive lanes carry
// -FLT_MAX/kVocab sentinels through the block reduction.
__device__ inline void execute_fused_reference(
    const Args& args, float* shared) {
  constexpr int kTile = kOutputTile;
  int* shared_index = reinterpret_cast<int*>(shared + blockDim.x);
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int output = static_cast<int>(item) * kTile
        + static_cast<int>(threadIdx.x);
    const bool active = threadIdx.x < kTile && output < kVocab;
    float corrected = -FLT_MAX;
    if (active) {
      float result = 0.0f;
      if (args.markov_w2_bf16 != nullptr) {
        const int64_t weight_base = static_cast<int64_t>(output) * kRank;
        const uint4* weight_vectors = reinterpret_cast<const uint4*>(
            args.markov_w2_bf16 + weight_base);
#pragma unroll 4
        for (int column = 0; column < kRank; column += 8) {
          const uint4 packet = weight_vectors[column / 8];
          const __nv_bfloat16* weights =
              reinterpret_cast<const __nv_bfloat16*>(&packet);
#pragma unroll
          for (int offset = 0; offset < 8; ++offset) {
            result = fmaf(
                __bfloat162float(args.embedding[column + offset]),
                __bfloat162float(weights[offset]),
                result);
          }
        }
      } else {
        const int64_t weight_base = static_cast<int64_t>(output) * kRank;
        const float4* weight_vectors =
            reinterpret_cast<const float4*>(args.markov_w2 + weight_base);
#pragma unroll 8
        for (int column = 0; column < kRank; column += 4) {
          const float4 weight_vector = weight_vectors[column / 4];
          const float* weights =
              reinterpret_cast<const float*>(&weight_vector);
#pragma unroll
          for (int offset = 0; offset < 4; ++offset) {
            result = fmaf(
                __bfloat162float(args.embedding[column + offset]),
                weights[offset],
                result);
          }
        }
      }
      args.markov_logits[output] = result;
      corrected = args.base_logits[output] + result;
      args.corrected_logits[output] = corrected;
    }
    shared[threadIdx.x] = corrected;
    shared_index[threadIdx.x] = active ? output : kVocab;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        const float other = shared[threadIdx.x + delta];
        const int other_index = shared_index[threadIdx.x + delta];
        if (other > shared[threadIdx.x]
            || (other == shared[threadIdx.x]
                && other_index < shared_index[threadIdx.x])) {
          shared[threadIdx.x] = other;
          shared_index[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicMax(args.argmax_slot, pack_argmax(shared[0], shared_index[0]));
    }
    __syncthreads();
  }
}

// CANDIDATE (mode 1): thread-utilization fix, BITWISE by construction.
// The reference wastes four of eight warps every item because a work item is
// 128 vocab rows under a 256-thread CTA. This body keeps the work-item space,
// the item->vocab mapping and every logit's ascending 256-column FP32 chain
// EXACTLY as the reference has them; it just runs TWO consecutive items in
// flight — lanes 0-127 walk item i, lanes 128-255 walk item i+1 — and folds
// both tiles through ONE 256-wide packed argmax reduction into ONE atomicMax.
// The packed encoding is a total order, so reducing two tiles jointly and
// publishing once is value-identical to the reference's two separate
// atomicMax calls under any arrival order. A trailing odd item falls back to
// the reference's half-active shape.
__device__ inline void execute_fused_dual(const Args& args, float* shared) {
  unsigned long long* shared_packed =
      reinterpret_cast<unsigned long long*>(shared);
  const int half = static_cast<int>(threadIdx.x) / kOutputTile;  // 0 or 1
  const int lane_row = static_cast<int>(threadIdx.x) % kOutputTile;
  uint32_t item = args.begin;
  for (; item + 1 < args.end; item += 2) {
    const int output =
        static_cast<int>(item + static_cast<uint32_t>(half)) * kOutputTile
        + lane_row;
    unsigned long long packed = 0ULL;
    if (output < kVocab) {
      float result = 0.0f;
      if (args.markov_w2_bf16 != nullptr) {
        const int64_t weight_base = static_cast<int64_t>(output) * kRank;
        const uint4* weight_vectors = reinterpret_cast<const uint4*>(
            args.markov_w2_bf16 + weight_base);
#pragma unroll 4
        for (int column = 0; column < kRank; column += 8) {
          const uint4 packet = weight_vectors[column / 8];
          const __nv_bfloat16* weights =
              reinterpret_cast<const __nv_bfloat16*>(&packet);
#pragma unroll
          for (int offset = 0; offset < 8; ++offset) {
            result = fmaf(
                __bfloat162float(args.embedding[column + offset]),
                __bfloat162float(weights[offset]),
                result);
          }
        }
      } else {
        const int64_t weight_base = static_cast<int64_t>(output) * kRank;
        const float4* weight_vectors =
            reinterpret_cast<const float4*>(args.markov_w2 + weight_base);
#pragma unroll 8
        for (int column = 0; column < kRank; column += 4) {
          const float4 weight_vector = weight_vectors[column / 4];
          const float* weights =
              reinterpret_cast<const float*>(&weight_vector);
#pragma unroll
          for (int offset = 0; offset < 4; ++offset) {
            result = fmaf(
                __bfloat162float(args.embedding[column + offset]),
                weights[offset],
                result);
          }
        }
      }
      args.markov_logits[output] = result;
      const float corrected = args.base_logits[output] + result;
      args.corrected_logits[output] = corrected;
      packed = pack_argmax(corrected, output);
    }
    // Warp-local max on the packed key, then a 8-entry shared fold.
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      const unsigned long long other =
          __shfl_down_sync(0xFFFFFFFFu, packed, delta);
      packed = other > packed ? other : packed;
    }
    if ((threadIdx.x & 31) == 0) {
      shared_packed[threadIdx.x / 32] = packed;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long best = shared_packed[0];
      for (int warp = 1; warp < 8; ++warp) {
        best = shared_packed[warp] > best ? shared_packed[warp] : best;
      }
      atomicMax(args.argmax_slot, best);
    }
    __syncthreads();
  }
  if (item < args.end) {
    Args tail = args;
    tail.begin = item;
    tail.end = args.end;
    execute_fused_reference(tail, shared);
  }
}

// CANDIDATE (mode 2): TCGen05 BF16 UMMA — a K=256 clone of
// dspark_lm::execute_umma_bf16 (lm_phase.cuh:430-617).
//
// The MMA itself buys nothing here (66 MFLOP/step ~ 0.03 us of tensor core).
// What the LM port actually bought, and what this copies, is the STAGING
// PIPELINE: cp.async 16-byte packets at perfect coalescence through a 4-deep
// ring, with the crews split so weight movement overlaps the reduction —
// exactly what the reference's 512-byte-lane-stride per-thread stream lacks
// (it measures 0.45-0.62 TB/s, the LM SCALAR ceiling this project already beat
// 5.85x). Only K changes vs the LM body: 4096 -> 256, so kUmmaSlotsPerItem
// 64 -> 4 and the padded-B buffer 256 chunks (64 KB) -> 16 chunks (4 KB). The
// A ring is unchanged at 4 x 16 KB, so SMEM lands ~78 KB of the 139,520 B
// budget.
//
// NUMERICS: not bitwise. Each output's 256-column chain becomes 16 chained
// K=16 tensor-core reduction trees — precisely the LM band's accepted
// reassociation class (which measured max 4.196e-5 at K=4096); at K=256 the
// reassociation is 16x shallower.
//
// The fused correct+argmax epilogue rides the drain crew, and the packed
// atomicMax encoding is preserved EXACTLY (pack_argmax above), because the
// TP2 cross-rank merge decodes it.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

using MarkovUmmaAtom = cute::SM100_MMA_F16BF16_SS<
    cute::bfloat16_t, cute::bfloat16_t, float, kOutputTile, 8,
    cute::UMMA::Major::K, cute::UMMA::Major::K>;
using MarkovUmmaTiledMma = decltype(cute::make_tiled_mma(MarkovUmmaAtom{}));
using MarkovUmmaShapeA = decltype(cute::partition_shape_A(
    MarkovUmmaTiledMma{},
    cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<16>{})));
using MarkovUmmaShapeB = decltype(cute::partition_shape_B(
    MarkovUmmaTiledMma{},
    cute::make_shape(cute::Int<8>{}, cute::Int<16>{})));
using MarkovUmmaSmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, MarkovUmmaShapeA{}));
using MarkovUmmaSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, MarkovUmmaShapeB{}));

constexpr int kUmmaK = 16;   // K per tcgen05.mma
constexpr int kUmmaChunks = kRank / kUmmaK;  // 16 B-descriptor chunks
// TWO live TMEM accumulators (8 columns each of the 32 allocated). At K=4096
// the LM body amortizes its single accumulator over 64 slots per item, so a
// drain-before-next-item serialization is invisible; at K=256 an item is only
// 2-4 slots, so a single accumulator would put the whole per-item drain
// latency between consecutive items. Alternating accumulators lets item i+1's
// MMAs run while item i drains (the campaign's run-4 finding, applied where it
// actually bites).
constexpr int kUmmaAccumulators = 2;

// kSubTiles = K16 sub-tiles per ring slot (the slot A tile is
// 128 x 16*kSubTiles BF16 = 4 KB * kSubTiles); kStages = ring depth. The two
// instantiations below trade per-row contiguity against ring depth at a fixed
// SMEM budget.
template <int kSubTiles, int kStages>
struct MarkovUmmaSharedStorage {
  static constexpr int kSlotElements = kOutputTile * kUmmaK * kSubTiles;
  alignas(128) __nv_bfloat16 b_all[kUmmaChunks * 8 * kUmmaK];   // 4 KB
  alignas(128) __nv_bfloat16 a_ring[kStages][kSlotElements];
  alignas(16) float partial[2][kOutputTile * 8];                // 8 KB
  alignas(16) unsigned long long argmax_partial[4];
  alignas(16) cute::uint64_t stage_full[kStages];
  alignas(16) cute::uint64_t slot_free[kStages];
  alignas(16) cute::uint64_t accumulator_full[kUmmaAccumulators];
  alignas(16) cute::uint64_t accumulator_free[kUmmaAccumulators];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <int kSubTiles, int kStages, int kRows = 1, bool kBulkStaging = false>
__device__ inline void execute_umma_impl(const Args& args) {
  using namespace cute;
  using Storage = MarkovUmmaSharedStorage<kSubTiles, kStages>;
  static_assert(kRows >= 1 && kRows <= 8,
                "batch rows ride the frozen N=8 UMMA columns");
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kTmemColumns = 32;
#if defined(DSPARK_MARKOV_DIRECT_DRAIN) && DSPARK_MARKOV_DIRECT_DRAIN
  // Only the current batch-one bulk128 Markov specialization uses col0.
  constexpr bool kDirectLiveColumn =
      kSubTiles == 8 && kStages == 3 && kRows == 1 && kBulkStaging;
#endif
  constexpr int kSlotK = kUmmaK * kSubTiles;
  constexpr int kSlotsPerItem = kRank / kSlotK;
  // log2(kSubTiles), for the shift-only staging lane->packet decomposition
  // below (a signed / or % by a constant costs a negative-operand fixup).
  constexpr int kSubTileShift =
      kSubTiles == 1 ? 0 : (kSubTiles == 2 ? 1 : (kSubTiles == 4 ? 2 : (kSubTiles == 8 ? 3 : 4)));
  static_assert((1 << kSubTileShift) == kSubTiles && kSubTiles <= 16,
                "kSubTiles must be a power of two no larger than 16");
  static_assert(kRank % kSlotK == 0, "K must tile the ring slot exactly");
  static_assert(
      sizeof(Storage) <= kSharedBytesBudget,
      "markov UMMA staging ring + padded B must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<Storage*>(dynamic_shared_memory);
  MarkovUmmaTiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<kOutputTile>{}, Int<8>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = static_cast<int>(threadIdx.x) / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(
        storage.stage_full, kBulkStaging ? 1 : kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(storage.slot_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaAccumulators>(
        storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaAccumulators>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Padded B, staged once: row 0 is the step's 256-element markov embedding,
  // rows 1-7 zero. One 16-byte packet covers 8 K-contiguous elements of one
  // row; chunk c holds K columns [c*16, c*16+16) at byte base c*256
  // (half-atoms at +0/+128).
  for (int index = static_cast<int>(threadIdx.x); index < kUmmaChunks * 16;
       index += static_cast<int>(blockDim.x)) {
    const int chunk = index / 16;
    const int row = (index % 16) / 2;
    const int half = index & 1;
    uint4 packet = make_uint4(0u, 0u, 0u, 0u);
    if (row < kRows) {
      packet = *reinterpret_cast<const uint4*>(
          args.embedding + row * args.embedding_stride
          + chunk * kUmmaK + half * 8);
    }
    *reinterpret_cast<uint4*>(
        reinterpret_cast<unsigned char*>(storage.b_all)
        + chunk * 256 + half * 128 + row * 16) = packet;
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

#if (DSPARK_DESCRIPTOR_ARITHMETIC & 8)
  const uint64_t a_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(
      make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[0])),
                  layout<0>(MarkovUmmaSmemLayoutA{}))).desc_;
  const auto a_descriptor_at = [=](int stage) {
    return a_descriptor_base + static_cast<uint64_t>(stage) * Storage::kSlotElements / 8;
  };
#else
  uint64_t a_descriptor[kStages];
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    a_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[s])),
            layout<0>(MarkovUmmaSmemLayoutA{}))).desc_;
  }
  const auto a_descriptor_at = [&](int stage) { return a_descriptor[stage]; };
#endif
  const uint64_t b_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(
      make_tensor(
          make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_all)),
          layout<0>(MarkovUmmaSmemLayoutB{}))).desc_;
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_slots = item_count * kSlotsPerItem;

  if (kBulkStaging && warp >= 5) {
    if (threadIdx.x == 160) {
      for (int c = 0; c < total_slots; ++c) {
        const int s = c % kStages;
        if (c >= kStages) {
          wait_barrier(storage.slot_free[s], (c / kStages - 1) & 1);
        }
        const int item = static_cast<int>(args.begin) + c / kSlotsPerItem;
        constexpr int bytes = Storage::kSlotElements * sizeof(__nv_bfloat16);
        const auto* source = args.markov_w2_bf16
            + (static_cast<int64_t>(item) * kSlotsPerItem + c % kSlotsPerItem)
                * Storage::kSlotElements;
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &storage.stage_full[s], bytes);
        cute::SM90_BULK_COPY_G2S::copy(
            source, &storage.stage_full[s], storage.a_ring[s], bytes);
      }
    }
  } else if (warp >= 5) {
    // Staging crew: 128 * kSubTiles * 2 cp.async 16-byte packets per slot.
    // Lane->packet map is bank-conflict aware (see execute_umma_bf16 in
    // dspark_lm_phase.cuh): giving eight consecutive lanes ONE vocab row's
    // eight packets puts the whole quarter-warp at SMEM offsets 2048 apart,
    // i.e. in the same four banks -- an 8-way store conflict. Interleaving
    // two bits of m below j spreads the quarter-warp over four rows while
    // keeping each row's two 16-byte halves adjacent, so each lane pair still
    // covers one full 32-byte global sector. Bitwise by construction.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kStages;
      if (c >= kStages) {
        wait_barrier(storage.slot_free[s], (c / kStages - 1) & 1);
      }
      const int item = static_cast<int>(args.begin) + c / kSlotsPerItem;
      const int64_t weight_base =
          static_cast<int64_t>(item) * kOutputTile * kRank;
      const int k_base = (c % kSlotsPerItem) * kSlotK;
      unsigned char* slot =
          reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      static_assert(kOutputTile % 4 == 0, "quarter-warp spreads 4 rows");
      for (int index = lane; index < kOutputTile * kSubTiles * 2;
           index += kStagingThreads) {
        const unsigned packet = static_cast<unsigned>(index);
        const int half = static_cast<int>(packet & 1u);
        const int j = static_cast<int>(
            (packet >> 3) & static_cast<unsigned>(kSubTiles - 1));
        const int m = static_cast<int>(
            ((packet >> (3 + kSubTileShift)) << 2) | ((packet >> 1) & 3u));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            slot + j * 4096 + half * 2048 + m * 16,
            args.markov_w2_bf16 + weight_base
                + static_cast<int64_t>(m) * kRank
                + k_base + j * kUmmaK + half * 8);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    // MMA crew: kSubTiles chained K=16 UMMAs per slot into the item's TMEM
    // accumulator (items alternate between the two); scaleC=0 opens each
    // item, 1 chains, so every vocab row's 256-column reduction is 16 chained
    // K=16 trees in ascending K order.
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kStages;
      const int slot_in_item = c % kSlotsPerItem;
      const int item_local = c / kSlotsPerItem;
      const int acc = item_local % kUmmaAccumulators;
      const int round = item_local / kUmmaAccumulators;
      wait_barrier(storage.stage_full[s], (c / kStages) & 1);
      if (slot_in_item == 0 && round >= 1) {
        wait_barrier(storage.accumulator_free[acc], (round - 1) & 1);
      }
      const uint32_t tmem_acc = storage.tmem_base_ptr + acc * 8;
      CUTE_UNROLL
      for (int j = 0; j < kSubTiles; ++j) {
        MarkovUmmaAtom::fma(
            a_descriptor_at(s) + static_cast<uint64_t>(j) * 256,   // 4 KB >> 4
            b_descriptor_base + static_cast<uint64_t>(
                slot_in_item * kSubTiles + j) * 16,             // 256 B >> 4
            tmem_acc,
            (slot_in_item == 0 && j == 0) ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kSlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[acc]);
      }
    }
  } else {
    // Drain crew (warps 0-3): one TMEM drain per item, then the FUSED
    // epilogue -- markov_logits store, base_logits correction,
    // corrected_logits store, and the packed argmax. Each drain lane keeps a
    // RUNNING best across the whole claim and the CTA publishes ONE atomicMax
    // at the end: pack_argmax is a total order and max is associative and
    // commutative, so the published value is identical to the reference's
    // per-tile publications under any arrival order, at one global atomic per
    // invocation instead of one per 128-row tile. The per-item barrier count
    // is then exactly the LM body's (one NamedBarrier).
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto view_a = tmem_accumulator;
    view_a.data() = storage.tmem_base_ptr;
    auto view_b = tmem_accumulator;
    view_b.data() = storage.tmem_base_ptr + 8;
    auto thread_tmem_a = thread_copy.partition_S(view_a);
    auto thread_tmem_b = thread_copy.partition_S(view_b);
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

    unsigned long long best[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      best[r] = 0ULL;
    }
    for (int i = 0; i < item_count; ++i) {
      const int acc = i % kUmmaAccumulators;
      const int round = i / kUmmaAccumulators;
      const int parity = i & 1;
      wait_barrier(storage.accumulator_full[acc], round & 1);
#if defined(DSPARK_MARKOV_DIRECT_DRAIN) && DSPARK_MARKOV_DIRECT_DRAIN
      uint32_t direct_result_bits = 0;
      if constexpr (kDirectLiveColumn) {
        // Same column-zero address as the original CuTe copy: upper16 bits
        // select this warp's32 data paths; the load supplies lane internally.
        const uint32_t direct_address = storage.tmem_base_ptr
            + static_cast<uint32_t>(acc) * 8u
            + (static_cast<uint32_t>(warp * 32) << 16);
        SM100_TMEM_LOAD_32dp32b1x::copy(direct_address, direct_result_bits);
        // Complete each lane's read before its contribution to the128-arrival
        // free barrier. The producer may then reuse TMEM; result bits are local.
        cutlass::arch::fence_view_async_tmem_load();
        cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[acc]);
      } else {
#endif
      if (acc == 0) {
        copy(tmem_to_register, thread_tmem_a, register_accumulator);
      } else {
        copy(tmem_to_register, thread_tmem_b, register_accumulator);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[acc]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
#if defined(DSPARK_MARKOV_DIRECT_DRAIN) && DSPARK_MARKOV_DIRECT_DRAIN
      }
#endif
      const int output =
          (static_cast<int>(args.begin) + i) * kOutputTile
          + static_cast<int>(threadIdx.x);
      if (output < kVocab) {
        // Columns 0..kRows-1 of the N=8 accumulator carry the live activation
        // rows (one per batch element); the rest are the zero padding.
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#if defined(DSPARK_MARKOV_DIRECT_DRAIN) && DSPARK_MARKOV_DIRECT_DRAIN
          const float result = kDirectLiveColumn
              ? __uint_as_float(direct_result_bits)
              : storage.partial[parity]
                    [r * kOutputTile + static_cast<int>(threadIdx.x)];
#else
          const float result = storage.partial[parity]
              [r * kOutputTile + static_cast<int>(threadIdx.x)];
#endif
          // The rank-1 TP2 shard consumes only the packed argmax, so it
          // passes null logit buffers; the pointers are grid-uniform.
          if (args.markov_logits != nullptr) {
            args.markov_logits[r * args.markov_logits_stride + output] =
                result;
          }
          const float corrected =
              args.base_logits[r * args.logits_stride + output] + result;
          if (args.corrected_logits != nullptr) {
            args.corrected_logits[r * args.logits_stride + output] = corrected;
          }
          const unsigned long long packed = pack_argmax(corrected, output);
          best[r] = packed > best[r] ? packed : best[r];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        const unsigned long long other =
            __shfl_down_sync(0xFFFFFFFFu, best[r], delta);
        best[r] = other > best[r] ? other : best[r];
      }
    }
    for (int r = 0; r < kRows; ++r) {
      if (r > 0) {
        // Every warp must be done folding row r-1 before argmax_partial is
        // reused. Unreachable (and uncompiled) at batch 1.
        cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      }
      if ((threadIdx.x & 31) == 0) {
        storage.argmax_partial[threadIdx.x / 32] = best[r];
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      if (threadIdx.x == 0) {
        unsigned long long folded = storage.argmax_partial[0];
        for (int w = 1; w < 4; ++w) {
          folded = storage.argmax_partial[w] > folded
              ? storage.argmax_partial[w] : folded;
        }
        atomicMax(args.argmax_slot + r * args.argmax_slot_stride, folded);
      }
    }
  }
  // Close each crew's final acknowledgements before the existing crew join.
  // The stage producer and MMA producer may finish this while drain epilogue runs.
  if (threadIdx.x == 160) {
    CUTE_UNROLL
    for (int slot = 0; slot < kStages; ++slot) {
      if (slot < total_slots) {
        const int last = slot + ((total_slots - 1 - slot) / kStages) * kStages;
        wait_barrier(storage.slot_free[slot], (last / kStages) & 1);
      }
    }
  }
  if (threadIdx.x == 128) {
    CUTE_UNROLL
    for (int acc = 0; acc < kUmmaAccumulators; ++acc) {
      if (acc < item_count) {
        const int last = acc + ((item_count - 1 - acc) / kUmmaAccumulators) * kUmmaAccumulators;
        wait_barrier(storage.accumulator_free[acc], (last / kUmmaAccumulators) & 1);
      }
    }
  }
  __syncthreads();

  // The scheduler may now repurpose this shared arena for another phase.
  // All final asynchronous acknowledgements completed before the crew join.
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int slot = 0; slot < kStages; ++slot) {
      cutlass::arch::ClusterBarrier::invalidate(&storage.stage_full[slot]);
      cutlass::arch::ClusterBarrier::invalidate(&storage.slot_free[slot]);
    }
    CUTE_UNROLL
    for (int acc = 0; acc < kUmmaAccumulators; ++acc) {
      cutlass::arch::ClusterBarrier::invalidate(&storage.accumulator_full[acc]);
      cutlass::arch::ClusterBarrier::invalidate(&storage.accumulator_free[acc]);
    }
  }
  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}

// Production body: 4 K16 sub-tiles per slot (16 KB), 4-deep ring -- the LM
// body's slot cadence and bytes-in-flight verbatim, 4 slots per item.
__device__ inline void execute_umma_bf16(const Args& args) {
  execute_umma_impl<4, 4>(args);
}

// Wide-slot variant: 8 K16 sub-tiles per slot (32 KB, so each vocab row is
// read as one 256-byte contiguous span instead of two 128-byte ones) with a
// 3-deep ring at 96 KB in flight, 2 slots per item. Trades ring depth for
// per-row contiguity and halves the per-slot barrier/commit overhead.
template <int kRows = 1>
__device__ inline void execute_umma_bf16_wide(const Args& args) {
  execute_umma_impl<8, 3, kRows>(args);
}

#endif  // __CUDA_ARCH__ >= 1000

}  // namespace dspark_markov
