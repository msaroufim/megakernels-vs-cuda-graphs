// Self-contained LM-head row phase bodies for the DeepSeek-V4 DSpark
// megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// Frozen semantics (ledger #66/#81): five per-row FP32 accumulators sharing
// each vocabulary weight row, columns ascending with float4-quantum ordering,
// FP32 outputs. The staged body reproduces the scalar chain bitwise; only the
// weight movement changes (cooperative coalesced staging through shared
// memory instead of one uncoalesced 16 KB stream per thread).
#pragma once

#include "dspark_tmem.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "dspark_batch.h"

namespace dspark_lm {

constexpr int kVocab = 129280;
constexpr int kHidden = 4096;
constexpr int kOutputTile = 128;
constexpr int kDraftBlock = 5;
constexpr int kColumnChunk = 128;
// +1 float pad keeps the per-thread column walk bank-conflict free.
constexpr int kChunkStride = kColumnChunk + 1;
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;

struct Args {
  const float* lm_head;
  const __nv_bfloat16* normalized;
  float* base_logits;
  uint32_t begin;
  uint32_t end;
  // Probe-only (bench lane): BF16 copy of the LM head for the wmma body.
  // Unused (and left null) by the production scalar/staged bodies.
  const __nv_bfloat16* lm_head_bf16 = nullptr;
};

// Retained #81 scalar body, verbatim: per-thread float4 weight stream.
__device__ inline void execute_scalar(const Args& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int output = static_cast<int>(item) * kOutputTile
        + static_cast<int>(threadIdx.x);
    if (threadIdx.x >= kOutputTile || output >= kVocab) {
      continue;
    }
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int64_t weight_base = static_cast<int64_t>(output) * kHidden;
    const float4* weight_vectors =
        reinterpret_cast<const float4*>(args.lm_head + weight_base);
    for (int column = 0; column < kHidden; column += 4) {
      const float4 weight_vector = weight_vectors[column / 4];
      const float* weights = reinterpret_cast<const float*>(&weight_vector);
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          result[row] = fmaf(
              __bfloat162float(
                  args.normalized[row * kHidden + column + offset]),
              weights[offset],
              result[row]);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.base_logits[static_cast<int64_t>(row) * kVocab + output] =
          result[row];
    }
  }
}

// Coalesced-staged body: identical arithmetic, double-buffered chunked
// weight movement. Warps 0-3 (the 128 output lanes) consume chunk c from one
// buffer while warps 4-7 stage chunk c+1 into the other, so the coalesced
// weight stream overlaps the FMA chain instead of barrier-stalling it.
__device__ inline void execute_staged(const Args& args) {
  constexpr int kQuadsPerRow = kColumnChunk / 4;
  static_assert(
      2 * kOutputTile * kChunkStride * static_cast<int>(sizeof(float))
          <= kSharedBytesBudget,
      "LM staging buffers must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  float* buffers[2] = {
      reinterpret_cast<float*>(dynamic_shared_memory),
      reinterpret_cast<float*>(dynamic_shared_memory)
          + kOutputTile * kChunkStride,
  };
  const int warp = static_cast<int>(threadIdx.x) / 32;

  auto stage_chunk = [&](
      int tile_base, int chunk, float* destination, int lane, int lanes) {
    for (int index = lane; index < kOutputTile * kQuadsPerRow;
         index += lanes) {
      const int output_lane = index / kQuadsPerRow;
      const int quad = index % kQuadsPerRow;
      const int output = tile_base + output_lane;
      if (output >= kVocab) {
        continue;
      }
      const float4 weight_vector = *reinterpret_cast<const float4*>(
          args.lm_head
          + static_cast<int64_t>(output) * kHidden + chunk + quad * 4);
      const float* weights = reinterpret_cast<const float*>(&weight_vector);
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        destination[output_lane * kChunkStride + quad * 4 + offset] =
            weights[offset];
      }
    }
  };

  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int tile_base = static_cast<int>(item) * kOutputTile;
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Prologue: everyone stages chunk 0.
    stage_chunk(tile_base, 0, buffers[0], static_cast<int>(threadIdx.x), 256);
    __syncthreads();
    for (int chunk_index = 0; chunk_index < kHidden / kColumnChunk;
         ++chunk_index) {
      const int chunk = chunk_index * kColumnChunk;
      if (warp < 4) {
        const int output = tile_base + static_cast<int>(threadIdx.x);
        if (output < kVocab) {
          const float* chunk_weights =
              buffers[chunk_index % 2] + threadIdx.x * kChunkStride;
#pragma unroll 4
          for (int offset = 0; offset < kColumnChunk; ++offset) {
            const int column = chunk + offset;
#pragma unroll
            for (int row = 0; row < kDraftBlock; ++row) {
              result[row] = fmaf(
                  __bfloat162float(args.normalized[row * kHidden + column]),
                  chunk_weights[offset],
                  result[row]);
            }
          }
        }
      } else if (chunk_index + 1 < kHidden / kColumnChunk) {
        stage_chunk(
            tile_base,
            chunk + kColumnChunk,
            buffers[(chunk_index + 1) % 2],
            static_cast<int>(threadIdx.x) - 128,
            128);
      }
      __syncthreads();
    }
    const int output = tile_base + static_cast<int>(threadIdx.x);
    if (threadIdx.x < kOutputTile && output < kVocab) {
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        args.base_logits[static_cast<int64_t>(row) * kVocab + output] =
            result[row];
      }
    }
  }
}

// PROBE body (contract-sizing only, NOT retained, NOT called by the
// production kernel): BF16 weights + wmma 16x16x16 with FP32 accumulate.
// Halves LM weight traffic (2.11 GB FP32 -> 1.06 GB BF16) and replaces the
// five serial FMA chains with tensor-core K=16 reduction trees. This is
// exactly the #65-class arithmetic the frozen base_logits mean bound
// (4.5e-5) rejects; the bench lane measures the latency prize AND the drift
// so a contract recalibration can be sized before anyone signs anything.
__device__ inline void execute_wmma_bf16(const Args& args) {
  using namespace nvcuda;
  constexpr int kTile = 16;
  constexpr int kARows = 16;  // 5 real draft rows, 11 zero rows of padding
  static_assert(
      kARows * kHidden * static_cast<int>(sizeof(__nv_bfloat16))
              + 8 * kTile * kTile * static_cast<int>(sizeof(float))
          <= kSharedBytesBudget,
      "padded A + per-warp C tiles must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  __nv_bfloat16* a_padded =
      reinterpret_cast<__nv_bfloat16*>(dynamic_shared_memory);
  float* c_tiles = reinterpret_cast<float*>(
      dynamic_shared_memory + kARows * kHidden * sizeof(__nv_bfloat16));

  // Stage the five real activation rows once, zero-fill rows 5..15 so the
  // 16-row A fragment never reads past the 5-row normalized buffer.
  for (int index = static_cast<int>(threadIdx.x); index < kARows * kHidden;
       index += static_cast<int>(blockDim.x)) {
    const int row = index / kHidden;
    a_padded[index] = row < kDraftBlock
        ? args.normalized[row * kHidden + (index % kHidden)]
        : __float2bfloat16(0.0f);
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    // 8 warps x 16 outputs cover the 128-output tile; kVocab is an exact
    // multiple of 128, so no warp's sub-tile ever straddles the vocab end.
    const int tile_base =
        static_cast<int>(item) * kOutputTile + warp * kTile;
    if (tile_base >= kVocab) {
      continue;
    }
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::fragment<
        wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16, wmma::col_major>
        b_frag;
    const __nv_bfloat16* weight_base =
        args.lm_head_bf16 + static_cast<int64_t>(tile_base) * kHidden;
    for (int k = 0; k < kHidden; k += kTile) {
      wmma::load_matrix_sync(a_frag, a_padded + k, kHidden);
      // col_major with ldb = kHidden reads element (k', n) from
      // weight_base[n * kHidden + k + k'], i.e. W[output=tile_base+n][k+k'].
      wmma::load_matrix_sync(b_frag, weight_base + k, kHidden);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    float* c_tile = c_tiles + warp * kTile * kTile;
    wmma::store_matrix_sync(c_tile, acc, kTile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kDraftBlock * kTile; index += 32) {
      const int row = index / kTile;
      const int column = index % kTile;
      args.base_logits[
          static_cast<int64_t>(row) * kVocab + tile_base + column] =
          c_tile[row * kTile + column];
    }
    __syncwarp();
  }
}

// PROBE body #2 (relaxed program stage 3): the wmma body above is
// latency-bound on direct-GMEM B-fragment loads (~1.8 TB/s achieved). This
// variant double-buffers 128-column A/B chunks through shared memory with
// cp.async. The mma sequence (same 256 K-tiles, same accumulator, same
// order) is unchanged, so outputs must match execute_wmma_bf16 bitwise.
__device__ inline void cp_async_16(void* smem, const void* gmem) {
  const unsigned address =
      static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address),
      "l"(gmem));
}

__device__ inline void execute_wmma_bf16_pipelined(const Args& args) {
  using namespace nvcuda;
  constexpr int kTile = 16;
  constexpr int kARows = 16;
  constexpr int kChunk = 128;                  // columns per stage
  constexpr int kChunks = kHidden / kChunk;    // 32
  constexpr int kWarps = 8;
  constexpr int kABytes = kARows * kChunk * 2; // 4 KB per buffer
  constexpr int kBBytes = kTile * kChunk * 2;  // 4 KB per warp per buffer
  static_assert(
      2 * kABytes + 2 * kWarps * kBBytes
              + kWarps * kTile * kTile * static_cast<int>(sizeof(float))
          <= kSharedBytesBudget,
      "pipelined LM staging must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  __nv_bfloat16* a_buffers =
      reinterpret_cast<__nv_bfloat16*>(dynamic_shared_memory);
  __nv_bfloat16* b_buffers = a_buffers + 2 * kARows * kChunk;
  float* c_tiles =
      reinterpret_cast<float*>(b_buffers + 2 * kWarps * kTile * kChunk);

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;

  // Rows 5..15 of both A buffers stay zero for the whole task.
  for (int index = static_cast<int>(threadIdx.x);
       index < 2 * kARows * kChunk;
       index += static_cast<int>(blockDim.x)) {
    if ((index / kChunk) % kARows >= kDraftBlock) {
      a_buffers[index] = __float2bfloat16(0.0f);
    }
  }
  __syncthreads();

  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int tile_base =
        static_cast<int>(item) * kOutputTile + warp * kTile;
    if (tile_base >= kVocab) {
      continue;
    }
    const __nv_bfloat16* weight_base =
        args.lm_head_bf16 + static_cast<int64_t>(tile_base) * kHidden;

    auto stage_chunk = [&](int chunk, int buffer) {
      __nv_bfloat16* a_dst = a_buffers + buffer * kARows * kChunk;
      // A: rows 0..4, 8 columns per 16-byte packet: 5*16 = 80 block-wide ops.
      for (int index = static_cast<int>(threadIdx.x);
           index < kDraftBlock * (kChunk / 8);
           index += static_cast<int>(blockDim.x)) {
        const int row = index / (kChunk / 8);
        const int packet = index % (kChunk / 8);
        cp_async_16(
            a_dst + row * kChunk + packet * 8,
            args.normalized + row * kHidden + chunk * kChunk + packet * 8);
      }
      // B: this warp's 16 output rows, 16 packets each: 8 ops per lane.
      __nv_bfloat16* b_dst =
          b_buffers + (buffer * kWarps + warp) * kTile * kChunk;
      for (int index = lane; index < kTile * (kChunk / 8); index += 32) {
        const int row = index / (kChunk / 8);
        const int packet = index % (kChunk / 8);
        cp_async_16(
            b_dst + row * kChunk + packet * 8,
            weight_base + static_cast<int64_t>(row) * kHidden
                + chunk * kChunk + packet * 8);
      }
      asm volatile("cp.async.commit_group;\n");
    };

    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::fragment<
        wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16, wmma::col_major>
        b_frag;

    stage_chunk(0, 0);
    for (int chunk = 0; chunk < kChunks; ++chunk) {
      const int buffer = chunk & 1;
      if (chunk + 1 < kChunks) {
        stage_chunk(chunk + 1, (chunk + 1) & 1);
        asm volatile("cp.async.wait_group 1;\n");
      } else {
        asm volatile("cp.async.wait_group 0;\n");
      }
      __syncthreads();
      const __nv_bfloat16* a_src = a_buffers + buffer * kARows * kChunk;
      const __nv_bfloat16* b_src =
          b_buffers + (buffer * kWarps + warp) * kTile * kChunk;
      for (int k = 0; k < kChunk; k += kTile) {
        wmma::load_matrix_sync(a_frag, a_src + k, kChunk);
        wmma::load_matrix_sync(b_frag, b_src + k, kChunk);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }
      // The buffer just consumed is restaged next iteration; every warp must
      // be past its mma reads before any thread issues new copies into it.
      __syncthreads();
    }

    float* c_tile = c_tiles + warp * kTile * kTile;
    wmma::store_matrix_sync(c_tile, acc, kTile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kDraftBlock * kTile; index += 32) {
      const int row = index / kTile;
      const int column = index % kTile;
      args.base_logits[
          static_cast<int64_t>(row) * kVocab + tile_base + column] =
          c_tile[row * kTile + column];
    }
    __syncwarp();
  }
}

// PROBE body #3 (relaxed program stage 3, iteration 3): TCGen05 BF16 UMMA.
// Iterations 1-2 falsified load latency and DRAM locality as the wmma
// ceiling; the limiter is per-fragment overhead of the 16x16x16 wmma tiles
// (~1 KB of SMEM fragment loads per mma doubles the SMEM+issue traffic vs
// the GMEM stream). This body replaces the eight per-warp 16x16 fragments
// with ONE M=128 x N=8 x K=16 tcgen05 UMMA per K-slice: the whole 128-output
// vocab tile is a single instruction, operands are read by the tensor core
// directly from SMEM descriptors (no register fragments), and the FP32
// accumulator lives in TMEM across all 256 chained K-slices (scaleC=0 opens
// each item, scaleC=1 chains — the dspark_shared_phase.cuh precedent).
//
// Roles mirror dspark_w13_phase.cuh (its comments encode the two deadlock
// fixes this copies: cpasync_barrier_arrive_noinc for the cp.async stage
// barriers, and per-slot mbarriers so every parity wait is at most one
// completion behind its barrier):
//   warps 5-7  stage 128x64 BF16 weight slots through a 4-deep cp.async ring
//   warp 4     issues 4 chained UMMAs per slot + per-slot tcgen05.commit
//   warps 0-3  drain the 128x8 TMEM accumulator once per item and store the
//              five real draft rows
// The padded 8 x 4096 activation matrix (5 real rows + 3 zero rows) is
// staged ONCE per invocation as 256 contiguous 256-byte K-major INTER
// chunks; per-chunk descriptors are base + chunk * 16 (256 B >> 4).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

using LmUmmaAtom = cute::SM100_MMA_F16BF16_SS<
    cute::bfloat16_t, cute::bfloat16_t, float, 128, 8,
    cute::UMMA::Major::K, cute::UMMA::Major::K>;
using LmUmmaTiledMma = decltype(cute::make_tiled_mma(LmUmmaAtom{}));
using LmUmmaShapeA = decltype(cute::partition_shape_A(
    LmUmmaTiledMma{},
    cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<16>{})));
using LmUmmaShapeB = decltype(cute::partition_shape_B(
    LmUmmaTiledMma{},
    cute::make_shape(cute::Int<8>{}, cute::Int<16>{})));
// Layout_K_INTER_Atom<bf16> tiles to (m,k) -> m*8 + k%8 + (k/8)*1024 for the
// 128x16 A sub-tile (4 KB) and n*8 + k%8 + (k/8)*64 for the 8x16 B chunk
// (256 B), verified against cute's layout algebra host-side.
using LmUmmaSmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, LmUmmaShapeA{}));
using LmUmmaSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, LmUmmaShapeB{}));

constexpr int kUmmaK = 16;                          // K per tcgen05.mma
constexpr int kUmmaSubTiles = 4;                    // K16 sub-tiles per slot
// log2(kUmmaSubTiles). The staging lane->packet map below decomposes the
// packet index with shifts only; a signed / or % by a constant would cost
// a negative-operand correction, and the megakernel has no register to spare.
constexpr int kUmmaSubTileShift = 2;
static_assert((1 << kUmmaSubTileShift) == kUmmaSubTiles, "shift must match");
constexpr int kUmmaSlotK = kUmmaK * kUmmaSubTiles;  // 64 columns per stage
constexpr int kUmmaSlotElements = kOutputTile * kUmmaSlotK;
// Production keeps the proven four-stage/two-drain-buffer geometry. The
// opt-in stage-5 experiment attacked the profiler's hottest addressable LM
// stall (slot_free): one 4 KB partial is enough when the drain crew performs
// an explicit reuse rendezvous after issuing its global stores.  That trade
// frees exactly one 4 KB buffer, which lets a fifth 16 KB A stage fit under
// the unchanged 153,600-byte megakernel launch envelope.
//
// FALSIFIED as a production default on GB300: seven interleaved points gave
// a tiny 2.607 us / 0.18% wall signal, while the device trace moved LM from
// 237.0 to 240.8 us. Full NCU explains the disagreement: slot_free
// long-scoreboard samples fall 2248 -> 2049, but accumulator_full rises
// 634 -> 863. The fifth stage only moves the wait from staging to MMA/drain.
#if defined(DSPARK_LM_STAGE5_RING)
constexpr int kUmmaStages = 5;
constexpr int kUmmaPartialBuffers = 1;
#else
constexpr int kUmmaStages = 4;
constexpr int kUmmaPartialBuffers = 2;
#endif
constexpr int kUmmaChunks = kHidden / kUmmaK;       // 256 B-descriptor chunks
constexpr int kUmmaSlotsPerItem = kHidden / kUmmaSlotK;

struct LmUmmaSharedStorage {
  alignas(128) __nv_bfloat16 b_all[kUmmaChunks * 8 * kUmmaK];        // 64 KB
  alignas(128) __nv_bfloat16 a_ring[kUmmaStages][kUmmaSlotElements];
  alignas(16) float partial[kUmmaPartialBuffers][kOutputTile * 8];
  alignas(16) cute::uint64_t stage_full[kUmmaStages];
  alignas(16) cute::uint64_t slot_free[kUmmaStages];
  alignas(16) cute::uint64_t accumulator_full[1];
  alignas(16) cute::uint64_t accumulator_free[1];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

// kStagingVariant selects the cp.async STAGING POLICY. It changes only which
// lane issues which packet and which cache operator the copy uses; the same
// 1024 packets land at the same shared addresses with the same bytes, so
// every variant is BITWISE identical by construction.
//   bit 0: cp.async.cg (CacheOperation::Global, L1-bypass) instead of the
//          shipped cp.async.ca. The lm_head stream is read exactly once, so
//          allocating it in L1 is pure pollution of the carveout that the
//          same SM's shared memory is taken from.
//   bit 1: m-consecutive quarter-warp destination map. See the R7 note in the
//          staging loop.
//
// kStagingProbe is a BENCH-ONLY timing attribution mask: bit 0 drops the
// lm_head weight stream from the ring. GARBAGE NUMERICS; it exists only to
// separate the band's memory cost from its MMA/barrier/drain skeleton.
// Production never instantiates a non-zero mask.
template <int kStagingProbe = 0, int kStagingVariant = 0>
__device__ inline void execute_umma_bf16(const Args& args) {
  using namespace cute;
#if !defined(DSPARK_W13_STAGING_PROBE)
  static_assert(
      kStagingProbe == 0,
      "staging attribution probes require -DDSPARK_W13_STAGING_PROBE");
#endif
  constexpr bool kProbeSkipWeights = (kStagingProbe & 1) != 0;
  constexpr cutlass::arch::CacheOperation::Kind kStagingCacheOp =
      (kStagingVariant & 1) != 0 ? cutlass::arch::CacheOperation::Global
                                 : cutlass::arch::CacheOperation::Always;
  constexpr bool kMConsecutive = (kStagingVariant & 2) != 0;
  constexpr int kStagingThreads = 3 * 32;
#ifdef DSPARK_LM_BULK_STAGING
  constexpr int kStageArrivals = 1;
#else
  constexpr int kStageArrivals = kStagingThreads;
#endif
  constexpr int kTmemColumns = 32;
  static_assert(
      sizeof(LmUmmaSharedStorage) <= kSharedBytesBudget,
      "UMMA staging ring + padded B must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<LmUmmaSharedStorage*>(dynamic_shared_memory);
  LmUmmaTiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<kOutputTile>{}, Int<8>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = static_cast<int>(threadIdx.x) / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(
        storage.stage_full, kStageArrivals);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(storage.slot_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Padded B, staged once: rows 0-4 real activations, rows 5-7 zeros. One
  // 16-byte packet covers 8 K-contiguous elements of one row; chunk c holds
  // K columns [c*16, c*16+16) at byte base c*256 (half-atoms at +0/+128).
  // These generic-proxy stores must be visible to the MMA's async proxy
  // before the first tcgen05.mma issues (fence below, then __syncthreads).
  for (int index = static_cast<int>(threadIdx.x); index < kUmmaChunks * 16;
       index += static_cast<int>(blockDim.x)) {
    const int chunk = index / 16;
    const int row = (index % 16) / 2;
    const int half = index & 1;
    uint4 packet = make_uint4(0u, 0u, 0u, 0u);
    if (row < kDraftBlock) {
      packet = *reinterpret_cast<const uint4*>(
          args.normalized + row * kHidden + chunk * kUmmaK + half * 8);
    }
    *reinterpret_cast<uint4*>(
        reinterpret_cast<unsigned char*>(storage.b_all)
        + chunk * 256 + half * 128 + row * 16) = packet;
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

  // Materialized descriptors, replicating mma_unpack (w13 precedent: cute's
  // degenerate single-atom descriptor-tensor gemm dispatch trips NVCC).
  uint64_t a_descriptor[kUmmaStages];
  CUTE_UNROLL
  for (int s = 0; s < kUmmaStages; ++s) {
    a_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[s])),
            layout<0>(LmUmmaSmemLayoutA{}))).desc_;
  }
  const uint64_t b_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(
      make_tensor(
          make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_all)),
          layout<0>(LmUmmaSmemLayoutB{}))).desc_;
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_slots = item_count * kUmmaSlotsPerItem;

#ifdef DSPARK_LM_BULK_STAGING
  if (warp >= 5) {
    // Setup packs weights in the existing operand layout. One bulk copy
    // supplies four consecutive K16 operands; their MMA order is unchanged.
    // The transaction barrier orders the copy before consumption, while
    // slot_free still orders the next copy after the previous MMA completion.
    if (threadIdx.x == 160) {
      for (int c = 0; c < total_slots; ++c) {
        const int s = c % kUmmaStages;
        if (c >= kUmmaStages) {
          wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
        }
        const int item = static_cast<int>(args.begin) + c / kUmmaSlotsPerItem;
        const int slot_in_item = c % kUmmaSlotsPerItem;
        constexpr int bytes = kUmmaSlotElements * sizeof(__nv_bfloat16);
        const auto* source = args.lm_head_bf16
            + (static_cast<int64_t>(item) * kUmmaSlotsPerItem + slot_in_item)
                * kUmmaSlotElements;
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &storage.stage_full[s], bytes);
        cute::SM90_BULK_COPY_G2S::copy(
            source, &storage.stage_full[s], storage.a_ring[s], bytes);
      }
    }
#else
  if (warp >= 5) {
    // Staging crew: 1024 16-byte cp.async packets per 128x64 slot. Slot s is
    // reusable once the commit of the ring round that consumed it completes,
    // observed through its per-slot slot_free barrier (one completion per
    // round, so the parity wait is exactly one completion behind).
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      if (c >= kUmmaStages) {
        wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
      }
      const int item = static_cast<int>(args.begin) + c / kUmmaSlotsPerItem;
      const int64_t weight_base =
          static_cast<int64_t>(item) * kOutputTile * kHidden;
      const int k_base = (c % kUmmaSlotsPerItem) * kUmmaSlotK;
      unsigned char* slot = reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      // R6b lane->packet map (bank-conflict free). The natural map
      // (m = index >> 3: eight consecutive lanes issue ONE row's eight
      // packets) puts a whole quarter-warp at SMEM byte offsets 2048 apart,
      // and 2048 % 128 == 0, so all eight land in the SAME four banks: an
      // 8-WAY CONFLICT on every cp.async store, the write-side twin of the
      // attention body's ldmatrix pathology. Interleaving two bits of m below
      // j spreads a quarter-warp over four rows (2-way) while keeping a row's
      // two 16-byte halves adjacent, so each lane pair still covers one full
      // 32-byte global sector -- the GLOBAL request pattern is unchanged.
      // This is a pure re-assignment of packets to lanes: the same 1024
      // packets land at the same SMEM addresses, so it is BITWISE by
      // construction.
      //
      // R7 variant bit 1 (m-consecutive): the R6b map above still puts a
      // lane PAIR (the two halves of one row) 2048 B apart, and 2048 % 128
      // == 0, so a quarter-warp is still 2-WAY conflicted. Putting the three
      // low packet bits on m instead makes a quarter-warp's eight 16-byte
      // destinations one contiguous 128-byte window -- all 32 banks, exactly
      // one wavefront. It is not free: the same eight lanes then read eight
      // DIFFERENT weight rows (8192 B apart), so the quarter-warp's global
      // side goes from 4 fully consumed 128 B lines to 8 half-consumed ones.
      // L2 sector traffic is unchanged (the warp still covers 512 contiguous
      // bytes per row group); only the L1 tag-request count doubles.
      static_assert(kOutputTile % 4 == 0, "quarter-warp spreads 4 rows");
      static_assert(kOutputTile % 8 == 0, "m-consecutive map needs 8 rows");
      for (int index =
               (kProbeSkipWeights ? kOutputTile * kUmmaSubTiles * 2 : lane);
           index < kOutputTile * kUmmaSubTiles * 2;
           index += kStagingThreads) {
        const unsigned packet = static_cast<unsigned>(index);
        int half;
        int j;
        int m;
        if (kMConsecutive) {
          half = static_cast<int>((packet >> 3) & 1u);
          j = static_cast<int>(
              (packet >> 4) & static_cast<unsigned>(kUmmaSubTiles - 1));
          m = static_cast<int>(
              ((packet >> (4 + kUmmaSubTileShift)) << 3) | (packet & 7u));
        } else {
          half = static_cast<int>(packet & 1u);
          j = static_cast<int>(
              (packet >> 3) & static_cast<unsigned>(kUmmaSubTiles - 1));
          m = static_cast<int>(
              ((packet >> (3 + kUmmaSubTileShift)) << 2) | ((packet >> 1) & 3u));
        }
        cutlass::arch::cp_async<16, kStagingCacheOp>(
            slot + j * 4096 + half * 2048 + m * 16,
            args.lm_head_bf16 + weight_base
                + static_cast<int64_t>(m) * kHidden
                + k_base + j * kUmmaK + half * 8);
      }
      // noinc: the barrier's expected count is the 96 staging threads.
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
#endif
  } else if (warp == 4) {
    // MMA crew: four chained K=16 UMMAs per slot into ONE TMEM accumulator;
    // scaleC=0 opens each item, 1 chains. fma and umma_arrive are
    // elect_one_sync-guarded internally.
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      const int slot_in_item = c % kUmmaSlotsPerItem;
      const int item_local = c / kUmmaSlotsPerItem;
      wait_barrier(storage.stage_full[s], (c / kUmmaStages) & 1);
      if (slot_in_item == 0 && item_local > 0) {
        // The drain crew must be done reading TMEM for the previous item
        // before scaleC=0 overwrites the accumulator.
        wait_barrier(storage.accumulator_free[0], (item_local - 1) & 1);
      }
      CUTE_UNROLL
      for (int j = 0; j < kUmmaSubTiles; ++j) {
        LmUmmaAtom::fma(
            a_descriptor[s] + static_cast<uint64_t>(j) * 256,   // 4 KB >> 4
            b_descriptor_base + static_cast<uint64_t>(
                slot_in_item * kUmmaSubTiles + j) * 16,         // 256 B >> 4
            storage.tmem_base_ptr,
            (slot_in_item == 0 && j == 0) ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kUmmaSlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[0]);
      }
    }
  } else {
    // Drain crew (warps 0-3): one TMEM drain per item. The production path
    // ping-pongs two partials. The stage-5 experiment uses one partial and a
    // second named-barrier rendezvous after the shared reads/global-store
    // issues, before any drain warp may overwrite the buffer.
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
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
#if !defined(DSPARK_LM_STAGE5_RING)
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
#endif
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));

    for (int i = 0; i < item_count; ++i) {
      const int parity = i & 1;
      wait_barrier(storage.accumulator_full[0], parity);
      copy(tmem_to_register, thread_tmem, register_accumulator);
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[0]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
#if defined(DSPARK_LM_STAGE5_RING)
        copy(register_accumulator, thread_partial_low);
#else
        copy(register_accumulator, thread_partial_high);
#endif
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const int64_t tile_base =
          (static_cast<int64_t>(args.begin) + i) * kOutputTile;
      CUTE_UNROLL
      for (int row = 0; row < kDraftBlock; ++row) {
        args.base_logits[
            static_cast<int64_t>(row) * kVocab + tile_base + threadIdx.x] =
            storage.partial[parity % kUmmaPartialBuffers][
                row * kOutputTile + static_cast<int>(threadIdx.x)];
      }
#if defined(DSPARK_LM_STAGE5_RING)
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
#endif
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}

#endif  // __CUDA_ARCH__ >= 1000

// PROBE body #4 (TP2 sharding campaign, S1 step 0): M=64 fine-tile variant
// of execute_umma_bf16. A work item is a 64-row vocab tile, so the TP2
// balanced-span assignment quantizes at half the granularity (505 -> 1010
// items per 148-CTA half-vocab shard: bound drops from 4x128 = 512 rows to
// 7x64 = 448, ideal 436.8). The per-logit K-accumulation chain is UNCHANGED:
// each vocab row still sees 256 chained K=16 tcgen05 MMAs in ascending K
// order with the FP32 accumulator resident in TMEM (scaleC=0 opens each
// item, 1 chains) — only the M-extent of each MMA drops from 128 to 64 rows,
// which parallelizes across independent output rows and cannot reorder any
// row's reduction. Slots keep the 16 KB cadence of the 128-row body (64 rows
// x 128 K per stage = 8 K16 sub-tiles), so the cp.async ring, barrier
// pattern, and bytes-in-flight are identical; only the TMEM drain changes
// (M=64 1-SM accumulators live in the FIRST 16 datapaths of each 32-DP
// subpartition, which is exactly what SM100_TMEM_LOAD_16dp32b1x reads — the
// cutlass sm100 epilogue builder's own num_dp==16 selection).
// execute_umma_bf16 above is production-frozen and untouched.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

constexpr int kFineTile = 64;

using LmUmmaM64Atom = cute::SM100_MMA_F16BF16_SS<
    cute::bfloat16_t, cute::bfloat16_t, float, kFineTile, 8,
    cute::UMMA::Major::K, cute::UMMA::Major::K>;
using LmUmmaM64TiledMma = decltype(cute::make_tiled_mma(LmUmmaM64Atom{}));
using LmUmmaM64ShapeA = decltype(cute::partition_shape_A(
    LmUmmaM64TiledMma{},
    cute::make_shape(cute::Int<kFineTile>{}, cute::Int<16>{})));
// Layout_K_INTER_Atom<bf16> is the 8x8 (m,k) -> m*8 + k atom, so the 64x16
// A sub-tile tiles to (m,k) -> m*8 + k%8 + (k/8)*512: one K=16 sub-tile is
// 2048 B with K-halves at +0/+1024 and row m at byte m*16 within a half.
using LmUmmaM64SmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, LmUmmaM64ShapeA{}));
// B (the padded 8 x 4096 activation matrix) is IDENTICAL to the 128-row
// body: same b_all staging, same LmUmmaSmemLayoutB chunk descriptors.

constexpr int kM64SubTiles = 8;                       // K16 sub-tiles per slot
constexpr int kM64SlotK = kUmmaK * kM64SubTiles;      // 128 columns per stage
constexpr int kM64SlotElements = kFineTile * kM64SlotK;
constexpr int kM64SlotsPerItem = kHidden / kM64SlotK; // 32

struct LmUmmaM64SharedStorage {
  alignas(128) __nv_bfloat16 b_all[kUmmaChunks * 8 * kUmmaK];       // 64 KB
  alignas(128) __nv_bfloat16 a_ring[kUmmaStages][kM64SlotElements]; // 64 KB
  alignas(16) float partial[2][kFineTile * 8];                      // 4 KB
  alignas(16) cute::uint64_t stage_full[kUmmaStages];
  alignas(16) cute::uint64_t slot_free[kUmmaStages];
  alignas(16) cute::uint64_t accumulator_full[1];
  alignas(16) cute::uint64_t accumulator_free[1];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

__device__ inline void execute_umma_bf16_m64(const Args& args) {
  using namespace cute;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kTmemColumns = 32;
  static_assert(
      sizeof(LmUmmaM64SharedStorage) <= kSharedBytesBudget,
      "M64 UMMA staging ring + padded B must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<LmUmmaM64SharedStorage*>(dynamic_shared_memory);
  LmUmmaM64TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<kFineTile>{}, Int<8>{}));
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
        storage.accumulator_free, 4 * 32);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Padded B, staged once — verbatim from execute_umma_bf16.
  for (int index = static_cast<int>(threadIdx.x); index < kUmmaChunks * 16;
       index += static_cast<int>(blockDim.x)) {
    const int chunk = index / 16;
    const int row = (index % 16) / 2;
    const int half = index & 1;
    uint4 packet = make_uint4(0u, 0u, 0u, 0u);
    if (row < kDraftBlock) {
      packet = *reinterpret_cast<const uint4*>(
          args.normalized + row * kHidden + chunk * kUmmaK + half * 8);
    }
    *reinterpret_cast<uint4*>(
        reinterpret_cast<unsigned char*>(storage.b_all)
        + chunk * 256 + half * 128 + row * 16) = packet;
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

  uint64_t a_descriptor[kUmmaStages];
  CUTE_UNROLL
  for (int s = 0; s < kUmmaStages; ++s) {
    a_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[s])),
            layout<0>(LmUmmaM64SmemLayoutA{}))).desc_;
  }
  const uint64_t b_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(
      make_tensor(
          make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_all)),
          layout<0>(LmUmmaSmemLayoutB{}))).desc_;
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_slots = item_count * kM64SlotsPerItem;

  if (warp >= 5) {
    // Staging crew: 1024 16-byte cp.async packets per 64x128 slot (same
    // packet count and barrier cadence as the 128-row body's 128x64 slots).
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      if (c >= kUmmaStages) {
        wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
      }
      const int item = static_cast<int>(args.begin) + c / kM64SlotsPerItem;
      const int64_t weight_base =
          static_cast<int64_t>(item) * kFineTile * kHidden;
      const int k_base = (c % kM64SlotsPerItem) * kM64SlotK;
      unsigned char* slot = reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      for (int index = lane; index < kFineTile * kM64SubTiles * 2;
           index += kStagingThreads) {
        const int m = index >> 4;      // consecutive lanes walk one row
        const int sub = index & 15;
        const int j = sub >> 1;
        const int half = sub & 1;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            slot + j * 2048 + half * 1024 + m * 16,
            args.lm_head_bf16 + weight_base
                + static_cast<int64_t>(m) * kHidden
                + k_base + j * kUmmaK + half * 8);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    // MMA crew: eight chained K=16 UMMAs per slot into ONE TMEM accumulator.
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      const int slot_in_item = c % kM64SlotsPerItem;
      const int item_local = c / kM64SlotsPerItem;
      wait_barrier(storage.stage_full[s], (c / kUmmaStages) & 1);
      if (slot_in_item == 0 && item_local > 0) {
        wait_barrier(storage.accumulator_free[0], (item_local - 1) & 1);
      }
      CUTE_UNROLL
      for (int j = 0; j < kM64SubTiles; ++j) {
        LmUmmaM64Atom::fma(
            a_descriptor[s] + static_cast<uint64_t>(j) * 128,  // 2 KB >> 4
            b_descriptor_base + static_cast<uint64_t>(
                slot_in_item * kM64SubTiles + j) * 16,         // 256 B >> 4
            storage.tmem_base_ptr,
            (slot_in_item == 0 && j == 0) ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kM64SlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[0]);
      }
    }
  } else {
    // Drain crew (warps 0-3): one TMEM drain per 64-row item. Each warp's
    // 16dp load reads the first 16 datapaths of its subpartition — exactly
    // where the M=64 interleaved accumulator lives.
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_16dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto slot_view = tmem_accumulator;
    slot_view.data() = storage.tmem_base_ptr;
    auto thread_tmem = thread_copy.partition_S(slot_view);
    auto partial_layout = make_layout(
        make_shape(Int<kFineTile>{}, Int<8>{}),
        make_stride(Int<1>{}, Int<kFineTile>{}));
    auto shared_partial_low = make_tensor(
        make_smem_ptr(storage.partial[0]), partial_layout);
    auto shared_partial_high = make_tensor(
        make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));

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
      const int64_t tile_base =
          (static_cast<int64_t>(args.begin) + i) * kFineTile;
      if (threadIdx.x < kFineTile) {
        CUTE_UNROLL
        for (int row = 0; row < kDraftBlock; ++row) {
          args.base_logits[
              static_cast<int64_t>(row) * kVocab + tile_base + threadIdx.x] =
              storage.partial[parity][
                  row * kFineTile + static_cast<int>(threadIdx.x)];
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

// PROBE body #5 (TP2 sharding campaign, S1 step 0, iteration 2): unified
// mixed-granularity body. Run 1 of the campaign measured the pure-M64 body
// at +46% single-GPU wall: a tcgen05 M=64 N=8 K=16 MMA costs about as much
// to issue as the M=128 one, so uniform fine tiles double the MMA-path time
// and it becomes the limiter (34.0 us per 64-row unit vs 22.6 staging-bound
// ideal). This body takes spans in 64-row units but processes the interior
// ALIGNED PAIRS as regular M=128 items and only the 0-2 span-edge units as
// M=64 — inside ONE pipeline invocation (same ring, same crews, same
// barriers), so the edge units' MMA overrun hides under the interior items'
// staging slack instead of standing alone. Per-logit K order is unchanged
// everywhere: every vocab row gets 256 chained K=16 MMAs ascending, scaleC=0
// opening its item. Ring slots are the same 16 KB either way (128r x 64K, 4
// sub-tiles for M128; 64r x 128K, 8 sub-tiles for M64); the M64 accumulator
// aliases the same TMEM columns (it only touches the first 16 datapaths of
// each subpartition) and items are serial through the accumulator chain, so
// no extra TMEM is needed. Reuses LmUmmaSharedStorage verbatim.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// The hybrid ring runs 2 stages of 32 KB slots (vs the frozen body's 4x16
// KB): run 3 of the campaign measured ~370 ns of per-slot MMA-warp overhead
// (barrier wait + commit latency) on top of ~86 ns per UMMA, which made the
// M64 edge slots (8 UMMAs each) the limiter. Doubling the K per slot halves
// the slot count, so the fixed per-slot cost is paid half as often while
// bytes-in-flight (64 KB) and per-logit K order are unchanged.
constexpr int kHybridStages = 2;
constexpr int kHy128SubTiles = 8;                        // K16 per M128 slot
constexpr int kHy128SlotK = kUmmaK * kHy128SubTiles;     // 128
constexpr int kHy128SlotsPerItem = kHidden / kHy128SlotK;  // 32
constexpr int kHy64SubTiles = 16;                        // K16 per M64 slot
constexpr int kHy64SlotK = kUmmaK * kHy64SubTiles;       // 256
constexpr int kHy64SlotsPerItem = kHidden / kHy64SlotK;  // 16
constexpr int kHybridSlotElements = kOutputTile * kHy128SlotK;  // 32 KB

// Three live accumulators in TMEM (32 columns allocated, 8 per accumulator):
// interior M128 items alternate accumulators 0/1 and the M64 edge units own
// accumulator 2, so an item's MMAs never wait on the PREVIOUS item's drain
// (run 4 measured per-item max(staging, MMA) serialization through the
// single accumulator as the reason the edge overrun could not hide).
constexpr int kHybridAccumulators = 3;

struct LmHybridSharedStorage {
  alignas(128) __nv_bfloat16 b_all[kUmmaChunks * 8 * kUmmaK];           // 64 KB
  alignas(128) __nv_bfloat16 a_ring[kHybridStages][kHybridSlotElements]; // 64 KB
  alignas(16) float partial[2][kOutputTile * 8];                         // 8 KB
  alignas(16) cute::uint64_t stage_full[kHybridStages];
  alignas(16) cute::uint64_t slot_free[kHybridStages];
  alignas(16) cute::uint64_t accumulator_full[kHybridAccumulators];
  alignas(16) cute::uint64_t accumulator_free[kHybridAccumulators];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

// Slot-stream walker for a fine-unit span [begin, end): optional M64 head
// unit (odd begin), interior M128 items, optional M64 tail unit (odd end).
// The edge units' slots are BRESENHAM-INTERLEAVED among the interior slots
// (one edge slot every ~n_int/n_edge interior slots, edges in head-then-tail
// order, each item's own slots in ascending K order — so every logit's
// K-accumulation chain is untouched). Interleaving spreads the edge units'
// MMA-bound overrun (an M64 UMMA executes in M128 time) across the whole
// span's staging slack instead of letting it stall the 2-slot ring at one
// point. All three crews replay the identical deterministic walk.
struct LmHybridWalk {
  // Schedule.
  int head;      // 0/1
  int interior;  // count of 128-row items
  int tail;      // 0/1
  int begin;     // fine-unit index of the head (or first interior unit)
  int aligned;   // first interior fine unit (== begin + head)
  int n_int;     // interior slots  (interior * 32)
  int n_edge;    // edge slots      ((head + tail) * 16)
  // Cursor.
  int j;         // interior slots emitted
  int k;         // edge slots emitted
  // Per-step outputs (valid after lm_hybrid_walk_next).
  int kind;          // 1 = interior M128 slot, 0 = edge M64 slot
  int item;          // ordinal WITHIN its chain (interior index / edge index)
  int slot_in_item;
  bool item_last;    // this is the item's final slot
  int unit;          // fine-unit base row of this slot's item
};

__device__ inline LmHybridWalk lm_hybrid_walk_init(int begin, int end) {
  LmHybridWalk w;
  w.head = ((begin & 1) != 0 && end > begin) ? 1 : 0;
  w.begin = begin;
  w.aligned = begin + w.head;
  w.interior = (end - w.aligned) / 2;
  w.tail = end - w.aligned - 2 * w.interior;  // 0 or 1
  w.n_int = w.interior * kHy128SlotsPerItem;
  w.n_edge = (w.head + w.tail) * kHy64SlotsPerItem;
  w.j = 0;
  w.k = 0;
  return w;
}

__device__ inline void lm_hybrid_walk_next(LmHybridWalk& w) {
  const bool emit_interior = w.j < w.n_int
      && (w.k >= w.n_edge
          || static_cast<long long>(w.j + 1) * w.n_edge
              <= static_cast<long long>(w.k + 1) * w.n_int);
  if (emit_interior) {
    w.kind = 1;
    w.item = w.j / kHy128SlotsPerItem;
    w.slot_in_item = w.j % kHy128SlotsPerItem;
    w.item_last = w.slot_in_item == kHy128SlotsPerItem - 1;
    w.unit = w.aligned + 2 * w.item;
    ++w.j;
  } else {
    w.kind = 0;
    w.item = w.k / kHy64SlotsPerItem;  // 0 = head (if present), else tail
    w.slot_in_item = w.k % kHy64SlotsPerItem;
    w.item_last = w.slot_in_item == kHy64SlotsPerItem - 1;
    w.unit = (w.head != 0 && w.item == 0)
        ? w.begin : w.aligned + 2 * w.interior;
    ++w.k;
  }
}

// Accumulator index for a chain item: interiors alternate accumulators 0/1
// (barrier round = ordinal / 2), edges own accumulator 2 (round = ordinal).
__device__ inline int lm_hybrid_acc(int kind, int item) {
  return kind == 1 ? (item & 1) : 2;
}

__device__ inline int lm_hybrid_acc_round(int kind, int item) {
  return kind == 1 ? (item >> 1) : item;
}

__device__ inline void execute_umma_bf16_hybrid(const Args& args) {
  using namespace cute;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kTmemColumns = 32;
  static_assert(
      sizeof(LmHybridSharedStorage) <= kSharedBytesBudget,
      "hybrid staging ring + padded B must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<LmHybridSharedStorage*>(dynamic_shared_memory);
  LmUmmaTiledMma tiled_mma_128;
  LmUmmaM64TiledMma tiled_mma_64;
  auto cta_mma_128 = tiled_mma_128.get_slice(Int<0>{});
  auto cta_mma_64 = tiled_mma_64.get_slice(Int<0>{});
  auto accumulator_128 = tiled_mma_128.make_fragment_C(partition_shape_C(
      tiled_mma_128, make_shape(Int<kOutputTile>{}, Int<8>{})));
  auto accumulator_64 = tiled_mma_64.make_fragment_C(partition_shape_C(
      tiled_mma_64, make_shape(Int<kFineTile>{}, Int<8>{})));
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = static_cast<int>(threadIdx.x) / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kHybridStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kHybridStages>(storage.slot_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kHybridAccumulators>(
        storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kHybridAccumulators>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Padded B, staged once — verbatim from execute_umma_bf16.
  for (int index = static_cast<int>(threadIdx.x); index < kUmmaChunks * 16;
       index += static_cast<int>(blockDim.x)) {
    const int chunk = index / 16;
    const int row = (index % 16) / 2;
    const int half = index & 1;
    uint4 packet = make_uint4(0u, 0u, 0u, 0u);
    if (row < kDraftBlock) {
      packet = *reinterpret_cast<const uint4*>(
          args.normalized + row * kHidden + chunk * kUmmaK + half * 8);
    }
    *reinterpret_cast<uint4*>(
        reinterpret_cast<unsigned char*>(storage.b_all)
        + chunk * 256 + half * 128 + row * 16) = packet;
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  accumulator_128.data() = storage.tmem_base_ptr;
  accumulator_64.data() = storage.tmem_base_ptr;

  uint64_t a_descriptor_128[kHybridStages];
  uint64_t a_descriptor_64[kHybridStages];
  CUTE_UNROLL
  for (int s = 0; s < kHybridStages; ++s) {
    auto base = make_smem_ptr(
        reinterpret_cast<bfloat16_t*>(storage.a_ring[s]));
    a_descriptor_128[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(base, layout<0>(LmUmmaSmemLayoutA{}))).desc_;
    a_descriptor_64[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(base, layout<0>(LmUmmaM64SmemLayoutA{}))).desc_;
  }
  const uint64_t b_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(
      make_tensor(
          make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_all)),
          layout<0>(LmUmmaSmemLayoutB{}))).desc_;
  const uint64_t idesc_128 =
      UMMA::make_runtime_instr_desc<>(tiled_mma_128.idesc_);
  const uint64_t idesc_64 =
      UMMA::make_runtime_instr_desc<>(tiled_mma_64.idesc_);

  LmHybridWalk walk = lm_hybrid_walk_init(
      static_cast<int>(args.begin), static_cast<int>(args.end));
  const int total_slots = walk.n_int + walk.n_edge;
  const int total_items = walk.head + walk.interior + walk.tail;

  if (warp >= 5) {
    // Staging crew: every slot is 32 KB / 2048 packets regardless of kind;
    // only the (row, column) decode differs.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      lm_hybrid_walk_next(walk);
      const int s = c % kHybridStages;
      if (c >= kHybridStages) {
        wait_barrier(storage.slot_free[s], (c / kHybridStages - 1) & 1);
      }
      const int64_t weight_base =
          static_cast<int64_t>(walk.unit) * kFineTile * kHidden;
      unsigned char* slot = reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      if (walk.kind == 1) {
        const int k_base = walk.slot_in_item * kHy128SlotK;
        for (int index = lane; index < kOutputTile * kHy128SubTiles * 2;
             index += kStagingThreads) {
          const int m = index >> 4;
          const int sub = index & 15;
          const int j = sub >> 1;
          const int half = sub & 1;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              slot + j * 4096 + half * 2048 + m * 16,
              args.lm_head_bf16 + weight_base
                  + static_cast<int64_t>(m) * kHidden
                  + k_base + j * kUmmaK + half * 8);
        }
      } else {
        const int k_base = walk.slot_in_item * kHy64SlotK;
        for (int index = lane; index < kFineTile * kHy64SubTiles * 2;
             index += kStagingThreads) {
          const int m = index >> 5;
          const int sub = index & 31;
          const int j = sub >> 1;
          const int half = sub & 1;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              slot + j * 2048 + half * 1024 + m * 16,
              args.lm_head_bf16 + weight_base
                  + static_cast<int64_t>(m) * kHidden
                  + k_base + j * kUmmaK + half * 8);
        }
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    // MMA crew: 8 (M128) or 16 (M64) chained K=16 UMMAs per slot into the
    // item's OWN accumulator (interiors alternate 0/1, edges own 2).
    for (int c = 0; c < total_slots; ++c) {
      lm_hybrid_walk_next(walk);
      const int s = c % kHybridStages;
      const int acc = lm_hybrid_acc(walk.kind, walk.item);
      const int round = lm_hybrid_acc_round(walk.kind, walk.item);
      wait_barrier(storage.stage_full[s], (c / kHybridStages) & 1);
      if (walk.slot_in_item == 0 && round >= 1) {
        wait_barrier(storage.accumulator_free[acc], (round - 1) & 1);
      }
      const uint32_t tmem_acc = storage.tmem_base_ptr + acc * 8;
      if (walk.kind == 1) {
        CUTE_UNROLL
        for (int j = 0; j < kHy128SubTiles; ++j) {
          LmUmmaAtom::fma(
              a_descriptor_128[s] + static_cast<uint64_t>(j) * 256,
              b_descriptor_base + static_cast<uint64_t>(
                  walk.slot_in_item * kHy128SubTiles + j) * 16,
              tmem_acc,
              (walk.slot_in_item == 0 && j == 0) ? 0u : 1u,
              idesc_128);
        }
      } else {
        CUTE_UNROLL
        for (int j = 0; j < kHy64SubTiles; ++j) {
          LmUmmaM64Atom::fma(
              a_descriptor_64[s] + static_cast<uint64_t>(j) * 128,
              b_descriptor_base + static_cast<uint64_t>(
                  walk.slot_in_item * kHy64SubTiles + j) * 16,
              tmem_acc,
              (walk.slot_in_item == 0 && j == 0) ? 0u : 1u,
              idesc_64);
        }
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (walk.item_last) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[acc]);
      }
    }
  } else {
    // Drain crew (warps 0-3): replays the walk and drains each item at its
    // FINAL slot's stream position (completion order), so the MMA crew's
    // mid-stream accumulator_free waits are always eventually served. Copy
    // atom and TMEM view are chosen by the item's kind/accumulator.
    auto copy_128 = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, accumulator_128);
    auto copy_64 = make_tmem_copy(
        SM100_TMEM_LOAD_16dp32b1x{}, accumulator_64);
    auto thread_copy_128 = copy_128.get_slice(threadIdx.x);
    auto thread_copy_64 = copy_64.get_slice(threadIdx.x);
    auto view_128_a = accumulator_128;
    view_128_a.data() = storage.tmem_base_ptr;
    auto view_128_b = accumulator_128;
    view_128_b.data() = storage.tmem_base_ptr + 8;
    auto view_64 = accumulator_64;
    view_64.data() = storage.tmem_base_ptr + 16;
    auto thread_tmem_128_a = thread_copy_128.partition_S(view_128_a);
    auto thread_tmem_128_b = thread_copy_128.partition_S(view_128_b);
    auto thread_tmem_64 = thread_copy_64.partition_S(view_64);
    auto layout_128 = make_layout(
        make_shape(Int<kOutputTile>{}, Int<8>{}),
        make_stride(Int<1>{}, Int<kOutputTile>{}));
    auto layout_64 = make_layout(
        make_shape(Int<kFineTile>{}, Int<8>{}),
        make_stride(Int<1>{}, Int<kFineTile>{}));
    auto partial_low_128 = thread_copy_128.partition_D(
        cta_mma_128.partition_C(make_tensor(
            make_smem_ptr(storage.partial[0]), layout_128)));
    auto partial_high_128 = thread_copy_128.partition_D(
        cta_mma_128.partition_C(make_tensor(
            make_smem_ptr(storage.partial[1]), layout_128)));
    auto partial_low_64 = thread_copy_64.partition_D(
        cta_mma_64.partition_C(make_tensor(
            make_smem_ptr(storage.partial[0]), layout_64)));
    auto partial_high_64 = thread_copy_64.partition_D(
        cta_mma_64.partition_C(make_tensor(
            make_smem_ptr(storage.partial[1]), layout_64)));
    auto register_128 = make_tensor<float>(shape(partial_low_128));
    auto register_64 = make_tensor<float>(shape(partial_low_64));

    int drained = 0;
    for (int c = 0; c < total_slots && drained < total_items; ++c) {
      lm_hybrid_walk_next(walk);
      if (!walk.item_last) {
        continue;
      }
      const int acc = lm_hybrid_acc(walk.kind, walk.item);
      const int round = lm_hybrid_acc_round(walk.kind, walk.item);
      const bool is_m128 = walk.kind == 1;
      const int parity = drained & 1;
      ++drained;
      wait_barrier(storage.accumulator_full[acc], round & 1);
      if (is_m128) {
        if (acc == 0) {
          copy(copy_128, thread_tmem_128_a, register_128);
        } else {
          copy(copy_128, thread_tmem_128_b, register_128);
        }
      } else {
        copy(copy_64, thread_tmem_64, register_64);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[acc]);
      if (is_m128) {
        if (parity == 0) {
          copy(register_128, partial_low_128);
        } else {
          copy(register_128, partial_high_128);
        }
      } else {
        if (parity == 0) {
          copy(register_64, partial_low_64);
        } else {
          copy(register_64, partial_high_64);
        }
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const int rows = is_m128 ? kOutputTile : kFineTile;
      const int64_t tile_base = static_cast<int64_t>(walk.unit) * kFineTile;
      if (static_cast<int>(threadIdx.x) < rows) {
        CUTE_UNROLL
        for (int row = 0; row < kDraftBlock; ++row) {
          args.base_logits[
              static_cast<int64_t>(row) * kVocab + tile_base + threadIdx.x] =
              storage.partial[parity][
                  row * rows + static_cast<int>(threadIdx.x)];
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

// PROBE body #6 (batch-amortization campaign, step 1). BENCH ONLY: gated on
// -DDSPARK_LM_BATCH_PROBE so the production translation unit is unchanged.
//
// Question it answers: does widening the UMMA N-mode (activation rows) leave
// the band's runtime FLAT, i.e. does the 1.06 GB lm_head stream amortize
// across a serving batch? execute_umma_bf16 is hard-wired to N=8 because
// LmUmmaSharedStorage::b_all holds ALL 256 K-chunks of the activation
// operand (kUmmaChunks * 8 * kUmmaK * 2 = 64 KB); at N=40 that array alone
// would be 320 KB.
//
// The fix here is to stop holding the activation operand resident and window
// it INSIDE THE EXISTING WEIGHT RING: each 128x64 A slot now carries the
// four K=16 activation chunks that cover exactly the same K span. That adds
// no barrier, no crew, and no new deadlock surface — stage_full[s] and
// slot_free[s] already fence precisely the bytes the MMA of slot s reads.
// SMEM becomes 65,536 (a_ring) + 512*N (b_ring) + 1024*N (partial), so
// N <= 48 fits kSharedBytesBudget with the production 4-deep ring intact.
//
// Cost of windowing: the activation window is re-read once per vocab tile
// instead of once per invocation, i.e. items * kHidden * N * 2 bytes extra
// (8.27 MB * N over the 1010-item stream). That working set is N * 8 KB, so
// it is L2-resident and adds no DRAM traffic — but it does add cp.async
// packets (128*N bytes per 16 KB weight slot).
//
// Numerics: the K chain per output element is IDENTICAL to
// execute_umma_bf16 — 256 chained K=16 tcgen05 MMAs in ascending K with the
// FP32 accumulator in TMEM (scaleC=0 opens each item, 1 chains). Widening N
// only adds independent output columns, so rows 0-4 must be BITWISE equal to
// the frozen body. The bench gates on exactly that.
//
// The B SMEM layout for arbitrary N was verified host-side against cute's
// layout algebra (nvidia-cutlass 4.2.0.0): layout<0>(SmemLayoutB) is
// (N,(8,2)):(8,(1,8N)), so element (row, k) sits at row*8 + k%8 + (k/8)*8N,
// one K=16 chunk is N*16 elements (32N bytes), and each 8-element K half of
// a row is 16 contiguous bytes — one cp.async packet. Checked for
// N = 8, 16, 32, 40, 48; N=8 reproduces the frozen body's
// chunk*256 + half*128 + row*16 byte formula exactly.
//
// PRODUCTION USE (R3 batched serving): the same body backs head.lm_row_0 in
// any DSPARK_V4_BATCH > 1 build, instantiated at kRows = round-up-to-8 of
// 5 * batch with kRealRows = 5 * batch. tcgen05's f16 N mode must be a
// multiple of 8, so batch 2 (10 rows) and batch 4 (20 rows) run a padded
// atom; the pad columns re-read the last real row (never out of bounds) and
// are simply not stored, exactly as the frozen N=8 body pads 5 -> 8.
#if defined(DSPARK_LM_BATCH_PROBE) || DSPARK_V4_BATCH > 1
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

template <int kRows>
struct LmBatchTraits {
  using Atom = cute::SM100_MMA_F16BF16_SS<
      cute::bfloat16_t, cute::bfloat16_t, float, kOutputTile, kRows,
      cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using TiledMma = decltype(cute::make_tiled_mma(Atom{}));
  using ShapeA = decltype(cute::partition_shape_A(
      TiledMma{},
      cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kUmmaK>{})));
  using ShapeB = decltype(cute::partition_shape_B(
      TiledMma{},
      cute::make_shape(cute::Int<kRows>{}, cute::Int<kUmmaK>{})));
  using SmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, ShapeA{}));
  using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{}, ShapeB{}));
  using BChunkLayout = decltype(cute::layout<0>(SmemLayoutB{}));
  static constexpr int kTmemColumns =
      kRows <= 32 ? 32 : (kRows <= 64 ? 64 : 128);
};

template <int kRows>
struct LmBatchSharedStorage {
  alignas(128) __nv_bfloat16 a_ring[kUmmaStages][kUmmaSlotElements];
  alignas(128) __nv_bfloat16
      b_ring[kUmmaStages][kUmmaSubTiles * kRows * kUmmaK];
  alignas(16) float partial[2][kOutputTile * kRows];
  alignas(16) cute::uint64_t stage_full[kUmmaStages];
  alignas(16) cute::uint64_t slot_free[kUmmaStages];
  alignas(16) cute::uint64_t accumulator_full[1];
  alignas(16) cute::uint64_t accumulator_free[1];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <int kRows, int kRealRows = kRows>
__device__ inline void execute_umma_bf16_batch(const Args& args) {
  using namespace cute;
  using Traits = LmBatchTraits<kRows>;
  using Atom = typename Traits::Atom;
  using Storage = LmBatchSharedStorage<kRows>;
  static_assert(kRows % 8 == 0 && kRows >= 8 && kRows <= 256,
                "tcgen05 f16 N-mode must be a multiple of 8 in [8, 256]");
  static_assert(kRealRows >= kDraftBlock && kRealRows <= kRows,
                "the batch body must cover the real draft rows and no more");
  static_assert(sizeof(Storage) <= kSharedBytesBudget,
                "windowed activation ring must fit dynamic shared memory");
  static_assert(cute::cosize_v<typename Traits::BChunkLayout> ==
                    kRows * kUmmaK,
                "one K16 activation chunk is exactly kRows*16 elements");
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kTmemColumns = Traits::kTmemColumns;
  constexpr int kBChunkElements = kRows * kUmmaK;
  // Descriptor addresses advance in 16-byte units: one chunk is 32*kRows B.
  constexpr uint64_t kBChunkUnits = static_cast<uint64_t>(kRows) * 2;

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
  const int total_slots = item_count * kUmmaSlotsPerItem;

  if (warp >= 5) {
    // Staging crew: the frozen body's 1024 weight packets per 128x64 slot,
    // plus 2*kRows activation packets per K16 chunk covering the same span.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      if (c >= kUmmaStages) {
        wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
      }
      const int item = static_cast<int>(args.begin) + c / kUmmaSlotsPerItem;
      const int64_t weight_base =
          static_cast<int64_t>(item) * kOutputTile * kHidden;
      const int k_base = (c % kUmmaSlotsPerItem) * kUmmaSlotK;
      unsigned char* slot = reinterpret_cast<unsigned char*>(storage.a_ring[s]);
      // Same bank-conflict-free lane->packet map as execute_umma_bf16.
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
            args.lm_head_bf16 + weight_base
                + static_cast<int64_t>(m) * kHidden
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
            args.normalized + static_cast<int64_t>(source_row) * kHidden
                + k_base + j * kUmmaK + half * 8);
      }
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      const int slot_in_item = c % kUmmaSlotsPerItem;
      const int item_local = c / kUmmaSlotsPerItem;
      wait_barrier(storage.stage_full[s], (c / kUmmaStages) & 1);
      if (slot_in_item == 0 && item_local > 0) {
        wait_barrier(storage.accumulator_free[0], (item_local - 1) & 1);
      }
      CUTE_UNROLL
      for (int j = 0; j < kUmmaSubTiles; ++j) {
        Atom::fma(
            a_descriptor[s] + static_cast<uint64_t>(j) * 256,
            b_descriptor[s] + static_cast<uint64_t>(j) * kBChunkUnits,
            storage.tmem_base_ptr,
            (slot_in_item == 0 && j == 0) ? 0u : 1u,
            instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kUmmaSlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[0]);
      }
    }
  } else {
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
      const int64_t tile_base =
          (static_cast<int64_t>(args.begin) + i) * kOutputTile;
      CUTE_UNROLL
      for (int row = 0; row < kRealRows; ++row) {
        args.base_logits[
            static_cast<int64_t>(row) * kVocab + tile_base + threadIdx.x] =
            storage.partial[parity][
                row * kOutputTile + static_cast<int>(threadIdx.x)];
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
#endif  // DSPARK_LM_BATCH_PROBE || DSPARK_V4_BATCH > 1

}  // namespace dspark_lm
