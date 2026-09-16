// Self-contained routed FP4 W13 phase bodies for the DeepSeek-V4 DSpark
// megakernel. This header is compiled by two translation units:
//   1. dspark_v4_kernel.cu (the production persistent kernel), and
//   2. dspark_w13_bench.cu (the fast phase-iteration microbenchmark).
// Keeping one implementation guarantees the microbenchmark measures exactly
// the code the megakernel executes. Numerical semantics are frozen: the
// scalar body is bit-exact against the official DeepSeek checkpoint oracle
// (validation ledger #56/#79), and the TCGen05 body must match it bitwise.
#pragma once

#include "dspark_tmem.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#if defined(DSPARK_W13_TMA_CANDIDATE)
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#endif

#include "dspark_batch.h"
#include "dspark_v4_tp2_ablate.cuh"

#ifndef DSPARK_ROUTED_COMPACT_TILES
#define DSPARK_ROUTED_COMPACT_TILES 0
#endif

#ifndef DSPARK_ROUTED_REUSE_TMEM
#define DSPARK_ROUTED_REUSE_TMEM 0
#endif

#ifndef DSPARK_ROUTED_GROUP_MASKS
#define DSPARK_ROUTED_GROUP_MASKS 0
#endif

#ifndef DSPARK_ROUTED_SF_RING
#define DSPARK_ROUTED_SF_RING 0
#endif

namespace dspark_w13 {

constexpr int kHidden = 4096;
constexpr int kQuantBlock = 128;
constexpr int kIntermediate = 2048;
constexpr int kActivatedExperts = 6;
// Draft block rows. Batched serving (R3) makes this batch * 5: the routed
// phases already carried a `batch *` work-unit factor, so the route table is
// simply longer (30 rows at batch 1, 240 at batch 8) and every route_row is
// still a flat (batch element, draft row, top-k) index into the
// (batch, block, topk, ...) workspace regions.
constexpr int kDraftRows = dspark_batch::kRows;
// SMEM route cache slots; batch 1 keeps the frozen 32-entry array.
constexpr int kRouteCacheSlots =
    kDraftRows * kActivatedExperts <= 32 ? 32 : kDraftRows * kActivatedExperts;
constexpr int kFp4Block = 32;
constexpr int kOutputTile = 128;
constexpr int kCombinedTiles = 2 * kIntermediate / kOutputTile;
constexpr int kWeightBlocks = kHidden / kFp4Block;
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;

// R11 TILE WIDTH. SGLang's stock routed W13 runs a 256-column output tile
// (grid 16 x 30 over 4096 combined columns) against our 128 (grid 32 x 30),
// and the R1-leg-8 attribution says why that matters here: with EVERY byte
// of staging traffic masked off the DG body still costs 64.7% of its round,
// so the pacer is the PER-ROUND SERIAL SKELETON (barrier cadence + the
// dependent N=8 UMMA chain + the per-item drain), not bandwidth. Fusing
// kDgTilePairs consecutive items into one 256-row pass halves the number of
// ring rounds per weight byte and halves the activation/SFB request count
// per byte, without moving one byte more or less of weight traffic.
//
// WHY PAIRS AND NOT A WIDER kOutputTile: pairing leaves the item space
// (960 units), the claim chunk, the route-table decomposition, the generated
// headers and the contract kernel completely untouched. Item 2k and item
// 2k+1 of the same claim always share a route row (kCombinedTiles and
// kW2OutputTiles are both 32, so item/32 is pair-invariant) and always carry
// adjacent output tiles of the same branch (tile/16 is pair-invariant
// because 2k+1 is never a multiple of 16), so the fused pass is exactly the
// two items' work with each output element's K chain, chunk order and
// accumulator untouched: BITWISE by construction.
//
// The pair pass needs two A ring planes per stage, so kDgStages must fall
// (see the sweep in the R11 ledger row); the budget is the build-time
// DSPARK_V4_DYNAMIC_SMEM_BYTES macro. Default 1 reproduces the shipped body
// byte for byte, including sizeof(DgSharedStorage) and every TMEM column.
#ifndef DSPARK_W13_DG_TILE_PAIRS
#define DSPARK_W13_DG_TILE_PAIRS 1
#endif
constexpr int kDgTilePairs = DSPARK_W13_DG_TILE_PAIRS;
// Item 2k/2k+1 (and, at 4, 4k..4k+3) share a route row and carry adjacent
// output tiles of one branch because kCombinedTiles and kW2OutputTiles are
// both 32 and the branch split is at 16: item/32 and (item%32)/16 are both
// invariant across an aligned group of 2 or 4. 8 would straddle the branch
// boundary, so the fused tile stops at 4 planes (512 output columns).
static_assert(kDgTilePairs == 1 || kDgTilePairs == 2 || kDgTilePairs == 4,
              "the fused routed tile is one, two or four 128-row planes");
constexpr int kDgTileRows = kDgTilePairs * kOutputTile;

// Flat argument view of one routed-W13 task: [begin, end) item indices over
// the (row, top, branch, output-tile) decomposition, plus the weight arena
// slots resolved by the caller for the current layer.
struct Args {
  const cutlass::float_e4m3_t* input;
  const uint8_t* input_scales;
  const int32_t* indices;
  __nv_bfloat16* routed_w13;
#ifdef DSPARK_ROUTED_FUSED_SWIGLU
  const float* route_weights;
  __nv_bfloat16* swiglu;
  cutlass::float_e4m3_t* swiglu_quantized;
  uint8_t* swiglu_scales;
#ifdef DSPARK_ROUTED_GROUP_READY
  uint32_t* group_ready;
#endif
#endif
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  int weight_base;
  int expert_weight_slots;
  int w1_slot;
  int w3_slot;
  int w1_scale_slot;
  int w3_scale_slot;
  uint32_t begin;
  uint32_t end;
#ifdef DSPARK_V4_FINE_GRAINED_OVERLAP
  // Candidate-only logical iterator: items before item_gap_begin are
  // contiguous, then later logical items jump by item_gap. This pairs the
  // matching gate/up half-bands without paying a second DG ring setup.
  uint32_t item_gap_begin;
  uint32_t item_gap;
#endif
};

template <class ArgsType>
__device__ __forceinline__ uint32_t dg_physical_item(
    const ArgsType&,
    uint32_t item) {
  return item;
}

#ifdef DSPARK_V4_FINE_GRAINED_OVERLAP
__device__ __forceinline__ uint32_t dg_physical_item(
    const Args& args,
    uint32_t item) {
  return item >= args.item_gap_begin ? item + args.item_gap : item;
}
#endif

// Bit-exact E8M0 and E2M1 decodes, verbatim from the retained #71/#72
// implementations in dspark_v4_kernel.cu.
__device__ __forceinline__ float decode_e8m0(uint8_t bits) {
  const uint32_t exponent = bits;
  const uint32_t float_bits =
      exponent == 0 ? 0x00400000U : exponent << 23;
  return __uint_as_float(float_bits);
}

__device__ __forceinline__ float decode_e2m1_code(uint32_t code) {
  const uint32_t magnitude = code & 0x07;
  uint32_t bits = 0;
  if (magnitude != 0) {
    bits = magnitude == 1
        ? 0x3f000000U
        : 0x3f000000U + (magnitude << 22);
  }
  bits |= (code & 0x08) << 28;
  return __uint_as_float(bits);
}

struct ItemPlan {
  const uint8_t* weight;
  const uint8_t* weight_scales;
  int row;
  int top;
  int branch;
  int output_base;
};

__device__ __forceinline__ ItemPlan item_plan(const Args& args, uint32_t item) {
  ItemPlan plan;
  const int route_row = static_cast<int>(item) / kCombinedTiles;
  const int combined_tile = static_cast<int>(item) % kCombinedTiles;
  plan.row = route_row / kActivatedExperts;
  plan.top = route_row % kActivatedExperts;
  plan.branch = combined_tile / (kIntermediate / kOutputTile);
  plan.output_base =
      (combined_tile % (kIntermediate / kOutputTile)) * kOutputTile;
  const int expert =
      args.indices[plan.row * kActivatedExperts + plan.top];
  const int weight_slot =
      plan.branch == 0 ? args.w1_slot : args.w3_slot;
  const int scale_slot =
      plan.branch == 0 ? args.w1_scale_slot : args.w3_scale_slot;
  const int table_base =
      args.weight_base + expert * args.expert_weight_slots;
  plan.weight = args.weight_arena + args.weight_offsets[table_base + weight_slot];
  plan.weight_scales =
      args.weight_arena + args.weight_offsets[table_base + scale_slot];
  return plan;
}

// Scalar body: exact paired-byte FP4 consumption with the frozen blockwise
// FP32 accumulation order (retained #79/#84 semantics).
__device__ inline void execute_scalar(const Args& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    if (threadIdx.x >= kOutputTile) {
      continue;
    }
    const ItemPlan plan = item_plan(args, item);
    const int output = plan.output_base + static_cast<int>(threadIdx.x);
    float result = 0.0f;
    for (int block = 0; block < kWeightBlocks; ++block) {
      float inner = 0.0f;
      const int input_base = plan.row * kHidden + block * kFp4Block;
      const int logical_weight_base = output * kHidden + block * kFp4Block;
#pragma unroll
      for (int offset = 0; offset < kFp4Block; offset += 2) {
        const uint8_t packed = plan.weight[(logical_weight_base + offset) >> 1];
        inner = fmaf(
            static_cast<float>(args.input[input_base + offset]),
            decode_e2m1_code(packed & 0x0f),
            inner);
        inner = fmaf(
            static_cast<float>(args.input[input_base + offset + 1]),
            decode_e2m1_code(packed >> 4),
            inner);
      }
      const float input_scale = decode_e8m0(
          args.input_scales[plan.row * (kHidden / kQuantBlock) + block / 4]);
      const float weight_scale =
          decode_e8m0(plan.weight_scales[output * kWeightBlocks + block]);
      result = fmaf(inner, input_scale * weight_scale, result);
    }
    args.routed_w13[
        ((plan.row * kActivatedExperts + plan.top) * 2 + plan.branch)
            * kIntermediate
        + output] = __float2bfloat16_rn(result);
  }
}

#if defined(DSPARK_W13_CREW_PROBE)
// Bench-only per-crew issue-occupancy probe. Every crew accumulates clock64
// deltas in registers and one lane per warp adds them once, after the crew's
// last round, so the counters cost no traffic inside the ring. Absolute
// numbers are perturbed by the clock reads; the SHARES are what this exists
// to measure.
//   crew 0 staging: 0 total, 1 stall(stage_free), 2 weights, 3 activations,
//                   4 scale gathers, 5 in-loop drain, 6 warps, 7 rounds
//   crew 1 MMA:     0 total, 1 stall(stage_full), 2 stall(accumulator_free),
//                   3 UTCCP+UMMA issue, 6 warps, 7 rounds
//   crew 2 drain:   0 total, 1 stall(accumulator_full), 2 TMEM->SMEM,
//                   3 global stores, 6 warps, 7 items
__device__ unsigned long long dg_crew_probe[3][8];
__device__ __forceinline__ void dg_probe_add(
    int crew, int slot, unsigned long long value) {
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&dg_crew_probe[crew][slot], value);
  }
}
#define DG_PROBE_NOW() clock64()
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

using TcgenWeightType = cutlass::detail::float_e2m1_unpacksmem_t;
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
// kind::f8f6f4 requires sub-byte operands to occupy one byte-slot per element
// in shared memory (16 packed E2M1 nibbles live in the first 8 bytes of each
// 16-byte group; the trailing 8 bytes are ignored padding), so the layouts
// use byte-granular interleaved atoms exactly like the CUTLASS SM100
// collective's SmemAllocType=uint8_t path.
using TcgenSmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<uint8_t>{},
    TcgenMmaShapeA{}));
using TcgenSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_INTER_Atom<TcgenInputType>{},
    TcgenMmaShapeB{}));

// Warp-specialized pipeline: warps 5-7 stage packed FP4 weights and the FP8
// input slice through a kStages-deep cp.async ring, warp 4 streams one K=32
// MMA per weight-scale block into kStages TMEM accumulator slots (8 columns
// each of the 32 allocated), and warps 0-3 drain each partial through
// registers into the original outer FP32 fmaf scale chain.
//
// Every barrier is per-slot and completes once per ring round, so each
// crew's parity wait is at most one completion behind its barrier by
// construction. (A 2-slot accumulator under a 4-deep ring deadlocks: the
// staging crew's wait can fall two completions behind and mbarrier parity
// waits alias after an even number of completions.)
constexpr int kStages = 4;

struct TcgenSharedStorage {
  alignas(128)
      cute::ArrayEngine<uint8_t, cute::cosize_v<TcgenSmemLayoutA>> a[kStages];
  alignas(128)
      cute::ArrayEngine<
          TcgenInputType, cute::cosize_v<TcgenSmemLayoutB>> b[kStages];
  alignas(16) float partial[2][128 * 8];
  alignas(16) cute::uint64_t stage_full[kStages];
  alignas(16) cute::uint64_t accumulator_full[kStages];
  alignas(16) cute::uint64_t accumulator_free[kStages];
  alignas(16) cute::uint32_t tmem_base_ptr;
  // SMEM-cached route table for the grouped relaxed bodies (unused by the
  // contract-shape pipeline; adding the field only grows the struct).
  alignas(16) int32_t route_indices[kRouteCacheSlots];
};

// Common per-item plan view consumed by the FP4/FP8 GEMV pipeline. All
// pointers are item-resolved bases; the pipeline adds output/block offsets.
struct PipelinePlan {
  const uint8_t* weight;
  const uint8_t* weight_scales;
  const cutlass::float_e4m3_t* input_row;
  const uint8_t* input_scales_row;
  __nv_bfloat16* store_row;
  int output_base;
};

template <class Policy>
__device__ inline void execute_fp4_pipeline(
    const typename Policy::Args& args) {
  using namespace cute;
  constexpr int kKBlocks = Policy::kKBlocks;
  constexpr int kKWidth = kKBlocks * kFp4Block;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<TcgenSmemLayoutB>);
  static_assert(sizeof(TcgenSharedStorage) <= kSharedBytesBudget);

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
        cutlass::arch::ClusterBarrier, kStages>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Rows 1..7 of every input stage stay zero for the whole task; row 0 is
  // refilled per block by cp.async. These generic-proxy stores must be made
  // visible to the MMA's async proxy before the first tcgen05.mma issues.
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
  // cute's degenerate single-atom descriptor-tensor gemm dispatch trips NVCC,
  // so the per-stage shared-memory descriptors and the instruction descriptor
  // are materialized directly, replicating mma_unpack for SM100_MMA_F8F6F4_SS.
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
      static_cast<int>(args.end - args.begin) * kKBlocks;

  if (warp >= 5) {
    // Staging crew. Stage s is reusable once the MMA that consumed it
    // (block b - kStages) has completed, observed through its
    // accumulator-full barrier.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      if (b >= kStages) {
        // Slot s was consumed by the MMA of block b - kStages, whose
        // accumulator-full completion is exactly one ring round behind.
        wait_barrier(storage.accumulator_full[s], (b / kStages - 1) & 1);
      }
      const PipelinePlan plan = Policy::plan(
          args, args.begin + static_cast<uint32_t>(b / kKBlocks));
      const int block = b % kKBlocks;
      // Each output row contributes 32 E2M1 elements = 16 packed global
      // bytes per K block; each packed 8-byte half lands at the base of a
      // 16-byte group (physical byte offset output*16 + group*2048).
      uint8_t* stage_a = reinterpret_cast<uint8_t*>(storage.a[s].begin());
      for (int copy_index = lane; copy_index < kOutputTile * 2;
           copy_index += kStagingThreads) {
        const int output = copy_index >> 1;
        const int group = copy_index & 1;
        const uint8_t* source = plan.weight
            + (static_cast<int64_t>(plan.output_base + output) * kKWidth
               + block * kFp4Block) / 2
            + group * 8;
        cutlass::arch::cp_async<8, cutlass::arch::CacheOperation::Always>(
            stage_a + output * 16 + group * 2048, source);
      }
      if (lane < 2) {
        // Input row 0: two 16-byte chunks at atom bases 0 and 128.
        const uint8_t* source =
            reinterpret_cast<const uint8_t*>(plan.input_row)
            + block * kFp4Block + lane * 16;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
            reinterpret_cast<uint8_t*>(storage.b[s].begin()) + lane * 128,
            source);
      }
      // noinc: the barrier's expected count is the 96 staging threads; the
      // default form pre-increments the count and would never complete.
      cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
    }
  } else if (warp == 4) {
    // MMA crew. scaleC = 0 overwrites the slot, giving one independent
    // tensor-core partial per 32-element weight-scale block. fma and
    // umma_arrive are elect_one_sync-guarded internally.
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      wait_barrier(storage.stage_full[s], (b / kStages) & 1);
      if (b >= kStages) {
        wait_barrier(storage.accumulator_free[s], (b / kStages - 1) & 1);
      }
      SM100_MMA_F8F6F4_SS::fma(
          weight_descriptor[s],
          input_descriptor[s],
          storage.tmem_base_ptr + s * 8,
          0u,
          instruction_descriptor);
      cutlass::arch::umma_arrive(&storage.accumulator_full[s]);
    }
  } else {
    // Accumulator crew (warps 0-3): drain each TMEM partial and apply the
    // original outer FP32 fmaf activation/weight E8M0 scale chain.
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int s) {
      auto slot = tmem_accumulator;
      slot.data() = storage.tmem_base_ptr + s * 8;
      return thread_copy.partition_S(slot);
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

    PipelinePlan plan = {};
    float result = 0.0f;
    for (int b = 0; b < total_blocks; ++b) {
      const int s = b % kStages;
      const int parity = b % 2;
      const int block = b % kKBlocks;
      if (block == 0) {
        plan = Policy::plan(
            args, args.begin + static_cast<uint32_t>(b / kKBlocks));
        result = 0.0f;
      }
      wait_barrier(storage.accumulator_full[s], (b / kStages) & 1);
      if (s == 0) {
        copy(tmem_to_register, thread_tmem_0, register_accumulator);
      } else if (s == 1) {
        copy(tmem_to_register, thread_tmem_1, register_accumulator);
      } else if (s == 2) {
        copy(tmem_to_register, thread_tmem_2, register_accumulator);
      } else {
        copy(tmem_to_register, thread_tmem_3, register_accumulator);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[s]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const float input_scale =
          decode_e8m0(plan.input_scales_row[block / 4]);
      const float weight_scale = decode_e8m0(
          plan.weight_scales[
              (plan.output_base + static_cast<int>(threadIdx.x))
                  * kKBlocks + block]);
      result = fmaf(
          storage.partial[parity][threadIdx.x],
          input_scale * weight_scale,
          result);
      if (block == kKBlocks - 1) {
        plan.store_row[plan.output_base + threadIdx.x] =
            __float2bfloat16_rn(result);
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}
struct W13Policy {
  using Args = dspark_w13::Args;
  static constexpr int kKBlocks = kHidden / kFp4Block;

  static __device__ __forceinline__ PipelinePlan plan(
      const Args& args, uint32_t item) {
    const ItemPlan resolved = item_plan(args, item);
    PipelinePlan view;
    view.weight = resolved.weight;
    view.weight_scales = resolved.weight_scales;
    view.input_row = args.input + resolved.row * kHidden;
    view.input_scales_row =
        args.input_scales + resolved.row * (kHidden / kQuantBlock);
    view.store_row = args.routed_w13
        + ((resolved.row * kActivatedExperts + resolved.top) * 2
           + resolved.branch) * kIntermediate;
    view.output_base = resolved.output_base;
    return view;
  }
};

__device__ inline void execute_tcgen(const Args& args) {
  execute_fp4_pipeline<W13Policy>(args);
}

// ---------------------------------------------------------------------------
// Grouped (expert-deduplicated) pipeline — relaxed-path candidate.
//
// The production N=8 UMMA carries ONE live activation column, so route_rows
// that select the same expert re-stream that expert's 128-wide weight tiles
// once each. This body batches up to kMaxGroup route_rows sharing an expert
// into the N dimension of one UMMA chain and streams the weights once.
//
// Item space and count are UNCHANGED from the contract decomposition: an
// item whose route_row is the k-th occurrence of its expert with
// k % kMaxGroup != 0 is a no-op (it exits before touching barriers or TMEM),
// and the chunk-leader occurrence (k % kMaxGroup == 0) computes the outputs
// of the next up-to-kMaxGroup occurrences. The leader predicate is a pure
// function of args.indices, so every CTA resolves the same grouping with no
// extra buffer, no host pass, and no schedule change.
//
// Numerics: an MMA output element depends only on its own A row and B
// column (the iteration-4 attention precedent), and each output element's
// outer FP32 fmaf scale chain runs in the same K-block order as the N=1
// body — the only reordering is ACROSS N columns, which never mixes into a
// chain. Outputs are therefore BITWISE identical to execute_tcgen /
// execute_w2_tcgen and hence to the scalar oracle.

constexpr int kMaxGroup = 8;
// Members are packed into one uint64_t so the context never contains a
// dynamically indexed array: NVCC would demote such a struct to local memory
// and every per-block read of ctx fields in the staging loop would pay an L1
// round-trip. 5 bits covers a 30-row batch-1 table; batched serving needs 8,
// which still packs kMaxGroup = 8 members into exactly 64 bits and caps the
// table at 256 rows == batch 8.
constexpr int kMemberBits =
    kDraftRows * kActivatedExperts <= 32 ? 5 : 8;
static_assert(kMemberBits * kMaxGroup <= 64);
static_assert(kDraftRows * kActivatedExperts <= (1 << kMemberBits),
              "the routed route-table encoding caps the batch at 8");

__device__ __forceinline__ int group_member(uint64_t members, int m) {
  return static_cast<int>(
      (members >> (kMemberBits * m)) & ((1u << kMemberBits) - 1));
}

struct GroupContext {
  const uint8_t* weight;
  const uint8_t* weight_scales;
  int output_base;
  int branch;              // W13 branch select; always 0 for W2.
  int member_count;        // 0 => this item is a non-leader duplicate.
  uint64_t members;        // route_rows riding B columns 0..count-1.
};

// Leader predicate over the SMEM-cached route table: true iff route_row is
// the k-th occurrence of its expert with k % kMaxGroup == 0.
__device__ __forceinline__ bool chunk_leader(
    const int32_t* route_indices, int route_row) {
  const int expert = route_indices[route_row];
  int occurrences_before = 0;
  for (int r = 0; r < route_row; ++r) {
    occurrences_before += (route_indices[r] == expert) ? 1 : 0;
  }
  return occurrences_before % kMaxGroup == 0;
}

// chunk_leader AND this rank's expert ownership. Every crew in the pipeline
// (producer, MMA, accumulator) must skip exactly the same item set or the
// stage barriers desynchronize, so the predicate lives in one place.
__device__ __forceinline__ bool chunk_selected(
    const int32_t* route_indices, int route_row) {
  if (!chunk_leader(route_indices, route_row)) {
    return false;
  }
#if defined(DSPARK_V4_TP2_ABLATE)
  if (!dspark_tp2_ablate::owns_expert(route_indices[route_row])) {
    return false;
  }
#endif
  return true;
}

// Gather the chunk members starting at a leader route_row into a packed
// word; returns the member count (0 if route_row is not a leader).
__device__ __forceinline__ int chunk_members(
    const int32_t* route_indices,
    int route_row,
    int route_rows,
    uint64_t* members_out) {
  if (!chunk_leader(route_indices, route_row)) {
    return 0;
  }
  const int expert = route_indices[route_row];
  uint64_t members = 0;
  int count = 0;
  for (int r = route_row; r < route_rows; ++r) {
    if (route_indices[r] == expert && count < kMaxGroup) {
      members |= static_cast<uint64_t>(r) << (kMemberBits * count);
      ++count;
    }
  }
  // Pad unused slots with member 0 (the leader) so unconditional unrolled
  // reads stay in a defined range.
  for (int m = count; m < kMaxGroup; ++m) {
    members |= static_cast<uint64_t>(group_member(members, 0))
        << (kMemberBits * m);
  }
  *members_out = members;
  return count;
}

// Per-crew memo of the last route row's grouping. chunk_members is an
// O(route_rows) scan and route_row(item) = item / kCombinedTiles, so 32
// consecutive items share one route row: without the memo the scan is
// O(route_rows) PER ITEM PER CREW, which is O(batch^2) once the table grows
// from 30 rows to 240. Grouping depends only on route_row; the policy's
// pointer/tile resolution still runs per item, so results are unchanged.
struct GroupCache {
  int route_row = -1;
  int member_count = 0;
  uint64_t members = 0;
};

// Full per-item context from the SMEM route cache: pipeline-side grouping
// plus the policy's pointer/tile resolution.
template <class Policy>
__device__ __forceinline__ GroupContext group_plan(
    const typename Policy::Args& args,
    uint32_t item,
    const int32_t* route_indices,
    GroupCache& cache) {
  static_assert(Policy::kRouteRows <= (1 << kMemberBits));
  GroupContext ctx;
  const int route_row = Policy::route_row(item);
  if (route_row != cache.route_row) {
    cache.route_row = route_row;
    cache.member_count = chunk_members(
        route_indices, route_row, Policy::kRouteRows, &cache.members);
  }
  ctx.member_count = cache.member_count;
  ctx.members = cache.members;
#if defined(DSPARK_V4_TP2_ABLATE)
  // Expert-parallel ownership, applied IN ADDITION to the leader predicate
  // and never instead of it: member_count is already 0 for a non-leader, and
  // an expert-keyed filter cannot orphan a member from its leader because
  // every row of a group carries the same expert.
  if (ctx.member_count != 0
      && !dspark_tp2_ablate::owns_expert(route_indices[route_row])) {
    ctx.member_count = 0;
  }
#endif
  if (ctx.member_count != 0) {
    Policy::resolve(args, item, route_indices[route_row], ctx);
  }
  return ctx;
}

// At batch one the complete route table fits in a warp. match_any builds
// each row's expert-membership mask once; no additional shared memory is used.
__device__ __forceinline__ bool masked_chunk_selected(const int32_t* masks, int row) {
  const uint32_t before = static_cast<uint32_t>(masks[row]) & ((1u << row) - 1u);
  return (__popc(before) % kMaxGroup) == 0;
}

template<class Policy>
__device__ __forceinline__ GroupContext masked_group_plan(
    const typename Policy::Args& args, uint32_t item,
    const int32_t* masks, GroupCache& cache) {
  static_assert(Policy::kRouteRows <= 32);
  const int row = Policy::route_row(item);
  if (cache.route_row != row) {
    cache.route_row = row;
    cache.member_count = 0;
    cache.members = 0;
    if (masked_chunk_selected(masks, row)) {
      uint32_t remaining = static_cast<uint32_t>(masks[row]) & ~((1u << row) - 1u);
      while (remaining && cache.member_count < kMaxGroup) {
        const int member = __ffs(remaining) - 1;
        cache.members |= static_cast<uint64_t>(member) << (kMemberBits * cache.member_count++);
        remaining &= remaining - 1;
      }
      for (int m = cache.member_count; m < kMaxGroup; ++m)
        cache.members |= static_cast<uint64_t>(row) << (kMemberBits * m);
    }
  }
  GroupContext ctx{};
  ctx.member_count = cache.member_count;
  ctx.members = cache.members;
  if (ctx.member_count) Policy::resolve(args, item, args.indices[row], ctx);
  return ctx;
}

template <class Policy>
__device__ inline void execute_fp4_grouped_pipeline(
    const typename Policy::Args& args) {
  using namespace cute;
  constexpr int kKBlocks = Policy::kKBlocks;
  constexpr int kKWidth = kKBlocks * kFp4Block;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<TcgenSmemLayoutB>);
  static_assert(sizeof(TcgenSharedStorage) <= kSharedBytesBudget);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage =
      *reinterpret_cast<TcgenSharedStorage*>(dynamic_shared_memory);

  // Stage the route table into shared memory once: every leader check and
  // group gather below reads SMEM instead of re-walking the global indices
  // array per crew per item (the dominant grouped-body overhead when it was
  // measured against the contract-shape pipeline).
  for (int r = static_cast<int>(threadIdx.x); r < Policy::kRouteRows;
       r += static_cast<int>(blockDim.x)) {
    storage.route_indices[r] = args.indices[r];
  }
  __syncthreads();

  // Duplicate-only invocations (the production case: one item per task whose
  // route_row is a non-leader occurrence of its expert) exit before touching
  // any barrier or TMEM state. The predicate is uniform across the CTA, so
  // the early return is safe.
  {
    bool any_leader = false;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      if (chunk_selected(storage.route_indices, Policy::route_row(item))) {
        any_leader = true;
        break;
      }
    }
    if (!any_leader) {
      return;
    }
  }

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
        cutlass::arch::ClusterBarrier, kStages>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kStages>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  // Columns beyond a group's member count stay zero (or hold a previous
  // leader's stale columns, which are computed but never drained); only
  // columns 0..member_count-1 are refilled per block by cp.async.
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

  // `staged` counts ring slots actually used; every crew skips the same
  // non-leader items, so the ring/parity sequence agrees across crews and
  // the per-slot single-completion-per-round invariant is preserved.
  if (warp >= 5) {
    // Staging crew.
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    int staged = 0;
    GroupCache group_cache;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      const GroupContext ctx =
          group_plan<Policy>(args, item, storage.route_indices, group_cache);
      if (ctx.member_count == 0) {
        continue;
      }
      const uint8_t* const weight = ctx.weight;
      const int output_base = ctx.output_base;
      // The two 16-byte B chunks this lane owns for the whole item: lanes
      // 0..2*count-1 map to (member, half); the source row base is
      // item-invariant, so resolve it once instead of per block.
      const bool stages_input = lane < 2 * ctx.member_count;
      const int member = lane >> 1;
      const int half = lane & 1;
      const uint8_t* input_base = nullptr;
      if (stages_input) {
        input_base = reinterpret_cast<const uint8_t*>(
            Policy::input_row(args, group_member(ctx.members, member)))
            + half * 16;
      }
      for (int block = 0; block < kKBlocks; ++block, ++staged) {
        const int s = staged % kStages;
        if (staged >= kStages) {
          wait_barrier(
              storage.accumulator_full[s], (staged / kStages - 1) & 1);
        }
        uint8_t* stage_a = reinterpret_cast<uint8_t*>(storage.a[s].begin());
        for (int copy_index = lane; copy_index < kOutputTile * 2;
             copy_index += kStagingThreads) {
          const int output = copy_index >> 1;
          const int group = copy_index & 1;
          const uint8_t* source = weight
              + (static_cast<int64_t>(output_base + output) * kKWidth
                 + block * kFp4Block) / 2
              + group * 8;
          cutlass::arch::cp_async<8, cutlass::arch::CacheOperation::Always>(
              stage_a + output * 16 + group * 2048, source);
        }
        if (stages_input) {
          // B column m carries member m's activation slice: two 16-byte
          // chunks at byte offsets m*16 and 128 + m*16 of the K-inter atom.
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              reinterpret_cast<uint8_t*>(storage.b[s].begin())
                  + half * 128 + member * 16,
              input_base + block * kFp4Block);
        }
        cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
      }
    }
  } else if (warp == 4) {
    // MMA crew: identical to the N=1 body; the instruction always computes
    // all 8 columns, live or not.
    int staged = 0;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      if (!chunk_selected(storage.route_indices, Policy::route_row(item))) {
        continue;
      }
      for (int block = 0; block < kKBlocks; ++block, ++staged) {
        const int s = staged % kStages;
        wait_barrier(storage.stage_full[s], (staged / kStages) & 1);
        if (staged >= kStages) {
          wait_barrier(
              storage.accumulator_free[s], (staged / kStages - 1) & 1);
        }
        SM100_MMA_F8F6F4_SS::fma(
            weight_descriptor[s],
            input_descriptor[s],
            storage.tmem_base_ptr + s * 8,
            0u,
            instruction_descriptor);
        cutlass::arch::umma_arrive(&storage.accumulator_full[s]);
      }
    }
  } else {
    // Accumulator crew: drain each 128x8 TMEM partial and run the original
    // outer FP32 fmaf scale chain PER LIVE COLUMN — the input scale is the
    // member row's, the weight scale is shared across columns.
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int s) {
      auto slot = tmem_accumulator;
      slot.data() = storage.tmem_base_ptr + s * 8;
      return thread_copy.partition_S(slot);
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

    int staged = 0;
    GroupCache group_cache;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      const GroupContext ctx =
          group_plan<Policy>(args, item, storage.route_indices, group_cache);
      if (ctx.member_count == 0) {
        continue;
      }
      const uint8_t* const weight_scales = ctx.weight_scales;
      const int output_base = ctx.output_base;
      const int member_count = ctx.member_count;
      // Compile-time-specialized epilogue: the member loop is fully unrolled
      // at the EXACT group size for the sizes real routing can produce
      // (m <= kDraftRows), so no dead predicated loads/fmafs ride the drain
      // chain — the 8-wide predicated fallback (only reachable with
      // within-route-row duplicate indices, impossible for real top-k) was
      // measured to cost ~35% even at m == 1. Every output element's scale
      // chain is untouched, so all instantiations stay bitwise.
      auto drain_group = [&](auto member_constant) {
        constexpr int kMembers = decltype(member_constant)::value;
        const uint8_t* member_scales[kMembers];
        __nv_bfloat16* member_store[kMembers];
        float result[kMembers];
        CUTE_UNROLL
        for (int m = 0; m < kMembers; ++m) {
          const int member_row = group_member(ctx.members, m);
          member_scales[m] = Policy::input_scales_row(args, member_row);
          member_store[m] = Policy::store_row(args, ctx, member_row);
          result[m] = 0.0f;
        }
        for (int block = 0; block < kKBlocks; ++block, ++staged) {
          const int s = staged % kStages;
          const int parity = staged % 2;
          wait_barrier(storage.accumulator_full[s], (staged / kStages) & 1);
          if (s == 0) {
            copy(tmem_to_register, thread_tmem_0, register_accumulator);
          } else if (s == 1) {
            copy(tmem_to_register, thread_tmem_1, register_accumulator);
          } else if (s == 2) {
            copy(tmem_to_register, thread_tmem_2, register_accumulator);
          } else {
            copy(tmem_to_register, thread_tmem_3, register_accumulator);
          }
          cutlass::arch::fence_view_async_tmem_load();
          cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[s]);
          if (parity == 0) {
            copy(register_accumulator, thread_partial_low);
          } else {
            copy(register_accumulator, thread_partial_high);
          }
          cutlass::arch::NamedBarrier::sync(4 * 32, 0);
          const float weight_scale = decode_e8m0(
              weight_scales[
                  (output_base + static_cast<int>(threadIdx.x))
                      * kKBlocks + block]);
          CUTE_UNROLL
          for (int m = 0; m < kMembers; ++m) {
            // Exact instantiations (kMembers < kMaxGroup) fold the guard
            // away; only the fallback keeps the runtime predicate, which
            // also protects the padded member slots from storing.
            if (kMembers < kMaxGroup || m < member_count) {
              const float input_scale =
                  decode_e8m0(member_scales[m][block / 4]);
              result[m] = fmaf(
                  storage.partial[parity][m * 128 + threadIdx.x],
                  input_scale * weight_scale,
                  result[m]);
            }
          }
          if (block == kKBlocks - 1) {
            CUTE_UNROLL
            for (int m = 0; m < kMembers; ++m) {
              if (kMembers < kMaxGroup || m < member_count) {
                member_store[m][output_base + threadIdx.x] =
                    __float2bfloat16_rn(result[m]);
              }
            }
          }
        }
      };
      static_assert(kMaxGroup == 8);
      switch (member_count) {
        case 1: drain_group(cute::Int<1>{}); break;
        case 2: drain_group(cute::Int<2>{}); break;
        case 3: drain_group(cute::Int<3>{}); break;
        case 4: drain_group(cute::Int<4>{}); break;
        case 5: drain_group(cute::Int<5>{}); break;
        default: drain_group(cute::Int<kMaxGroup>{}); break;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}

struct W13GroupPolicy {
  using Args = dspark_w13::Args;
  static constexpr int kKBlocks = kHidden / kFp4Block;
  static constexpr int kRouteRows = kDraftRows * kActivatedExperts;
#if defined(DSPARK_W13_TMA_CANDIDATE)
  static constexpr int kTmaPlanes = 4;
  static constexpr int kWeightRowBytes = kHidden / 2;
  // Output rows per routed W1/W3 matrix (the stage-major repack's plane
  // stride is kWeightRows * 64 B).
  static constexpr int kWeightRows = kIntermediate;
#endif
  // Measured breakeven vs the N=1 body (overlap sweep, iteration 6): the
  // grouped W13 body needs ~12 deduplicated route_rows to win.
  static constexpr int kMinDuplicates = 12;

  static __device__ __forceinline__ int route_row(uint32_t item) {
    return static_cast<int>(item) / kCombinedTiles;
  }

  static __device__ __forceinline__ void resolve(
      const Args& args, uint32_t item, int expert, GroupContext& ctx) {
    const int combined_tile = static_cast<int>(item) % kCombinedTiles;
#ifdef DSPARK_ROUTED_INTERLEAVE
    ctx.branch = 0;
    ctx.output_base = combined_tile * kOutputTile;
#else
    ctx.branch = combined_tile / (kIntermediate / kOutputTile);
    ctx.output_base =
        (combined_tile % (kIntermediate / kOutputTile)) * kOutputTile;
#endif
    const int weight_slot = ctx.branch == 0 ? args.w1_slot : args.w3_slot;
    const int scale_slot =
        ctx.branch == 0 ? args.w1_scale_slot : args.w3_scale_slot;
    const int table_base =
        args.weight_base + expert * args.expert_weight_slots;
    ctx.weight =
        args.weight_arena + args.weight_offsets[table_base + weight_slot];
    ctx.weight_scales =
        args.weight_arena + args.weight_offsets[table_base + scale_slot];
  }

  static __device__ __forceinline__ const cutlass::float_e4m3_t* input_row(
      const Args& args, int route_row) {
    return args.input + (route_row / kActivatedExperts) * kHidden;
  }
  static __device__ __forceinline__ const uint8_t* input_scales_row(
      const Args& args, int route_row) {
    return args.input_scales
        + (route_row / kActivatedExperts) * (kHidden / kQuantBlock);
  }
  static __device__ __forceinline__ __nv_bfloat16* store_row(
      const Args& args, const GroupContext& ctx, int route_row) {
    return args.routed_w13 + (route_row * 2 + ctx.branch) * kIntermediate;
  }
};

__device__ inline void execute_tcgen_grouped(const Args& args) {
  execute_fp4_grouped_pipeline<W13GroupPolicy>(args);
}

// Auto dispatch: run the grouped body only when the route table has enough
// duplicate (row, expert) pairs to amortize the grouped machinery, else the
// contract-shape N=1 body. The predicate is a pure function of args.indices,
// so every CTA (and every item's invocation) dispatches identically — the
// item space is shared by both bodies, which is what makes the per-phase
// choice legal.
//
// MEASURED NEGATIVE (iteration 6, do not wire): the O(kRouteRows^2)
// global-index scan runs per invocation and costs more than it ever saves,
// and after the exact-size drain specialization the grouped body beats the
// N=1 body at EVERY overlap (1.05-1.06x even with zero duplicates), so
// there is nothing left to dispatch away from. Kept only as the recorded
// negative result; the wiring target is execute_tcgen_grouped /
// execute_w2_tcgen_grouped unconditionally.
template <class Policy>
__device__ __forceinline__ bool grouped_profitable(
    const typename Policy::Args& args) {
  int duplicates = 0;
  for (int r = 1; r < Policy::kRouteRows; ++r) {
    const int expert = args.indices[r];
    for (int q = 0; q < r; ++q) {
      if (args.indices[q] == expert) {
        ++duplicates;
        break;
      }
    }
  }
  return duplicates >= Policy::kMinDuplicates;
}

__device__ inline void execute_tcgen_auto(const Args& args) {
  if (grouped_profitable<W13GroupPolicy>(args)) {
    execute_fp4_grouped_pipeline<W13GroupPolicy>(args);
  } else {
    execute_fp4_pipeline<W13Policy>(args);
  }
}

// ---------------------------------------------------------------------------
// DeepGEMM-transplanted grouped pipeline (relaxed-drafter Leg 2 item 1).
//
// Provenance: structure and device helpers adapted from DeepGEMM (MIT),
// https://github.com/deepseek-ai/DeepGEMM
// commit 559d79fb6994a58b8a15b4b93bf13ccc16edf247:
//   - deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh
//     (block-scaled UMMA with UE8M0 scale factors resident in TMEM, UTCCP
//      scale staging, BLOCK_K=128 pipeline granularity with 4 chained
//      UMMAs per stage, TMEM-chained accumulation over the full K extent,
//      ONE epilogue drain per output block instead of one per 32-K block)
//   - deep_gemm/include/deep_gemm/mma/sm100.cuh (make_sf_desc /
//     make_runtime_instr_desc_with_sf_id equivalents)
//   - deep_gemm/include/deep_gemm/ptx/tcgen05.cuh (SM100_MMA_MXF8F6F4_SS
//     inline-asm form; CUTLASS 4.2.0's cute atom of the same name has a
//     missing comma in the UTCCP asm operand list, so the tcgen05.cp is
//     also emitted by a local adapted wrapper)
// See the repository NOTICE file for the DeepGEMM credit entry.
//
// What changes vs execute_fp4_grouped_pipeline:
//   - the E8M0 scales are applied INSIDE the tensor core
//     (kind::mxf8f6f4.block_scale): SFA = per-output-row per-32-K weight
//     scales (4 packed per uint32, sf_id selects the byte = DeepGEMM's
//     kGranK=32 path), SFB = per-member per-128-K activation scales
//     (one uint32 covers 4 stages, sf_id = stage % 4 = the kGranK=128
//     path). Both ride the cp.async ring as transposed 512 B blocks and
//     reach TMEM via tcgen05.cp 32x128b.warpx4, exactly DeepGEMM's flow.
//   - the FP32 accumulator chains in TMEM across all K blocks of an item
//     (scaleC = 0 only on the item's first UMMA), so the accumulator crew
//     drains ONCE per item instead of once per 32-K block. That drain and
//     its two GMEM scale reads per thread per block were the grouped
//     body's steady-state limiter.
//   - ring slots stage 128 K elements (4 UMMA chunks) instead of 32, so
//     each item runs kKBlocks/4 barrier rounds instead of kKBlocks.
//
// Item semantics, item count, Args ABI and the expert-grouped leader /
// no-op structure are UNCHANGED from execute_tcgen_grouped: this is a
// pure body swap over the same 960-tile space.
//
// Numerics: the TC 32-element dot per (row, member, K-block) is the same
// datapath as kind::f8f6f4 (bitwise vs the scalar inner chain, iteration-6
// precedent) and the E8M0 scales are exact powers of two, but the
// per-K-block scale-and-add now happens in the tensor core's chained FP32
// accumulator instead of the frozen outer fmaf chain. Expected drift class
// is the shared-expert/q_b precedent: bitwise on real checkpoint data,
// <= 1 BF16 ULP on adversarial synthetic scales. The bench gates report
// bitwise mismatches and enforce ULP <= 1.

// Ring depth of the DeepGEMM-transplanted routed body. 7 was chosen because
// DgSharedStorage at 7 stages already fills 137,216 B of the 139,520 B
// budget: deeper rings were rejected for lack of room, not because they
// stopped helping. Raising DSPARK_V4_DYNAMIC_SMEM_BYTES is what buys the
// room back, so the depth has to be settable alongside it. Default is the
// shipped 7, so an unset build is byte-identical.
#ifndef DSPARK_W13_DG_STAGES
#define DSPARK_W13_DG_STAGES 7
#endif
constexpr int kDgStages = DSPARK_W13_DG_STAGES;
#ifndef DSPARK_W13_DG_CHUNKS
#define DSPARK_W13_DG_CHUNKS 4
#endif
constexpr int kDgChunks = DSPARK_W13_DG_CHUNKS;
static_assert(kDgChunks == 4 || kDgChunks == 8 || kDgChunks == 16,
              "the routed pipeline supports K=128, K=256, or K=512 stages");
constexpr int kDgScaleGroups = kDgChunks / 4;
constexpr int kDgBlockK = kDgChunks * kFp4Block;

// TRTLLM-gen's selected GB300 tactic is `t128x8x512u2_s3`: the `u2`
// mainloop duplicates two K=512 pipeline iterations in one software loop
// body. The 2026-08-26 GB300 A/B/A kept every output bitwise and removed
// 1,752 decoded instructions without adding spills, but it was mixed:
// static queues improved only 5.5 us / 0.37%, dynamic queues regressed
// 12.5 us / 0.66%, and the production trace showed no routed-phase win.
// Keep it opt-in: unlike the standalone tactic, this translation unit
// contains every proposal phase and already sits at the 255-register ceiling.
// A value of 1 preserves the retained SASS structure.
#ifndef DSPARK_W13_DG_STAGE_UNROLL
#define DSPARK_W13_DG_STAGE_UNROLL 1
#endif
constexpr int kDgStageUnroll = DSPARK_W13_DG_STAGE_UNROLL;
static_assert(
    kDgStageUnroll == 1 || kDgStageUnroll == 2,
    "the routed K-stage loop supports unroll factors one or two");

// R13 crew assignment of the routed DeepGEMM body.
//
// The frozen split of the 256-thread CTA is 4 drain warps (0..3), one MMA
// warp (4) and 3 staging warps (5..7). The drain runs ONCE PER ITEM while the
// ring turns kStagesPerItem (32 for W13, 16 for W2) rounds, so half the CTA's
// issue capacity sits on a barrier for ~97% of an item. Threads-per-CTA is
// megakernel-wide and off limits; WHICH WARP DOES WHAT is not.
//
//   bit 0 (kDgWideStage): warps 0..3 join the staging crew and perform their
//     once-per-item drain INSIDE that loop, kDgStages rounds into the next
//     item -- the point at which the ring's own stage_free handshake already
//     guarantees the accumulator barrier has fired, so the drain never
//     stalls. Staging threads 96 -> 224 (or 192 with bit 1).
//   bit 1 (kDgSplitMma): the fused tile's two 128-row planes are independent
//     UMMA chains into separate TMEM slots, so warp 7 issues plane 1 while
//     warp 4 issues plane 0. Costs one staging warp; only meaningful with
//     kDgTilePairs >= 2.
//
// BOTH ARE FALSIFIED ON THE PATH. Default stays 0; this is apparatus and a
// standing record, not a shipped body.
//
//   Bit 0 IS worth 1.0208x IN THE PHASE BENCH (two independent same-session
//   sweeps: pair 166.78 -> 163.39 us at claim 4, and 161.98 -> 158.78 /
//   156.26 -> 151.26 at claims 4/8 in the second), and it is bitwise
//   everywhere. But the same build measured IN ONE PROCESS ON ONE GPU against
//   crew 0 on the REAL PATH (`v4_batch_bench.py --crew-configs 0,1`) is
//   SLOWER: routed W13 303.5 -> 308.4 us, routed W2 187.8 -> 193.7, pair
//   491.3 -> 502.1 (1.022x SLOWER), batch-1 wall 2.250832 -> 2.261280 ms.
//   WHY THE TWO LANES DISAGREE, and it is a caveat for every future routed
//   candidate: the phase bench builds its route table with
//   make_overlap_indices(..., 0.0), i.e. ZERO shared experts, so every group
//   has member_count == 1 and the drain is at its cheapest. Real routing
//   produces multi-member groups, and this variant moves that bigger drain
//   ONTO the ring's critical path (it used to be free on four dedicated
//   warps). At batch 8, where items per CTA rise and the staging saving grows
//   relative to the drain, the sign flips back: wall 5.304176 -> 5.273664.
//
//   Bit 1 loses even in the bench -- crew 3 lands at 1.0073x, i.e. 1.33%
//   BEHIND crew 1 -- because the tcgen05 unit serializes the UTCCP/UMMA
//   stream no matter which warp issues it, so a second MMA warp buys nothing
//   and the staging warp it costs is worth 1.33%.
//
// THE STRUCTURAL READING, measured with DSPARK_W13_CREW_PROBE: the drain crew
// really is idle (94.2% of the W13 body parked on accumulator_full, 89.1% on
// W2 -- the R11 claim is confirmed), but recruiting it cannot pay, because
// per-warp staging issue falls only 28% for +133% threads (the streams queue
// on a shared per-SM request pipe, not on thread issue capacity) and the MMA
// warp is the pacer -- its stall on stage_full is already only 27.2%, and the
// wide crew drops that to 20.0% while the round barely moves.
//
// Both are pure work re-assignment: the same packets reach the same shared
// addresses and every accumulator's K chain, chunk order and scale schedule
// are untouched, so both are BITWISE by construction. Default 0 keeps the
// frozen assignment, so an unset build is byte-identical.
#ifndef DSPARK_W13_DG_CREW
#define DSPARK_W13_DG_CREW 0
#endif
constexpr int kDgCrew = DSPARK_W13_DG_CREW;
constexpr bool kDgWideStage = (kDgCrew & 1) != 0;
constexpr bool kDgSplitMma = (kDgCrew & 2) != 0;
static_assert(kDgCrew >= 0 && kDgCrew <= 3, "crew mask is two bits");



// R7 SFA scale-line cache (DSPARK_W13_SFA_CACHE).
//
// The shipped body re-reads the weight scale array once PER STAGE: 128
// output rows x one uint32 each, and a row's whole per-32-K scale line is
// exactly kKBlocks bytes = one 128 B line for W13. Over the item's
// kStagesPerItem stages that is 128 x 32 = 4,096 four-byte global requests
// for 16 KiB of data. Nsight measured the consequence directly: the SFA
// LDGSTS carries 32 L1 tag requests and 32 shared wavefronts per warp
// instruction (ideal 1), and it alone accounts for 2,637,824 of W13's
// 2,644,992 excessive L2 sectors and 2,920,448 of its 3,108,864 excessive
// shared wavefronts. Masking the stream off is worth 10.28% of the ring
// round on B200.
//
// The cache reads each scale line ONCE per item with 16-byte loads and
// transposes it in registers into all kStagesPerItem UTCCP source blocks at
// once, so the per-stage gather disappears entirely. Cost accounting per
// item, W13 shape:
//   shipped:  4,096 tag requests + 4,096 shared wavefronts + 4,096 sectors
//   cached:     256 tag requests +   512 shared wavefronts +   512 sectors
// Double-buffered on item parity because the staging crew runs up to
// kDgStages rounds ahead of the MMA crew.
//
// The transpose writes with PLAIN shared stores, which
// cpasync_barrier_arrive_noinc does not order (it tracks cp.async traffic
// only), so the buffer gets its own release/acquire pair: an async-proxy
// fence plus a dedicated mbarrier the MMA crew waits on once per item, and
// a tcgen05.commit-backed free barrier so the buffer is not overwritten
// while a UTCCP is still reading it.
constexpr int kDgSfaCacheStages = kHidden / kFp4Block / kDgChunks;

struct DgSharedStorage {
#if defined(DSPARK_W13_TMA_CANDIDATE)
  // SM100's 128B-swizzled UMMA descriptor repeats every 1 KiB. Align the
  // first stage to that repeat boundary; every 16 KiB A stage then remains
  // aligned without a nonzero descriptor base offset.
  alignas(1024) cute::ArrayEngine<
#else
  alignas(128) cute::ArrayEngine<
#endif
      uint8_t,
      kDgTilePairs * kDgChunks * cute::cosize_v<TcgenSmemLayoutA>>
      a[kDgStages];
  alignas(128) cute::ArrayEngine<
      TcgenInputType,
      kDgChunks * cute::cosize_v<TcgenSmemLayoutB>> b[kDgStages];
  // Transposed scale-factor blocks (word l*4+i holds row i*32+l, the
  // post-transpose layout DeepGEMM's UTCCP path expects). Only the first
  // 8 SFB rows are live (N = 8 columns); the rest are never consumed.
  // SFA is per tile plane (the weight scales are output-row indexed); SFB is
  // shared by both planes (the activations are plane invariant).
  alignas(128) uint32_t
      sfa[kDgStages][kDgTilePairs][kDgScaleGroups][kOutputTile];
  alignas(128) uint32_t sfb[kDgStages][kOutputTile];
  alignas(16) float partial[2][kDgTilePairs][128 * 8];
  alignas(16) cute::uint64_t stage_full[kDgStages];
  alignas(16) cute::uint64_t stage_free[kDgStages];
  alignas(16) cute::uint64_t accumulator_full[2];
  alignas(16) cute::uint64_t accumulator_free[2];
  alignas(16) cute::uint32_t tmem_base_ptr;
  alignas(16) int32_t route_indices[kRouteCacheSlots];
#if defined(DSPARK_W13_TMA_CANDIDATE)
  // Append the candidate-only transaction barriers so every retained member
  // keeps its original offset in the control instantiation.
  alignas(16) cute::uint64_t weight_full[kDgStages];
#endif
#if defined(DSPARK_W13_SFA_CACHE)
  // Appended LAST, after every retained member, so the uncached build has a
  // byte-identical layout. sfa[] above goes dead in this build; keeping it
  // costs 3,584 B and keeps every other offset where it was.
  alignas(128) uint32_t sfa_item[2][kDgSfaCacheStages][kOutputTile];
  alignas(16) cute::uint64_t sfa_group_ready[kDgSfaCacheStages / 4];
  alignas(16) cute::uint64_t sfa_free[2];
#endif
  // Immutable compact ordinal lookup, published by the existing CTA join.
  alignas(16) uint32_t compact_leader_rows[32];
  // W2-only immutable packed members; initialized before the existing CTA join.
  // This field occupies verified tail padding in the current two-stage layout.
  alignas(16) uint64_t compact_w2_members[32];

};

// Adapted from deep_gemm::mma::sm100::make_sf_desc: SWIZZLE_NONE atom of
// 8 x 128 bits, SBO = 128 bytes between atoms on MN, LBO = 0 (one atom
// on K).
__device__ __forceinline__ uint64_t dg_make_sf_desc(const void* smem_ptr) {
  cute::UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ =
      static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_NONE);
  desc.start_address_ = static_cast<uint16_t>(
      cute::cast_smem_ptr_to_uint(smem_ptr) >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = (8 * 16) >> 4;
  desc.leading_byte_offset_ = 0;
  return desc.desc_;
}

#if defined(DSPARK_W13_TMA_CANDIDATE)
// DeepGEMM's packed-subbyte TMA destination is a 128B-swizzled, K-major
// 128x128 byte-slot tile. CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B expands
// every 8 packed GMEM bytes into one 16-byte SMEM group; this descriptor
// consumes those padded byte slots directly, without repacking HBM weights.
__device__ __forceinline__ uint64_t dg_make_tma_weight_desc(
    const void* smem_ptr) {
  cute::UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ =
      static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  desc.start_address_ = static_cast<uint16_t>(
      cute::cast_smem_ptr_to_uint(smem_ptr) >> 4);
  desc.base_offset_ = 0;
  // The 128B swizzle repeats at a 1 KiB stride across output rows; there is
  // only one swizzle atom along K, so LBO remains zero.
  desc.stride_byte_offset_ = (8 * 128) >> 4;
  desc.leading_byte_offset_ = 0;
  return desc.desc_;
}

// kStageMajor selects the R1-leg-8 candidate layout. The logical weight
// matrix is [row][stage][64 B cell]; the repacked arena stores it as
// [stage][row][64 B cell], so the 128 cells a stage needs form ONE
// contiguous 8 KiB run instead of 128 cells strided by kWeightRowBytes.
// The tensor map keeps the SAME global dims/strides (the repack is a pure
// byte permutation of the arena, not a reinterpretation); only the box
// changes, from {128,1,1,128} to {128,8,planes,128/(8*planes)}. TMA fills
// SMEM in box-dimension order with dim0 innermost, so the SMEM row index
// becomes d1 + 8*d2 + 8*planes*d3 — exactly the cell's position inside the
// contiguous run, i.e. the same row ordering the row-major box produced.
// The destination tile is therefore byte-identical and the UMMA operands
// are unchanged: BITWISE by construction.
template <class Policy, bool kStageMajor = false>
__device__ __forceinline__ void dg_tma_load_weight(
    const typename Policy::Args& args,
    const GroupContext& ctx,
    const void* weight_tma_descriptor,
    int micro_stage,
    void* destination,
    uint64_t* barrier,
    int row_offset = 0,
    int expect_bytes = kOutputTile * 4 * kFp4Block / 2) {
  constexpr uint64_t kCellBytes = 4 * kFp4Block / 2;
  constexpr uint64_t kCellsPerPlane = 8;
  constexpr uint64_t kPlanes = Policy::kTmaPlanes;
  constexpr uint64_t kRowBytes = kCellBytes * kCellsPerPlane * kPlanes;
  static_assert(kCellBytes == 64);
  static_assert(kRowBytes == Policy::kWeightRowBytes);
  static_assert(kOutputTile % (kCellsPerPlane * kPlanes) == 0);

  const uint64_t matrix_offset = static_cast<uint64_t>(
      ctx.weight - args.weight_arena);
  uint32_t coord_1;
  uint32_t coord_2;
  uint32_t coord_3;
  const int output_base = ctx.output_base + row_offset;
  if constexpr (kStageMajor) {
    const uint64_t byte_offset =
        matrix_offset
        + static_cast<uint64_t>(micro_stage)
            * (static_cast<uint64_t>(Policy::kWeightRows) * kCellBytes)
        + static_cast<uint64_t>(output_base) * kCellBytes;
    coord_1 = 0;
    coord_2 = 0;
    coord_3 = static_cast<uint32_t>(byte_offset / kRowBytes);
  } else {
    const uint64_t byte_offset =
        matrix_offset
        + static_cast<uint64_t>(output_base) * kRowBytes
        + static_cast<uint64_t>(micro_stage) * kCellBytes;
    const uint64_t cell = byte_offset / kCellBytes;
    coord_1 = static_cast<uint32_t>(cell % kCellsPerPlane);
    coord_2 = static_cast<uint32_t>((cell / kCellsPerPlane) % kPlanes);
    coord_3 = static_cast<uint32_t>(cell / (kCellsPerPlane * kPlanes));
  }

  // Set the byte expectation before the asynchronous copy can retire. The
  // barrier was initialized with one arrival, so this both arrives and arms
  // the transaction count for the stage's packed 8 KiB payload. With a fused
  // tile the expectation covers every live plane and is armed once, by the
  // first plane's call.
  if (expect_bytes > 0) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        barrier, expect_bytes);
  }
  cute::SM90_TMA_LOAD_4D::copy(
      weight_tma_descriptor,
      barrier,
      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
      destination,
      0,
      coord_1,
      coord_2,
      coord_3);
}
#endif

// Adapted from deep_gemm::mma::sm100::make_runtime_instr_desc_with_sf_id.
__device__ __forceinline__ uint64_t dg_runtime_instr_desc(
    cute::UMMA::InstrDescriptorBlockScaled desc,
    uint32_t sfa_id,
    uint32_t sfb_id) {
  desc.a_sf_id_ = sfa_id;
  desc.b_sf_id_ = sfb_id;
  return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

// Adapted from deep_gemm::ptx (tcgen05.cuh): UTCCP and block-scaled MMA
// issued unguarded so the caller's single elect_one region keeps every
// tcgen05 op on one thread (in-order async execution, the ordering the
// per-stage SF TMEM overwrite relies on).
__device__ __forceinline__ void dg_utccp_32x128b_warpx4(
    uint64_t src_desc, uint32_t dst_tmem) {
  asm volatile(
      "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
      :
      : "r"(dst_tmem), "l"(src_desc));
}

__device__ __forceinline__ void dg_mma_mxf8f6f4(
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t tmem_c,
    uint32_t scale_c,
    uint64_t idesc,
    uint32_t tmem_sfa,
    uint32_t tmem_sfb) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
      "[%0], %1, %2, %3, [%5], [%6], p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
        "r"(static_cast<uint32_t>(idesc >> 32)), "r"(scale_c),
        "r"(tmem_sfa), "r"(tmem_sfb));
}

// Live tile planes of the fused pass starting at `item`. A routed claim is
// pair aligned by construction (generate_v4_header.py rounds the routed claim
// chunk up to an even value and every routed work-unit count is a multiple of
// 32), so this returns kDgTilePairs everywhere in production; the clamp keeps
// a short trailing range correct instead of reading the next route row's
// weights. At kDgTilePairs == 1 it folds to the literal 1 and every pair loop
// below collapses to the frozen single-plane body.
template <class Args>
__device__ __forceinline__ int dg_live_planes(const Args& args, uint32_t item) {
  if constexpr (kDgTilePairs == 1) {
    return 1;
  } else {
    const uint32_t remaining = args.end - item;
    return remaining >= static_cast<uint32_t>(kDgTilePairs)
        ? kDgTilePairs
        : static_cast<int>(remaining);
  }
}

struct W2GroupPolicy;

template <
    class Policy,
    bool kUseTmaWeights = false,
    bool kStageMajorWeights = false,
    int kStagingProbe = 0,
    int kPackedSfa = 0,
    int kSfRing = 0,
    bool kGroupMasks = false,
    bool kReuseTmem = false,
    bool kCompact = false,
    bool kFuseSwiglu = false>
__device__ inline void execute_dg_grouped_pipeline(
    const typename Policy::Args& args
#if defined(DSPARK_W13_TMA_CANDIDATE)
    ,
    const void* weight_tma_descriptor = nullptr
#endif
) {
  using namespace cute;
  static_assert(kPackedSfa >= 0 && kPackedSfa <= 2, "unsupported packed SFA mode");
#ifdef DSPARK_W13_SFA_CACHE
  static_assert(kPackedSfa == 0, "offline SFA packing replaces the runtime SFA cache");
#endif
  constexpr int kKBlocks = Policy::kKBlocks;
  constexpr int kStagesPerItem = kKBlocks / kDgChunks;
  constexpr int kKWidth = kKBlocks * kFp4Block;
  // Accumulator slots (2 epilogue stages x kDgTilePairs planes x 8 columns),
  // then kDgTilePairs SFA tiles of 4 columns, then one shared SFB tile.
  // kDgTilePairs = 1 reproduces the frozen map exactly: acc 0..15,
  // SFA 16..19, SFB 20..23 inside a 32-column allocation.
  constexpr int kAccColumns = 2 * kDgTilePairs * 8;
  // A second MMA warp gets its OWN SFB tile: tcgen05 ops are ordered within
  // the issuing warp, so a warp must never read a scale tile another warp
  // copied without a handshake. TMEM is not scarce here (32 or 64 of 512
  // columns at 1 CTA/SM), so the copy is free.
  static_assert(!kFuseSwiglu || (kCompact && std::is_same_v<Policy, W13GroupPolicy> && !kDgWideStage));
  static_assert(!kCompact || (kGroupMasks && kDgTilePairs == 1));
  static_assert(!kCompact || Policy::kRouteRows == 30);
#ifdef DSPARK_ROUTED_W2_WORKERS
  static_assert(DSPARK_ROUTED_W2_WORKERS == 128);
  static_assert(DSPARK_ROUTED_COMPACT_WORKERS == 120);
  constexpr uint32_t kCompactWorkers = std::is_same_v<Policy, W2GroupPolicy>
      ? DSPARK_ROUTED_W2_WORKERS : DSPARK_ROUTED_COMPACT_WORKERS;
#else
#ifdef DSPARK_ROUTED_COMPACT_WORKERS
  constexpr uint32_t kCompactWorkers = DSPARK_ROUTED_COMPACT_WORKERS;
#else
  constexpr uint32_t kCompactWorkers = 152;
#endif
#endif
  static_assert(kCompactWorkers > 0 && kCompactWorkers <= 152);
  if constexpr (kCompact) {
    // Existing scheduler tickets still receive their original completion
    // credits. Only the first 152 tickets own compact cyclic tile streams.
    if (args.begin >= kCompactWorkers) return;
  }
  constexpr int kDgMmaWarps = kDgSplitMma ? 2 : 1;
  static_assert(kSfRing >= 0 && kSfRing <= 2);
  constexpr int kSfStages = kSfRing == 2 ? kDgStages : 1;
  constexpr int kSfGroups = kSfRing != 0 ? kDgScaleGroups : 1;
  constexpr int kSfaColumns = 4 * kDgTilePairs * kSfGroups * kSfStages;
  constexpr int kSfColumns = kSfaColumns + 4 * kDgMmaWarps * kSfStages;
  constexpr int kTmemNeeded = kAccColumns + kSfColumns;
  static_assert(kTmemNeeded <= 128, "routed scale ring exceeds its TMEM allocation");
  constexpr int kTmemColumns =
      kTmemNeeded <= 32 ? 32 : (kTmemNeeded <= 64 ? 64 : 128);
  constexpr int kStageWeightBytes = kDgTilePairs * kDgChunks
      * static_cast<int>(cute::cosize_v<TcgenSmemLayoutA>);
  constexpr int kPlaneWeightBytes = kStageWeightBytes / kDgTilePairs;
  constexpr int kMicroStageWeightBytes = 4
      * static_cast<int>(cute::cosize_v<TcgenSmemLayoutA>);
  constexpr int kMicroStageInputBytes = 4
      * static_cast<int>(cute::cosize_v<TcgenSmemLayoutB>);
  // Staging warps: 5..7 in the frozen assignment, plus the four drain warps
  // when kDgWideStage, minus warp 7 when it is the second MMA warp.
  // Dedicate warp5 to immutable weights/SFA for fused W13.
#ifdef DSPARK_ROUTED_WEIGHT_WARP
  constexpr bool kDedicatedWeights = std::is_same_v<Policy, W13GroupPolicy>
      && kFuseSwiglu && kUseTmaWeights && kPackedSfa == 2
      && !kDgWideStage && !kDgSplitMma;
#else
  constexpr bool kDedicatedWeights = false;
#endif
  constexpr int kDgStagingWarps =
      (kDedicatedWeights ? 2 : (kDgSplitMma ? 2 : 3)) + (kDgWideStage ? 4 : 0);
  constexpr int kStagingThreads = kDgStagingWarps * 32;
  static_assert(!kDgSplitMma || kDgTilePairs >= 2,
                "the second MMA warp only has work with a fused tile");
  // The drain rides the next item's rounds; kDgStages of them are enough for
  // the ring handshake to have already fired the accumulator barrier.
  constexpr int kDgDrainStage =
      kDgStages < kStagesPerItem ? kDgStages : kStagesPerItem - 1;
  // kStagingProbe is a BENCH-ONLY timing attribution mask: bit 0 drops the
  // weight stream, bit 1 the member activation slices, bit 2 the SFA scale
  // gather, bit 3 the SFB gather. Any non-zero mask produces GARBAGE
  // NUMERICS and exists only to price each staging stream inside the
  // 465 ns ring round. Production never instantiates it.
#if !defined(DSPARK_W13_STAGING_PROBE)
  static_assert(
      kStagingProbe == 0,
      "staging attribution probes require -DDSPARK_W13_STAGING_PROBE");
#endif
  constexpr bool kProbeSkipWeights = (kStagingProbe & 1) != 0;
  constexpr bool kProbeSkipInput = (kStagingProbe & 2) != 0;
  constexpr bool kProbeSkipSfa = (kStagingProbe & 4) != 0;
  constexpr bool kProbeSkipSfb = (kStagingProbe & 8) != 0;
  static_assert(kKBlocks % kDgChunks == 0);
  static_assert(kStagesPerItem >= kDgStages);
  static_assert(kMaxGroup == 8);  // members ride the N=8 UMMA columns
  static_assert(sizeof(DgSharedStorage) <= kSharedBytesBudget);
#if defined(DSPARK_W13_SFA_CACHE)
  static_assert(
      kDgTilePairs == 1,
      "the falsified SFA line cache is single-plane only");
  static_assert(
      kDgChunks == 4,
      "the falsified SFA line cache is a K=128 experiment");
  static_assert(
      !kDgWideStage,
      "the falsified SFA line cache assumes the frozen crew assignment");
#endif

#if defined(DSPARK_W13_TMA_CANDIDATE)
  extern __shared__ __align__(1024) unsigned char dynamic_shared_memory[];
#else
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
#endif
  auto& storage =
      *reinterpret_cast<DgSharedStorage*>(dynamic_shared_memory);

  if constexpr (kGroupMasks) {
    static_assert(Policy::kRouteRows <= 32);
#ifdef DSPARK_V4_TP2_ABLATE
    static_assert(!kGroupMasks, "group masks are a single-GPU experiment");
#endif
    if (threadIdx.x < 32) {
      const int row = threadIdx.x;
      const int expert = row < Policy::kRouteRows ? args.indices[row] : -1 - row;
      const uint32_t members = __match_any_sync(0xffffffffu, expert);
      if (row < Policy::kRouteRows) storage.route_indices[row] = static_cast<int32_t>(members);
      if constexpr (kCompact) {
        const uint32_t earlier = members & ((1u << row) - 1u);
        const bool leader = row < Policy::kRouteRows && (__popc(earlier) % 8 == 0);
        const uint32_t leaders = __ballot_sync(0xffffffffu, leader);
        // Batch 1 leaves two padding words in the existing 32-entry cache.
        if (row == 0) storage.route_indices[30] = static_cast<int32_t>(leaders);
        if (leader) {
          const unsigned ordinal = __popc(leaders & ((1u << row) - 1u));
          storage.compact_leader_rows[ordinal] = static_cast<uint32_t>(row);
          if constexpr (std::is_same_v<Policy, W2GroupPolicy>) {
            uint32_t remaining = members & ~((1u << row) - 1u);
            uint64_t packed = 0;
            #pragma unroll
            for (int m = 0; m < kMaxGroup; ++m) {
              const int member = remaining ? __ffs(remaining) - 1 : row;
              packed |= static_cast<uint64_t>(member) << (kMemberBits * m);
              remaining &= remaining - 1;
            }
            storage.compact_w2_members[row] = packed;
          }
        }
      }
    }
  } else {
  for (int r = static_cast<int>(threadIdx.x); r < Policy::kRouteRows;
       r += static_cast<int>(blockDim.x)) {
    storage.route_indices[r] = args.indices[r];
  }
  }
  __syncthreads();
  const uint32_t compact_leaders = kCompact ? static_cast<uint32_t>(storage.route_indices[30]) : 0;
  const uint32_t item_end = kCompact ? __popc(compact_leaders) * 32 : args.end;
  const uint32_t item_stride = kCompact ? kCompactWorkers : kDgTilePairs;
  const uint32_t item_begin = kFuseSwiglu ? args.begin * 2 : args.begin;
  const auto next_item = [&](uint32_t item) {
    if constexpr (kFuseSwiglu) return item + ((item & 1) ? item_stride * 2 - 1 : 1);
    else return item + item_stride;
  };
  const auto physical_item_at = [&](uint32_t item) {
    if constexpr (kCompact) {
      const uint32_t row = storage.compact_leader_rows[item / 32];
      const uint32_t tile = kFuseSwiglu ? (item % 32) / 2 + (item % 2) * 16 : item % 32;
      return row * 32 + tile;
    } else return dg_physical_item(args, item);
  };
  const auto selected = [&](int row) {
    if constexpr (kGroupMasks) return masked_chunk_selected(storage.route_indices, row);
    else return chunk_selected(storage.route_indices, row);
  };
  const auto resolve_group = [&](uint32_t item, GroupCache& cache) {
    if constexpr (kCompact && std::is_same_v<Policy, W2GroupPolicy>) {
      const int row = Policy::route_row(item);
      GroupContext ctx{};
      const uint32_t remaining = static_cast<uint32_t>(storage.route_indices[row]) & ~((1u << row) - 1u);
      ctx.member_count = min(__popc(remaining), kMaxGroup);
      ctx.members = storage.compact_w2_members[row];
      // Compact physical_item_at admits only chunk leaders. resolve and all
      // padded-column math retain their original member order and padding.
      Policy::resolve(args, item, args.indices[row], ctx);
      return ctx;
    } else if constexpr (kGroupMasks) return masked_group_plan<Policy>(args, item, storage.route_indices, cache);
    else return group_plan<Policy>(args, item, storage.route_indices, cache);
  };

  // Duplicate-only invocations exit before touching barriers or TMEM,
  // exactly like execute_fp4_grouped_pipeline.
  {
    bool any_leader = false;
    for (uint32_t item = item_begin; item < item_end; item = (kCompact ? next_item(item) : item + 1)) {
      const uint32_t physical_item = physical_item_at(item);
      if (selected(Policy::route_row(physical_item))) {
        any_leader = true;
        break;
      }
    }
    if (!any_leader) {
      return;
    }
  }

  TcgenTiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<128>{}, Int<8>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  using TmemAllocator = std::conditional_t<kReuseTmem,
      dspark_tmem::RoutedAllocator1Sm, dspark_tmem::Allocator1Sm>;
  TmemAllocator tmem_allocator{};
  const int warp = threadIdx.x / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kDgStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kDgStages>(
        storage.stage_free, kDgMmaWarps);
#if defined(DSPARK_W13_TMA_CANDIDATE)
    if constexpr (kUseTmaWeights) {
      cutlass::arch::detail::initialize_barrier_array_aligned<
          cutlass::arch::ClusterTransactionBarrier, kDgStages>(
          storage.weight_full, 1);
    }
#endif
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(
        storage.accumulator_full, kDgMmaWarps);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(
        storage.accumulator_free, kOutputTile);
#if defined(DSPARK_W13_SFA_CACHE)
    // sfa_group_ready[g] counts the 96 staging threads that wrote the four
    // stage blocks of scale group g; sfa_free counts the single
    // tcgen05.commit that retires the item's last UTCCP read of the buffer.
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kDgSfaCacheStages / 4>(
        storage.sfa_group_ready, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(storage.sfa_free, 1);
#endif
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  if constexpr (kReuseTmem)
    dspark_tmem::remember_routed_reservation(storage.tmem_base_ptr, kTmemColumns);
  tmem_accumulator.data() = storage.tmem_base_ptr;

  // TMEM map: accumulator slots at columns 0..7 / 8..15 (two item-paced
  // epilogue stages, DeepGEMM's kNumEpilogueStages = 2), SFA at 16..19,
  // SFB at 20..23. With a fused tile the accumulator block widens to
  // (epilogue stage, tile plane) and SFA gains one 4-column tile per plane;
  // SFB stays single because both planes share the activation columns.
  const uint32_t tmem_sfa = storage.tmem_base_ptr + kAccColumns;
  const uint32_t tmem_sfb = tmem_sfa + kSfaColumns;

#if (DSPARK_DESCRIPTOR_ARITHMETIC & 4)
  uint64_t weight_descriptor_base;
#if defined(DSPARK_W13_TMA_CANDIDATE)
  if constexpr (kUseTmaWeights) {
    weight_descriptor_base = dg_make_tma_weight_desc(storage.a[0].begin());
  } else
#endif
  {
    weight_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
        make_smem_ptr(reinterpret_cast<uint8_t*>(storage.a[0].begin())),
        layout<0>(TcgenSmemLayoutA{}))).desc_;
  }
  const uint64_t input_descriptor_base = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
      make_smem_ptr(reinterpret_cast<TcgenInputType*>(storage.b[0].begin())),
      layout<0>(TcgenSmemLayoutB{}))).desc_;
  const auto weight_descriptor_at = [=](int stage, int plane, int group) {
    return weight_descriptor_base
        + (static_cast<uint64_t>(stage) * kStageWeightBytes
           + plane * kPlaneWeightBytes + group * kMicroStageWeightBytes) / 16;
  };
  const auto input_descriptor_at = [=](int stage, int group) {
    return input_descriptor_base
        + (static_cast<uint64_t>(stage) * kDgScaleGroups + group)
            * kMicroStageInputBytes / 16;
  };
#else
  uint64_t
      weight_descriptor[kDgStages][kDgTilePairs][kDgScaleGroups];
  uint64_t input_descriptor[kDgStages][kDgScaleGroups];
  CUTE_UNROLL
  for (int s = 0; s < kDgStages; ++s) {
    CUTE_UNROLL
    for (int p = 0; p < kDgTilePairs; ++p) {
      uint8_t* const plane =
          reinterpret_cast<uint8_t*>(storage.a[s].begin())
          + p * kPlaneWeightBytes;
      CUTE_UNROLL
      for (int q = 0; q < kDgScaleGroups; ++q) {
        uint8_t* const window = plane + q * kMicroStageWeightBytes;
#if defined(DSPARK_W13_TMA_CANDIDATE)
        if constexpr (kUseTmaWeights) {
          weight_descriptor[s][p][q] = dg_make_tma_weight_desc(window);
        } else
#endif
        {
          weight_descriptor[s][p][q] =
              UMMA::make_umma_desc<UMMA::Major::K>(
                  make_tensor(
                      make_smem_ptr(window),
                      layout<0>(TcgenSmemLayoutA{}))).desc_;
        }
      }
    }
    CUTE_UNROLL
    for (int q = 0; q < kDgScaleGroups; ++q) {
      auto* const window = reinterpret_cast<TcgenInputType*>(
          reinterpret_cast<uint8_t*>(storage.b[s].begin())
          + q * kMicroStageInputBytes);
      input_descriptor[s][q] = UMMA::make_umma_desc<UMMA::Major::K>(
          make_tensor(
              make_smem_ptr(window),
              layout<0>(TcgenSmemLayoutB{}))).desc_;
    }
  }
  const auto weight_descriptor_at = [&](int stage, int plane, int group) {
    return weight_descriptor[stage][plane][group];
  };
  const auto input_descriptor_at = [&](int stage, int group) {
    return input_descriptor[stage][group];
  };
#endif
  const UMMA::InstrDescriptorBlockScaled base_instr_desc =
      UMMA::make_instr_desc_block_scaled<
          TcgenWeightType, TcgenInputType, float, cutlass::float_ue8m0_t,
          128, 8, UMMA::Major::K, UMMA::Major::K>();

  // Crew predicates. With the frozen mask (kDgCrew == 0) these fold to the
  // original `warp >= 5` / `warp == 4` / else split and nothing below changes.
  const bool mma_crew =
      kDgSplitMma ? (warp == 4 || warp == 7) : (warp == 4);
  const bool stage_crew = kDgWideStage
      ? (warp < 4 || (warp >= 5 && !mma_crew))
      : (warp >= (kDedicatedWeights ? 6 : 5));
  if (kDedicatedWeights && warp == 5) {
    // Independent progress; only lane0 issues existing weight/SFA copies.
    // The same stage_free commit protects both producers' disjoint regions.
    const int lane = static_cast<int>(threadIdx.x) % 32;
    int round = 0;
    GroupCache weight_cache;
    for (uint32_t item = item_begin; item < item_end; item = next_item(item)) {
      const uint32_t physical_item = physical_item_at(item);
      const GroupContext ctx = resolve_group(physical_item, weight_cache);
      if (ctx.member_count == 0) continue;
      const uint8_t* const weight = ctx.weight;
      const uint8_t* const weight_scales = ctx.weight_scales;
      const int output_base = ctx.output_base;
      const int planes = kCompact ? 1 : dg_live_planes(args, item);
#pragma unroll 1
      for (int stage = 0; stage < kStagesPerItem; ++stage, ++round) {
        const int s = round % kDgStages;
        if (round >= kDgStages)
          wait_barrier(storage.stage_free[s], (round / kDgStages - 1) & 1);
        // Weights: the candidate issues one packed-subbyte tensor-map load;
        // the retained body issues 1,024 scattered 8-byte cp.async copies.
        uint8_t* stage_a = reinterpret_cast<uint8_t*>(storage.a[s].begin());
#if defined(DSPARK_W13_TMA_CANDIDATE)
        if constexpr (kUseTmaWeights) {
          if (lane == 0 && !kProbeSkipWeights) {
#if defined(DSPARK_ROUTED_BULK_WEIGHTS)
            if constexpr (kPackedSfa == 2 && kDgChunks == 16) {
              cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                  &storage.weight_full[s],
                  planes * (kPlaneWeightBytes / 2 + kDgScaleGroups * 512));
              for (int p = 0; p < planes; ++p) {
                const int tile = output_base / kOutputTile + p;
                const uint64_t byte_offset = static_cast<uint64_t>(weight - args.weight_arena)
                    + (static_cast<uint64_t>(tile) * kStagesPerItem + stage)
                        * (kPlaneWeightBytes / 2);
                // The 16U4 tensor map expands each packed 8-byte group into
                // its 16-byte MMA slot and applies the original 128B swizzle.
                // A raw bulk copy cannot perform either transformation.
                cute::SM90_TMA_LOAD_3D::copy(
                    weight_tma_descriptor, &storage.weight_full[s],
                    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                    stage_a + p * kPlaneWeightBytes, 0, 0,
                    static_cast<int32_t>(byte_offset / 256));
                const auto* source_sfa = weight_scales
                    + (static_cast<int64_t>(tile) * (kKBlocks / 4)
                       + stage * kDgScaleGroups) * 512;
                cute::SM90_BULK_COPY_G2S::copy(
                    source_sfa, &storage.weight_full[s], storage.sfa[s][p],
                    kDgScaleGroups * 512);
              }
            } else
#endif
            {
            for (int p = 0; p < planes; ++p) {
              CUTE_UNROLL
              for (int q = 0; q < kDgScaleGroups; ++q) {
                dg_tma_load_weight<Policy, kStageMajorWeights>(
                    args,
                    ctx,
                    weight_tma_descriptor,
                    stage * kDgScaleGroups + q,
                    stage_a + p * kPlaneWeightBytes
                        + q * kMicroStageWeightBytes,
                    &storage.weight_full[s],
                    p * kOutputTile,
                    (p == 0 && q == 0)
                        ? planes * kOutputTile * kDgBlockK / 2
                            + (kPackedSfa == 2 ? planes * kDgScaleGroups * 512 : 0)
                        : 0);
              }
              if constexpr (kPackedSfa == 2) {
                // Offline-transposed scales are contiguous in the UTCCP
                // layout. Join their bulk copy to the weight transaction.
                const auto* source_sfa = weight_scales
                    + ((static_cast<int64_t>(output_base / kOutputTile + p)
                        * (kKBlocks / 4)) + stage * kDgScaleGroups) * 512;
                cute::SM90_BULK_COPY_G2S::copy(
                    source_sfa, &storage.weight_full[s], storage.sfa[s][p],
                    kDgScaleGroups * 512);
              }
            }
            }
          }
        } else
#endif
        if constexpr (!kProbeSkipWeights) {
          constexpr int kWeightPackets = kDgBlockK / 16;
          for (int p = 0; p < kDgTilePairs; ++p) {
            if (p >= planes) {
              break;
            }
            const int plane_base = output_base + p * kOutputTile;
            uint8_t* const plane_a = stage_a + p * kPlaneWeightBytes;
            for (int ci = lane; ci < kOutputTile * kWeightPackets;
                 ci += kStagingThreads) {
              const int row = ci / kWeightPackets;
              const int g = ci % kWeightPackets;
              const uint8_t* source = weight
                  + static_cast<int64_t>(plane_base + row) * (kKWidth / 2)
                  + stage * (kDgBlockK / 2) + g * 8;
              cutlass::arch::cp_async<
                  8, cutlass::arch::CacheOperation::Always>(
                  plane_a + (g >> 1) * 4096 + (g & 1) * 2048 + row * 16,
                  source);
            }
          }
        }

      }
    }
  } else if (stage_crew) {
    // Staging crew: weights (8 KB), member activation slices, and both
    // transposed SF blocks ride one cp.async ring slot per 128-K stage.
    // Lane order keeps warps 5.. at the head of the range, so the frozen
    // build's lane -> packet map is bit-for-bit the one R9 measured.
    const int lane = kDgWideStage
        ? (warp >= 5
               ? static_cast<int>(threadIdx.x) - 5 * 32
               : static_cast<int>(threadIdx.x) + (kDgSplitMma ? 2 * 32 : 3 * 32))
        : (static_cast<int>(threadIdx.x) - (kDedicatedWeights ? 6 : 5) * 32);
    const bool drain_crew = kDgWideStage && warp < 4;
    int round = 0;
#if defined(DSPARK_W13_SFA_CACHE)
    int leader = 0;
#endif
#if defined(DSPARK_W13_CREW_PROBE)
    unsigned long long probe_stall = 0;
    unsigned long long probe_weights = 0;
    unsigned long long probe_input = 0;
    unsigned long long probe_scales = 0;
    unsigned long long probe_drain = 0;
    unsigned long long probe_rounds = 0;
    const unsigned long long probe_start = DG_PROBE_NOW();
#endif
    // Wide-stage drain state. Every use is inside `if constexpr
    // (kDgWideStage)`, so the frozen build never materializes these.
    GroupContext pending_ctx{};
    int pending_planes = 0;
    int pending_leader = 0;
    bool pending = false;
    int drain_leader = 0;
    GroupCache group_cache;
    // The once-per-item drain, run by warps 0..3 from inside the staging
    // loop. This is a SEPARATE COPY of the frozen accumulator-crew body
    // rather than a shared helper, so that the kDgCrew == 0 build below is
    // textually untouched and stays byte-identical.
    auto wide_drain = [&](const GroupContext& ctx, int e, int planes,
                          int drained) {
      // Compiled out entirely in the frozen build, so its mere presence
      // cannot move a register or an instruction there.
      if constexpr (!kDgWideStage) {
        return;
      } else {
      auto tmem_to_register = make_tmem_copy(
          SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
      auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
      auto thread_source = [&](int slot_index) {
        auto slot = tmem_accumulator;
        slot.data() = storage.tmem_base_ptr + slot_index * 8;
        return thread_copy.partition_S(slot);
      };
      auto partial_layout = make_layout(
          make_shape(Int<128>{}, Int<8>{}),
          make_stride(Int<1>{}, Int<128>{}));
      auto thread_partial = [&](int stage_index, int p) {
        return thread_copy.partition_D(cta_mma.partition_C(make_tensor(
            make_smem_ptr(&storage.partial[stage_index][p][0]),
            partial_layout)));
      };
      auto register_accumulator =
          make_tensor<float>(shape(thread_partial(0, 0)));
      wait_barrier(storage.accumulator_full[e], (drained / 2) & 1);
      CUTE_UNROLL
      for (int p = 0; p < kDgTilePairs; ++p) {
        if (p >= planes) {
          break;
        }
        copy(
            tmem_to_register,
            thread_source(e * kDgTilePairs + p),
            register_accumulator);
        copy(register_accumulator, thread_partial(e, p));
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[e]);
      // Join the 128 drain threads while the other staging warps continue.
      // The pinned CUTLASS wrapper adds eight to user barrier IDs: this is
      // hardware barrier 9; user ID 0 would be hardware barrier 8. Both are
      // separate from the hardware barrier 0 used by __syncthreads().
      cutlass::arch::NamedBarrier::sync(4 * 32, 1);
      for (int p = 0; p < kDgTilePairs; ++p) {
        if (p >= planes) {
          break;
        }
        const int plane_base = ctx.output_base + p * kOutputTile;
        for (int m = 0; m < ctx.member_count; ++m) {
          __nv_bfloat16* store = Policy::store_row(
              args, ctx, group_member(ctx.members, m));
#ifdef DSPARK_ROUTED_INTERLEAVE
          if constexpr (std::is_same_v<Policy, W13GroupPolicy>) {
            const int lane = threadIdx.x;
            const int column = plane_base / 2 + lane % 64;
            const int branch = lane / 64;
            args.routed_w13[(group_member(ctx.members, m) * 2 + branch)
                               * kIntermediate + column] =
                __float2bfloat16_rn(storage.partial[e][p][m * 128 + lane]);
          } else
#endif
          {
          store[plane_base + static_cast<int>(threadIdx.x)] =
              __float2bfloat16_rn(
                  storage.partial[e][p][m * 128 + threadIdx.x]);
          }
        }
      }
      }
    };
    (void)wide_drain;
    (void)drain_crew;
    for (uint32_t item = item_begin; item < item_end; item = next_item(item)) {
      const uint32_t physical_item = physical_item_at(item);
      const GroupContext ctx =
          resolve_group(physical_item, group_cache);
      if (ctx.member_count == 0) {
        continue;
      }
#ifdef DSPARK_ROUTED_GROUP_READY
      if constexpr (!std::is_same_v<Policy, W13GroupPolicy>) {
        const uint32_t* ready = args.group_ready + group_member(ctx.members, 0);
        uint32_t completed;
        do {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                       : "=r"(completed) : "l"(ready) : "memory");
          if (completed != 16) __nanosleep(32);
        } while (completed != 16);
      }
#endif
      const uint8_t* const weight = ctx.weight;
      const uint8_t* const weight_scales = ctx.weight_scales;
      const int output_base = ctx.output_base;
      const int planes = kCompact ? 1 : dg_live_planes(args, item);
#if defined(DSPARK_W13_SFA_CACHE)
      // Whole-line SFA cache. The item's 128 scale lines are read ONCE with
      // 16-byte loads and transposed in registers into every UTCCP block
      // they feed, instead of being re-gathered four bytes at a time on
      // every stage.
      //
      // The work is SPREAD over the item's own rounds -- scale group g on
      // round 2g -- rather than done in one burst at the item boundary. A
      // burst measured +4.5% on W13 and +15.6% on W2 even though it cut the
      // SFA tag requests and shared wavefronts by 32x: the staging crew
      // stops feeding the ring for the length of the burst, and the MMA
      // crew drains it. Group g is consumed by stages 4g..4g+3, so filling
      // it on round 2g is always early enough, and the 2x spacing also
      // keeps the staging crew from lapping a group barrier (it runs at
      // most kDgStages rounds ahead, and kStagesPerItem - 2*(kSfaGroups-1)
      // exceeds that for both policies).
      constexpr int kSfaGroups = kStagesPerItem / 4;   // 16 B = 4 stages
      static_assert(kStagesPerItem % 4 == 0);
      static_assert(kStagesPerItem <= kDgSfaCacheStages);
      static_assert(kOutputTile == 128, "row->word transpose assumes 128");
      static_assert(
          kStagesPerItem - 2 * (kSfaGroups - 1) > kDgStages,
          "a group barrier must not be re-armed before it is waited on");
      const int sfa_buffer = leader & 1;
      if (leader >= 2) {
        wait_barrier(storage.sfa_free[sfa_buffer], (leader / 2 - 1) & 1);
      }
      ++leader;
#endif
#if DSPARK_W13_DG_STAGE_UNROLL == 2
#pragma unroll 2
#endif
      for (int stage = 0; stage < kStagesPerItem; ++stage, ++round) {
        const int s = round % kDgStages;
#if defined(DSPARK_W13_CREW_PROBE)
        unsigned long long probe_mark = DG_PROBE_NOW();
#endif
        if (round >= kDgStages) {
          wait_barrier(
              storage.stage_free[s], (round / kDgStages - 1) & 1);
        }
#if defined(DSPARK_W13_CREW_PROBE)
        probe_stall += DG_PROBE_NOW() - probe_mark;
        ++probe_rounds;
        probe_mark = DG_PROBE_NOW();
#endif
        if constexpr (!kDedicatedWeights) {
        // Weights: the candidate issues one packed-subbyte tensor-map load;
        // the retained body issues 1,024 scattered 8-byte cp.async copies.
        uint8_t* stage_a = reinterpret_cast<uint8_t*>(storage.a[s].begin());
#if defined(DSPARK_W13_TMA_CANDIDATE)
        if constexpr (kUseTmaWeights) {
          if (lane == 0 && !kProbeSkipWeights) {
#if defined(DSPARK_ROUTED_BULK_WEIGHTS)
            if constexpr (kPackedSfa == 2 && kDgChunks == 16) {
              cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                  &storage.weight_full[s],
                  planes * (kPlaneWeightBytes / 2 + kDgScaleGroups * 512));
              for (int p = 0; p < planes; ++p) {
                const int tile = output_base / kOutputTile + p;
                const uint64_t byte_offset = static_cast<uint64_t>(weight - args.weight_arena)
                    + (static_cast<uint64_t>(tile) * kStagesPerItem + stage)
                        * (kPlaneWeightBytes / 2);
                // The 16U4 tensor map expands each packed 8-byte group into
                // its 16-byte MMA slot and applies the original 128B swizzle.
                // A raw bulk copy cannot perform either transformation.
                cute::SM90_TMA_LOAD_3D::copy(
                    weight_tma_descriptor, &storage.weight_full[s],
                    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                    stage_a + p * kPlaneWeightBytes, 0, 0,
                    static_cast<int32_t>(byte_offset / 256));
                const auto* source_sfa = weight_scales
                    + (static_cast<int64_t>(tile) * (kKBlocks / 4)
                       + stage * kDgScaleGroups) * 512;
                cute::SM90_BULK_COPY_G2S::copy(
                    source_sfa, &storage.weight_full[s], storage.sfa[s][p],
                    kDgScaleGroups * 512);
              }
            } else
#endif
            {
            for (int p = 0; p < planes; ++p) {
              CUTE_UNROLL
              for (int q = 0; q < kDgScaleGroups; ++q) {
                dg_tma_load_weight<Policy, kStageMajorWeights>(
                    args,
                    ctx,
                    weight_tma_descriptor,
                    stage * kDgScaleGroups + q,
                    stage_a + p * kPlaneWeightBytes
                        + q * kMicroStageWeightBytes,
                    &storage.weight_full[s],
                    p * kOutputTile,
                    (p == 0 && q == 0)
                        ? planes * kOutputTile * kDgBlockK / 2
                            + (kPackedSfa == 2 ? planes * kDgScaleGroups * 512 : 0)
                        : 0);
              }
              if constexpr (kPackedSfa == 2) {
                // Offline-transposed scales are contiguous in the UTCCP
                // layout. Join their bulk copy to the weight transaction.
                const auto* source_sfa = weight_scales
                    + ((static_cast<int64_t>(output_base / kOutputTile + p)
                        * (kKBlocks / 4)) + stage * kDgScaleGroups) * 512;
                cute::SM90_BULK_COPY_G2S::copy(
                    source_sfa, &storage.weight_full[s], storage.sfa[s][p],
                    kDgScaleGroups * 512);
              }
            }
            }
          }
        } else
#endif
        if constexpr (!kProbeSkipWeights) {
          constexpr int kWeightPackets = kDgBlockK / 16;
          for (int p = 0; p < kDgTilePairs; ++p) {
            if (p >= planes) {
              break;
            }
            const int plane_base = output_base + p * kOutputTile;
            uint8_t* const plane_a = stage_a + p * kPlaneWeightBytes;
            for (int ci = lane; ci < kOutputTile * kWeightPackets;
                 ci += kStagingThreads) {
              const int row = ci / kWeightPackets;
              const int g = ci % kWeightPackets;
              const uint8_t* source = weight
                  + static_cast<int64_t>(plane_base + row) * (kKWidth / 2)
                  + stage * (kDgBlockK / 2) + g * 8;
              cutlass::arch::cp_async<
                  8, cutlass::arch::CacheOperation::Always>(
                  plane_a + (g >> 1) * 4096 + (g & 1) * 2048 + row * 16,
                  source);
            }
          }
        }
        }
#if defined(DSPARK_W13_CREW_PROBE)
        probe_weights += DG_PROBE_NOW() - probe_mark;
        probe_mark = DG_PROBE_NOW();
#endif
        // Member activation columns: two sixteen-byte pieces per K=32 chunk
        // (chunk j at j*256, half h at h*128, member column at m*16).
        // Lane->packet map is bank-conflict aware, same argument as the LM
        // A-ring (dspark_lm_phase.cuh execute_umma_bf16): member = ci >> 3
        // would give a quarter-warp one member's eight pieces, whose SMEM
        // offsets are 128 B apart and therefore all in the SAME four banks
        // (8-way store conflict). Interleaving two bits of member below the
        // chunk index spreads the quarter-warp over four members (2-way, the
        // floor) while keeping a member's two 16-byte halves adjacent, so the
        // global 32-byte sectors are unchanged. The walk now covers the full
        // kMaxGroup x 8 space with the tail predicated off; the SAME packets
        // reach the SAME SMEM addresses, so this is bitwise by construction.
        constexpr int kInputPieces = kDgChunks * 2;
        constexpr int kInputPieceBits = kDgChunks == 4 ? 3 : (kDgChunks == 8 ? 4 : 5);
        constexpr int kMemberPackets = kMaxGroup * kInputPieces;
        static_assert(kMaxGroup == 8, "member map assumes an 8-member group");
        uint8_t* stage_b = reinterpret_cast<uint8_t*>(storage.b[s].begin());
        for (int ci = (kProbeSkipInput ? kMemberPackets : lane);
             ci < kMemberPackets;
             ci += kStagingThreads) {
          const unsigned packet = static_cast<unsigned>(ci);
          const int member = static_cast<int>(
              ((packet >> (kInputPieceBits + 2)) << 2)
              | ((packet >> 1) & 3u));
          if (member >= ctx.member_count) {
            continue;
          }
          const int piece =
              static_cast<int>(
                  (((packet >> 3) & (kInputPieces / 2 - 1)) << 1)
                  | (packet & 1u));
          const uint8_t* source = reinterpret_cast<const uint8_t*>(
              Policy::input_row(args, group_member(ctx.members, member)))
              + stage * kDgBlockK + piece * 16;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              stage_b + (piece >> 1) * 256 + (piece & 1) * 128
                  + member * 16,
              source);
        }
#if defined(DSPARK_W13_CREW_PROBE)
        probe_input += DG_PROBE_NOW() - probe_mark;
        probe_mark = DG_PROBE_NOW();
#endif
#if !defined(DSPARK_W13_SFA_CACHE)
        if constexpr (kPackedSfa == 1 || (kPackedSfa == 2 && !kUseTmaWeights)) {
          for (int p = 0; p < kDgTilePairs; ++p) {
            if (p >= planes) {
              break;
            }
            const auto* source_sfa = weight_scales
                + ((static_cast<int64_t>(output_base / kOutputTile + p)
                    * (kKBlocks / 4)) + stage * kDgScaleGroups) * 512;
            auto* destination = reinterpret_cast<uint8_t*>(storage.sfa[s][p]);
            for (int packet = lane; packet < kDgScaleGroups * 32;
                 packet += kStagingThreads) {
              cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
                  destination + packet * 16, source_sfa + packet * 16);
            }
          }
        } else if constexpr (kPackedSfa == 0) {
          // SFA: one transposed uint32 per output row and K=128 micro-stage.
          for (int p = 0; p < kDgTilePairs; ++p) {
            if (p >= planes) {
              break;
            }
            const int plane_base = output_base + p * kOutputTile;
            CUTE_UNROLL
            for (int q = 0; q < kDgScaleGroups; ++q) {
              uint8_t* stage_sfa = reinterpret_cast<uint8_t*>(
                  storage.sfa[s][p][q]);
              for (int ci = (kProbeSkipSfa ? kOutputTile : lane);
                   ci < kOutputTile; ci += kStagingThreads) {
                const int m = (ci & 3) * 32 + (ci >> 2);
                const uint8_t* source = weight_scales
                    + (plane_base + m) * kKBlocks
                    + stage * kDgChunks + q * 4;
                cutlass::arch::cp_async<
                    4, cutlass::arch::CacheOperation::Always>(
                    stage_sfa + ci * 4, source);
              }
            }
          }
        }
#endif
        // SFB: K=128 selects one byte across four ring stages; K=512 uses all
        // four bytes in the current ring stage.
        uint8_t* stage_sfb = reinterpret_cast<uint8_t*>(storage.sfb[s]);
        for (int ci = (kProbeSkipSfb ? ctx.member_count : lane);
             ci < ctx.member_count; ci += kStagingThreads) {
          const uint8_t* source =
              Policy::input_scales_row(args, group_member(ctx.members, ci))
              + ((stage * kDgScaleGroups) & ~3);
          cutlass::arch::cp_async<4, cutlass::arch::CacheOperation::Always>(
              stage_sfb + ci * 16, source);
        }
#if defined(DSPARK_W13_SFA_CACHE)
        // Scale group g = stage/2 on the even rounds. Lane -> row: one row
        // per lane per pass, 16 contiguous global bytes each. The shared
        // scatter's bank is 4*(row%8) + (row>>5), independent of the stage,
        // so 32 consecutive rows cover 8 banks (4-way). Reading a whole row
        // with one quarter-warp would cut the tag requests to 4 but make
        // the store 8-way; one row per lane is the other extreme. This walk
        // sits between them and is the minimum of the two costs.
        if ((stage & 1) == 0 && (stage >> 1) < kSfaGroups) {
          const int group_base = stage >> 1;
          uint32_t* const buffer = &storage.sfa_item[sfa_buffer][0][0];
          for (int row = (kProbeSkipSfa ? kOutputTile : lane);
               row < kOutputTile; row += kStagingThreads) {
            const uint4 line = *reinterpret_cast<const uint4*>(
                weight_scales
                + static_cast<int64_t>(output_base + row) * kKBlocks
                + group_base * 16);
            // Transposed word index: block word l*4+i holds row i*32+l.
            uint32_t* const destination = buffer + (row & 31) * 4
                + (row >> 5) + group_base * 4 * kOutputTile;
            destination[0 * kOutputTile] = line.x;
            destination[1 * kOutputTile] = line.y;
            destination[2 * kOutputTile] = line.z;
            destination[3 * kOutputTile] = line.w;
          }
          // Plain shared stores: cpasync_barrier_arrive_noinc orders
          // cp.async traffic only, so release these into the async proxy the
          // UTCCP reads through and arrive on the group's own mbarrier.
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::ClusterBarrier::arrive(
              &storage.sfa_group_ready[group_base]);
        }
#endif
        cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
#if defined(DSPARK_W13_CREW_PROBE)
        probe_scales += DG_PROBE_NOW() - probe_mark;
        probe_mark = DG_PROBE_NOW();
#endif
        // The drain of the PREVIOUS item, run here instead of on four
        // dedicated warps. kDgDrainStage rounds into this item the ring's own
        // stage_free handshake has already proven that the MMA crew retired
        // the previous item's last stage (staging never runs more than
        // kDgStages rounds ahead), so accumulator_full is already set and the
        // wait below is a formality rather than a stall.
        if constexpr (kDgWideStage) {
          if (drain_crew && pending && stage == kDgDrainStage) {
            wide_drain(
                pending_ctx, pending_leader & 1, pending_planes,
                pending_leader);
            pending = false;
          }
        }
#if defined(DSPARK_W13_CREW_PROBE)
        probe_drain += DG_PROBE_NOW() - probe_mark;
#endif
      }
      if constexpr (kDgWideStage) {
        pending_ctx = ctx;
        pending_planes = planes;
        pending_leader = drain_leader;
        pending = true;
        ++drain_leader;
      }
    }
    if constexpr (kDgWideStage) {
#if defined(DSPARK_W13_CREW_PROBE)
      const unsigned long long probe_tail = DG_PROBE_NOW();
#endif
      if (drain_crew && pending) {
        wide_drain(
            pending_ctx, pending_leader & 1, pending_planes, pending_leader);
      }
#if defined(DSPARK_W13_CREW_PROBE)
      probe_drain += DG_PROBE_NOW() - probe_tail;
#endif
    }
#if defined(DSPARK_W13_CREW_PROBE)
    dg_probe_add(0, 0, DG_PROBE_NOW() - probe_start);
    dg_probe_add(0, 1, probe_stall);
    dg_probe_add(0, 2, probe_weights);
    dg_probe_add(0, 3, probe_input);
    dg_probe_add(0, 4, probe_scales);
    dg_probe_add(0, 5, probe_drain);
    dg_probe_add(0, 6, 1);
    dg_probe_add(0, 7, probe_rounds);
#endif
  } else if (mma_crew) {
    // MMA crew: per stage, refresh the SF TMEM (tcgen05.cp executes in
    // issue order with the MMAs, DeepGEMM's per-k-block pattern) and
    // chain 4 block-scaled UMMAs into the item's accumulator slot.
    int round = 0;
    int leader = 0;
    // With kDgSplitMma the fused tile's planes are dealt to two warps: warp 4
    // issues plane 0, warp 7 plane 1. Each warp copies its own SFB tile, so
    // no warp ever reads a TMEM scale block another warp wrote (tcgen05 ops
    // are ordered within the issuing warp, not across warps). Both folds to
    // the frozen single-warp body when the mask is 0.
    const int plane_begin = kDgSplitMma ? (warp == 4 ? 0 : 1) : 0;
    constexpr int plane_step = kDgSplitMma ? 2 : 1;
    const uint32_t tmem_sfb_warp_base =
        tmem_sfb + ((kDgSplitMma && warp != 4) ? 4 : 0);
#if defined(DSPARK_W13_CREW_PROBE)
    unsigned long long probe_stall_full = 0;
    unsigned long long probe_stall_acc = 0;
    unsigned long long probe_issue = 0;
    unsigned long long probe_rounds = 0;
    const unsigned long long probe_start = DG_PROBE_NOW();
#endif
    for (uint32_t item = item_begin; item < item_end; item = next_item(item)) {
      const uint32_t physical_item = physical_item_at(item);
      if (!selected(Policy::route_row(physical_item))) {
        continue;
      }
      const int e = leader & 1;
      const int planes = kCompact ? 1 : dg_live_planes(args, item);
      (void)planes;
#if DSPARK_W13_DG_STAGE_UNROLL == 2
#pragma unroll 2
#endif
      for (int stage = 0; stage < kStagesPerItem; ++stage, ++round) {
        const int s = round % kDgStages;
#if defined(DSPARK_W13_CREW_PROBE)
        unsigned long long probe_mark = DG_PROBE_NOW();
#endif
        wait_barrier(storage.stage_full[s], (round / kDgStages) & 1);
#if defined(DSPARK_W13_CREW_PROBE)
        probe_stall_full += DG_PROBE_NOW() - probe_mark;
        ++probe_rounds;
#endif
#if defined(DSPARK_W13_SFA_CACHE)
        // Acquire the four stage blocks of this scale group. The matching
        // release is the staging crew's async-proxy fence plus 96 arrivals
        // on the same group barrier, one item earlier at the latest.
        if ((stage & 3) == 0) {
          wait_barrier(storage.sfa_group_ready[stage >> 2], leader & 1);
        }
#endif
#if defined(DSPARK_W13_TMA_CANDIDATE)
        if constexpr (kUseTmaWeights && !kProbeSkipWeights) {
          wait_barrier(storage.weight_full[s], (round / kDgStages) & 1);
        }
#endif
#if defined(DSPARK_W13_CREW_PROBE)
        probe_mark = DG_PROBE_NOW();
#endif
        if (stage == 0 && leader >= 2) {
          wait_barrier(
              storage.accumulator_free[e], (leader / 2 - 1) & 1);
        }
#if defined(DSPARK_W13_CREW_PROBE)
        probe_stall_acc += DG_PROBE_NOW() - probe_mark;
        probe_mark = DG_PROBE_NOW();
#endif
        const uint32_t tmem_sfb_warp = tmem_sfb_warp_base
            + (kSfRing == 2 ? s * kDgMmaWarps * 4 : 0);
        const auto sfa_address = [&](int p, int q) {
          return tmem_sfa + ((s % kSfStages) * kDgTilePairs + p) * kSfGroups * 4
              + (kSfRing != 0 ? q * 4 : 0);
        };
        if (elect_one_sync()) {
          // SFB words repeat within a 4-stage scale group (sf_id walks the
          // bytes). K=512 consumes all four bytes in one ring stage, so it
          // refreshes the word every turn; K=128 retains the old cadence.
          if (kDgScaleGroups > 1 || (stage & 3) == 0) {
            dg_utccp_32x128b_warpx4(
                dg_make_sf_desc(storage.sfb[s]), tmem_sfb_warp);
          }
          if constexpr (kSfRing != 0) {
            // Distinct slots avoid overwriting a scale block still read by
            // earlier MMAs. The stage barrier already made every SFA visible.
            CUTE_UNROLL
            for (int q = 0; q < kDgScaleGroups; ++q) {
              for (int p = plane_begin; p < kDgTilePairs; p += plane_step) {
                if (p < planes)
                  dg_utccp_32x128b_warpx4(
                      dg_make_sf_desc(storage.sfa[s][p][q]), sfa_address(p, q));
              }
            }
          }
          // A K=512 ring stage contains four of the K=128 micro-stages used
          // by tcgen05's block-scaled MMA. Refresh the scale tiles immediately
          // before the four MMAs that consume them; this preserves the proven
          // K=128 issue ordering while cutting ring/barrier traffic by 4x.
          CUTE_UNROLL
          for (int q = 0; q < kDgScaleGroups; ++q) {
            for (int p = plane_begin; p < kDgTilePairs; p += plane_step) {
              if (p >= planes) {
                break;
              }
#if defined(DSPARK_W13_SFA_CACHE)
              dg_utccp_32x128b_warpx4(
                  dg_make_sf_desc(&storage.sfa_item[e][stage][0]),
                  tmem_sfa + p * 4);
#else
              if constexpr (kSfRing == 0) {
                dg_utccp_32x128b_warpx4(
                    dg_make_sf_desc(storage.sfa[s][p][q]), sfa_address(p, q));
              }
#endif
            }
            // Plane p accumulates into its own TMEM slot, so the two planes
            // are independent chains. Each q window is the original four-
            // chunk K=128 MMA body with descriptors advanced by 128 K.
            for (int p = plane_begin; p < kDgTilePairs; p += plane_step) {
              if (p >= planes) {
                break;
              }
              CUTE_UNROLL
              for (int j_local = 0; j_local < 4; ++j_local) {
#if defined(DSPARK_W13_TMA_CANDIDATE)
                const uint64_t weight_chunk_offset =
                    kUseTmaWeights
                    ? static_cast<uint64_t>(j_local * 2)
                    : static_cast<uint64_t>(j_local * 256);
#else
                const uint64_t weight_chunk_offset =
                    static_cast<uint64_t>(j_local * 256);
#endif
                dg_mma_mxf8f6f4(
                    weight_descriptor_at(s, p, q) + weight_chunk_offset,
                    input_descriptor_at(s, q) + j_local * 16,
                    storage.tmem_base_ptr + (e * kDgTilePairs + p) * 8,
                    (stage > 0 || q > 0 || j_local > 0) ? 1u : 0u,
                    dg_runtime_instr_desc(
                        base_instr_desc,
                        static_cast<uint32_t>(j_local),
                        static_cast<uint32_t>(
                            (stage * kDgScaleGroups + q) & 3)),
                    sfa_address(p, q), tmem_sfb_warp);
              }
            }
          }
        }
        __syncwarp();
        cutlass::arch::umma_arrive(&storage.stage_free[s]);
        if (stage == kStagesPerItem - 1) {
          cutlass::arch::umma_arrive(&storage.accumulator_full[e]);
#if defined(DSPARK_W13_SFA_CACHE)
          // tcgen05.commit: signals only once every UTCCP this warp issued
          // has retired, so the staging crew cannot overwrite the buffer
          // out from under an in-flight scale copy.
          cutlass::arch::umma_arrive(&storage.sfa_free[e]);
#endif
        }
#if defined(DSPARK_W13_CREW_PROBE)
        probe_issue += DG_PROBE_NOW() - probe_mark;
#endif
      }
      ++leader;
    }
    // Each MMA crew observes every final acknowledgement it submitted.
    // 'round' and 'leader' count only actual consumed stages/items, excluding
    // skipped duplicate groups. Do not assume ordering across commit barriers.
    if ((threadIdx.x & 31) == 0) {
      CUTE_UNROLL
      for (int slot = 0; slot < kDgStages; ++slot) {
        if (slot < round) {
          const int last = slot + ((round - 1 - slot) / kDgStages) * kDgStages;
          wait_barrier(storage.stage_free[slot], (last / kDgStages) & 1);
        }
      }
      CUTE_UNROLL
      for (int acc = 0; acc < 2; ++acc) {
        if (acc < leader) {
          const int last = acc + ((leader - 1 - acc) / 2) * 2;
          wait_barrier(storage.accumulator_free[acc], (last / 2) & 1);
#if defined(DSPARK_W13_SFA_CACHE)
          wait_barrier(storage.sfa_free[acc], (last / 2) & 1);
#endif
        }
      }
    }
#if defined(DSPARK_W13_CREW_PROBE)
    dg_probe_add(1, 0, DG_PROBE_NOW() - probe_start);
    dg_probe_add(1, 1, probe_stall_full);
    dg_probe_add(1, 2, probe_stall_acc);
    dg_probe_add(1, 3, probe_issue);
    dg_probe_add(1, 6, 1);
    dg_probe_add(1, 7, probe_rounds);
#endif
  } else if (!kDgWideStage) {
    // Accumulator crew: ONE drain per item. The TC already applied every
    // scale, so the drain is a TMEM read plus BF16 stores per member.
    // (kDgWideStage moves this body into the staging loop above; the guard
    // keeps it out of the register allocation of that build.)
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int slot_index) {
      auto slot = tmem_accumulator;
      slot.data() = storage.tmem_base_ptr + slot_index * 8;
      return thread_copy.partition_S(slot);
    };
    auto partial_layout = make_layout(
        make_shape(Int<128>{}, Int<8>{}),
        make_stride(Int<1>{}, Int<128>{}));
    // One destination view per (epilogue stage, tile plane) slot. The slot
    // index is a runtime value only through the shared-memory base pointer;
    // the partition itself is static, exactly as in the unfused body.
    auto thread_partial = [&](int e, int p) {
      return thread_copy.partition_D(cta_mma.partition_C(make_tensor(
          make_smem_ptr(&storage.partial[e][p][0]), partial_layout)));
    };
    auto register_accumulator = make_tensor<float>(shape(thread_partial(0, 0)));

    int leader = 0;
    GroupCache group_cache;
#if defined(DSPARK_W13_CREW_PROBE)
    unsigned long long probe_stall_acc = 0;
    unsigned long long probe_tmem = 0;
    unsigned long long probe_stores = 0;
    unsigned long long probe_items = 0;
    const unsigned long long probe_start = DG_PROBE_NOW();
#endif
    for (uint32_t item = item_begin; item < item_end; item = next_item(item)) {
      const uint32_t physical_item = physical_item_at(item);
      const GroupContext ctx =
          resolve_group(physical_item, group_cache);
      if (ctx.member_count == 0) {
        continue;
      }
      const int e = leader & 1;
      const int planes = kCompact ? 1 : dg_live_planes(args, item);
#if defined(DSPARK_W13_CREW_PROBE)
      unsigned long long probe_mark = DG_PROBE_NOW();
      ++probe_items;
#endif
      wait_barrier(storage.accumulator_full[e], (leader / 2) & 1);
#if defined(DSPARK_W13_CREW_PROBE)
      probe_stall_acc += DG_PROBE_NOW() - probe_mark;
      probe_mark = DG_PROBE_NOW();
#endif
      CUTE_UNROLL
      for (int p = 0; p < kDgTilePairs; ++p) {
        if (p >= planes) {
          break;
        }
        copy(
            tmem_to_register,
            thread_source(e * kDgTilePairs + p),
            register_accumulator);
        copy(register_accumulator, thread_partial(e, p));
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[e]);
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
#if defined(DSPARK_W13_CREW_PROBE)
      probe_tmem += DG_PROBE_NOW() - probe_mark;
      probe_mark = DG_PROBE_NOW();
#endif
      for (int p = 0; p < kDgTilePairs; ++p) {
        if (p >= planes) {
          break;
        }
        const int plane_base = ctx.output_base + p * kOutputTile;
        for (int m = 0; m < ctx.member_count; ++m) {
          __nv_bfloat16* store = Policy::store_row(
              args, ctx, group_member(ctx.members, m));
#ifdef DSPARK_ROUTED_INTERLEAVE
          if constexpr (std::is_same_v<Policy, W13GroupPolicy>) {
            const int lane = threadIdx.x;
            const int column = plane_base / 2 + lane % 64;
            const int branch = lane / 64;
            args.routed_w13[(group_member(ctx.members, m) * 2 + branch)
                               * kIntermediate + column] =
                __float2bfloat16_rn(storage.partial[e][p][m * 128 + lane]);
          } else
#endif
          {
          store[plane_base + static_cast<int>(threadIdx.x)] =
              __float2bfloat16_rn(
                  storage.partial[e][p][m * 128 + threadIdx.x]);
          }
        }
      }
#ifdef DSPARK_ROUTED_FUSED_SWIGLU
      if constexpr (kFuseSwiglu) {
        if (ctx.branch == 1) {
          // The preceding gate drain occupies slot 0; this up drain is slot 1.
          const int lane = threadIdx.x % 32;
          cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
          for (int m = warp; m < ctx.member_count; m += 4) {
            const int row = group_member(ctx.members, m);
            const int base = row * kIntermediate + ctx.output_base;
            float values[4];
            float maximum = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const int column = lane + p * 32;
              float gate = __bfloat162float(
                  __float2bfloat16_rn(storage.partial[0][0][m * 128 + column]));
              float up = __bfloat162float(
                  __float2bfloat16_rn(storage.partial[1][0][m * 128 + column]));
              gate = fminf(gate, 10.0f);
              up = fminf(fmaxf(up, -10.0f), 10.0f);
              float value = gate / (1.0f + expf(-gate)) * up;
              value *= args.route_weights[row];
              const auto rounded = __float2bfloat16_rn(value);
              args.swiglu[base + column] = rounded;
              values[p] = __bfloat162float(rounded);
              maximum = fmaxf(maximum, fabsf(values[p]));
            }
#pragma unroll
            for (int delta = 16; delta > 0; delta /= 2)
              maximum =
                  fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
            maximum = fmaxf(maximum, 1.0e-4f);
            const int exponent =
                static_cast<int>(ceilf(log2f(maximum / 448.0f)));
            const float scale = ldexpf(1.0f, exponent);
            if (lane == 0)
              args.swiglu_scales[base / 128] =
                  static_cast<uint8_t>(max(0, min(254, exponent + 127)));
#pragma unroll
            for (int p = 0; p < 4; ++p)
              args.swiglu_quantized[base + lane + p * 32] =
                  convert(values[p] / scale);
          }
#ifdef DSPARK_ROUTED_GROUP_READY
          // Publish every member's quantized values/scales before readiness.
          __threadfence();
#endif
          // Prevent the next gate drain overwriting slot 0 while a peer reads
          // it.
          cutlass::arch::NamedBarrier::sync(4 * 32, 0);
#ifdef DSPARK_ROUTED_GROUP_READY
          if (threadIdx.x == 0) {
            uint32_t prior;
            uint32_t* ready = args.group_ready + group_member(ctx.members, 0);
            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(prior) : "l"(ready) : "memory");
          }
#endif
        }
      }
#endif
#if defined(DSPARK_W13_CREW_PROBE)
      probe_stores += DG_PROBE_NOW() - probe_mark;
#endif
      ++leader;
    }
#if defined(DSPARK_W13_CREW_PROBE)
    dg_probe_add(2, 0, DG_PROBE_NOW() - probe_start);
    dg_probe_add(2, 1, probe_stall_acc);
    dg_probe_add(2, 2, probe_tmem);
    dg_probe_add(2, 3, probe_stores);
    dg_probe_add(2, 6, 1);
    dg_probe_add(2, 7, probe_items);
#endif
  }
  __syncthreads();

  // All producer/drain threads have joined and the MMA crews explicitly
  // observed the final async acknowledgements. End each initialized lifetime
  // before another scheduler phase repurposes this shared arena.
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int slot = 0; slot < kDgStages; ++slot) {
      cutlass::arch::ClusterBarrier::invalidate(&storage.stage_full[slot]);
      cutlass::arch::ClusterBarrier::invalidate(&storage.stage_free[slot]);
#if defined(DSPARK_W13_TMA_CANDIDATE)
      if constexpr (kUseTmaWeights)
        cutlass::arch::ClusterBarrier::invalidate(&storage.weight_full[slot]);
#endif
    }
    CUTE_UNROLL
    for (int acc = 0; acc < 2; ++acc) {
      cutlass::arch::ClusterBarrier::invalidate(&storage.accumulator_full[acc]);
      cutlass::arch::ClusterBarrier::invalidate(&storage.accumulator_free[acc]);
#if defined(DSPARK_W13_SFA_CACHE)
      cutlass::arch::ClusterBarrier::invalidate(&storage.sfa_free[acc]);
#endif
    }
#if defined(DSPARK_W13_SFA_CACHE)
    CUTE_UNROLL
    for (int group = 0; group < kDgSfaCacheStages / 4; ++group)
      cutlass::arch::ClusterBarrier::invalidate(&storage.sfa_group_ready[group]);
#endif
  }
  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}

__device__ inline void execute_w13_dg(const Args& args) {
#ifdef DSPARK_ROUTED_PACKED_SFA
  execute_dg_grouped_pipeline<W13GroupPolicy, false, false, 0, DSPARK_ROUTED_PACKED_SFA, DSPARK_ROUTED_SF_RING, DSPARK_ROUTED_GROUP_MASKS, DSPARK_ROUTED_REUSE_TMEM, DSPARK_ROUTED_COMPACT_TILES
#ifdef DSPARK_ROUTED_FUSED_SWIGLU
      , true
#endif
      >(args);
#else
  execute_dg_grouped_pipeline<W13GroupPolicy>(args);
#endif
}

#if defined(DSPARK_W13_TMA_CANDIDATE)
__device__ inline void execute_w13_dg_tma(
    const Args& args, const void* weight_tma_descriptor) {
#ifdef DSPARK_ROUTED_PACKED_SFA
  execute_dg_grouped_pipeline<W13GroupPolicy, true, false, 0, DSPARK_ROUTED_PACKED_SFA, DSPARK_ROUTED_SF_RING, DSPARK_ROUTED_GROUP_MASKS, DSPARK_ROUTED_REUSE_TMEM, DSPARK_ROUTED_COMPACT_TILES
#ifdef DSPARK_ROUTED_FUSED_SWIGLU
      , true
#endif
      >(args, weight_tma_descriptor);
#else
  execute_dg_grouped_pipeline<W13GroupPolicy, true>(
      args, weight_tma_descriptor);
#endif
}

// Stage-major candidate: identical body, but the tensor map views a
// host-repacked [stage][row][64 B] arena, so each stage's 8 KiB payload is
// one contiguous DRAM run.
__device__ inline void execute_w13_dg_tma_stage_major(
    const Args& args, const void* weight_tma_descriptor) {
  execute_dg_grouped_pipeline<W13GroupPolicy, true, true>(
      args, weight_tma_descriptor);
}

#if defined(DSPARK_W13_STAGING_PROBE)
// BENCH-ONLY, GARBAGE NUMERICS. Prices one staging stream out of the ring
// round so the routed band's residual can be attributed.
template <int kProbe>
__device__ inline void execute_w13_dg_tma_probe(
    const Args& args, const void* weight_tma_descriptor) {
  execute_dg_grouped_pipeline<W13GroupPolicy, true, false, kProbe>(
      args, weight_tma_descriptor);
}
#endif
#endif

#endif  // __CUDA_ARCH__ >= 1000

// Routed W2: the second FP4 contraction. Same mixed E4M3-by-E2M1 semantics
// and 32-element scale boundary as W13; K runs over the 2048-wide requantized
// swiglu activations and outputs accumulate per (route_row, 128-output tile).
struct W2Args {
  const cutlass::float_e4m3_t* swiglu_quantized;
  const uint8_t* swiglu_scales;
  const int32_t* indices;
  __nv_bfloat16* output_partials;
#ifdef DSPARK_ROUTED_GROUP_READY
  const uint32_t* group_ready;
#endif
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  int weight_base;
  int expert_weight_slots;
  int w2_slot;
  int w2_scale_slot;
  uint32_t begin;
  uint32_t end;
};

constexpr int kW2OutputTiles = kHidden / kOutputTile;
constexpr int kW2KBlocks = kIntermediate / kFp4Block;

struct W2ItemPlan {
  const uint8_t* weight;
  const uint8_t* weight_scales;
  int route_row;
  int output_base;
};

__device__ __forceinline__ W2ItemPlan w2_item_plan(
    const W2Args& args, uint32_t item) {
  W2ItemPlan plan;
  plan.route_row = static_cast<int>(item) / kW2OutputTiles;
  plan.output_base =
      (static_cast<int>(item) % kW2OutputTiles) * kOutputTile;
  const int expert = args.indices[plan.route_row];
  const int table_base =
      args.weight_base + expert * args.expert_weight_slots;
  plan.weight =
      args.weight_arena + args.weight_offsets[table_base + args.w2_slot];
  plan.weight_scales =
      args.weight_arena + args.weight_offsets[table_base + args.w2_scale_slot];
  return plan;
}

// Scalar W2 body: exact retained #80/#84 semantics.
__device__ inline void execute_w2_scalar(const W2Args& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    if (threadIdx.x >= kOutputTile) {
      continue;
    }
    const W2ItemPlan plan = w2_item_plan(args, item);
    const int output = plan.output_base + static_cast<int>(threadIdx.x);
    float result = 0.0f;
    for (int block = 0; block < kW2KBlocks; ++block) {
      float inner = 0.0f;
      const int input_base =
          plan.route_row * kIntermediate + block * kFp4Block;
      const int logical_weight_base = output * kIntermediate + block * kFp4Block;
#pragma unroll
      for (int offset = 0; offset < kFp4Block; offset += 2) {
        const uint8_t packed = plan.weight[(logical_weight_base + offset) >> 1];
        inner = fmaf(
            static_cast<float>(args.swiglu_quantized[input_base + offset]),
            decode_e2m1_code(packed & 0x0f),
            inner);
        inner = fmaf(
            static_cast<float>(args.swiglu_quantized[input_base + offset + 1]),
            decode_e2m1_code(packed >> 4),
            inner);
      }
      const float input_scale = decode_e8m0(
          args.swiglu_scales[
              plan.route_row * (kIntermediate / kQuantBlock) + block / 4]);
      const float weight_scale =
          decode_e8m0(plan.weight_scales[output * kW2KBlocks + block]);
      result = fmaf(inner, input_scale * weight_scale, result);
    }
    args.output_partials[plan.route_row * kHidden + output] =
        __float2bfloat16_rn(result);
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
struct W2Policy {
  using Args = W2Args;
  static constexpr int kKBlocks = kW2KBlocks;

  static __device__ __forceinline__ PipelinePlan plan(
      const Args& args, uint32_t item) {
    const W2ItemPlan resolved = w2_item_plan(args, item);
    PipelinePlan view;
    view.weight = resolved.weight;
    view.weight_scales = resolved.weight_scales;
    view.input_row = args.swiglu_quantized + resolved.route_row * kIntermediate;
    view.input_scales_row =
        args.swiglu_scales + resolved.route_row * (kIntermediate / kQuantBlock);
    view.store_row = args.output_partials + resolved.route_row * kHidden;
    view.output_base = resolved.output_base;
    return view;
  }
};

__device__ inline void execute_w2_tcgen(const W2Args& args) {
  execute_fp4_pipeline<W2Policy>(args);
}

// Grouped W2: symmetric to the W13 grouped body, but every member has its
// OWN K-input row (the swiglu activations are route_row-indexed), so the
// only sharing is the expert's weight/scale stream — which is the whole win.
struct W2GroupPolicy {
  using Args = W2Args;
  static constexpr int kKBlocks = kW2KBlocks;
  static constexpr int kRouteRows = kDraftRows * kActivatedExperts;
#if defined(DSPARK_W13_TMA_CANDIDATE)
  static constexpr int kTmaPlanes = 2;
  static constexpr int kWeightRowBytes = kIntermediate / 2;
  static constexpr int kWeightRows = kHidden;
#endif
  // Measured breakeven (overlap sweep, iteration 6): W2 wins from ~8
  // deduplicated route_rows.
  static constexpr int kMinDuplicates = 8;

  static __device__ __forceinline__ int route_row(uint32_t item) {
    return static_cast<int>(item) / kW2OutputTiles;
  }

  static __device__ __forceinline__ void resolve(
      const Args& args, uint32_t item, int expert, GroupContext& ctx) {
    ctx.branch = 0;
    ctx.output_base =
        (static_cast<int>(item) % kW2OutputTiles) * kOutputTile;
    const int table_base =
        args.weight_base + expert * args.expert_weight_slots;
    ctx.weight =
        args.weight_arena + args.weight_offsets[table_base + args.w2_slot];
    ctx.weight_scales = args.weight_arena
        + args.weight_offsets[table_base + args.w2_scale_slot];
  }

  static __device__ __forceinline__ const cutlass::float_e4m3_t* input_row(
      const Args& args, int route_row) {
    return args.swiglu_quantized + route_row * kIntermediate;
  }
  static __device__ __forceinline__ const uint8_t* input_scales_row(
      const Args& args, int route_row) {
    return args.swiglu_scales + route_row * (kIntermediate / kQuantBlock);
  }
  static __device__ __forceinline__ __nv_bfloat16* store_row(
      const Args& args, const GroupContext&, int route_row) {
    return args.output_partials + route_row * kHidden;
  }
};

__device__ inline void execute_w2_tcgen_grouped(const W2Args& args) {
  execute_fp4_grouped_pipeline<W2GroupPolicy>(args);
}

__device__ inline void execute_w2_tcgen_auto(const W2Args& args) {
  if (grouped_profitable<W2GroupPolicy>(args)) {
    execute_fp4_grouped_pipeline<W2GroupPolicy>(args);
  } else {
    execute_fp4_pipeline<W2Policy>(args);
  }
}

// DeepGEMM-transplanted W2 body (see the execute_w13_dg provenance block):
// same grouped leader/no-op item semantics, K = 2048 so 16 pipeline stages
// per item.
__device__ inline void execute_w2_dg(const W2Args& args) {
#ifdef DSPARK_ROUTED_PACKED_SFA
  execute_dg_grouped_pipeline<W2GroupPolicy, false, false, 0, DSPARK_ROUTED_PACKED_SFA, DSPARK_ROUTED_SF_RING, DSPARK_ROUTED_GROUP_MASKS, DSPARK_ROUTED_REUSE_TMEM, DSPARK_ROUTED_COMPACT_TILES>(args);
#else
  execute_dg_grouped_pipeline<W2GroupPolicy>(args);
#endif
}

#if defined(DSPARK_W13_TMA_CANDIDATE)
__device__ inline void execute_w2_dg_tma(
    const W2Args& args, const void* weight_tma_descriptor) {
#ifdef DSPARK_ROUTED_PACKED_SFA
  execute_dg_grouped_pipeline<W2GroupPolicy, true, false, 0, DSPARK_ROUTED_PACKED_SFA, DSPARK_ROUTED_SF_RING, DSPARK_ROUTED_GROUP_MASKS, DSPARK_ROUTED_REUSE_TMEM, DSPARK_ROUTED_COMPACT_TILES>(
      args, weight_tma_descriptor);
#else
  execute_dg_grouped_pipeline<W2GroupPolicy, true>(
      args, weight_tma_descriptor);
#endif
}

__device__ inline void execute_w2_dg_tma_stage_major(
    const W2Args& args, const void* weight_tma_descriptor) {
  execute_dg_grouped_pipeline<W2GroupPolicy, true, true>(
      args, weight_tma_descriptor);
}
#endif

// BATCH-AMORTIZATION PROBE (step 2). BENCH ONLY: gated on
// -DDSPARK_W13_GROUP_PROBE, so the production translation unit is unchanged.
//
// Widens the routed DG body's N-mode (route rows riding one expert weight
// pass) from the production kMaxGroup = 8 to 16 and 32, and re-measures
// against the SAME deduplicated weight stream. Real members keep their real
// destinations, so the widened body stays bitwise against the production DG
// body on every output it actually owns; columns beyond member_count carry
// additional REAL route rows (activation + E8M0 input scales), which is the
// throughput question a batch-B route table (30 -> 30B rows) would pose.
//
// RING DEPTH IS CUT TO 4 FOR EVERY WIDTH, and that is forced, not chosen.
// DgSharedStorage at the production kDgStages = 7 already occupies 137,216 B
// of the 139,520 B budget, 114,688 B of which is the WEIGHT ring — the one
// thing that cannot be windowed away (unlike the LM body's activation
// operand). Widening N adds 896*N bytes of B ring plus 1024*N bytes of
// drain ping-pong, i.e. +15,360 B at N=16 against ~1,900 B of headroom.
// Host-computed totals (cute 4.2.0.0 cosizes):
//   kDgStages=7: N=8 137,216   N=16 152,576 (over)  N=32 183,296 (over)
//   kDgStages=4: N=8  81,920   N=16  94,208         N=32 118,784
// So 4 is the deepest ring common to all three widths. Holding it fixed
// makes the N axis a controlled comparison; the absolute times are NOT the
// production band's (which runs a 7-deep ring at N=8) and the production
// N=8 point is reported separately to price the ring-depth change.
//
// Numerics: widening N adds independent output columns only. Each column's
// K chain is the same 32-stage x 4-chunk block-scaled UMMA sequence, so
// every real member's output is bitwise invariant in N.
//
// The B SMEM layout for arbitrary N was verified host-side against cute's
// layout algebra: layout<0>(SmemLayoutB) places element (n, k) at
// n*16 + k%16 + (k/16)*16N bytes, one K=32 chunk is 32N bytes, and each
// 16-byte K-quarter of a row is one cp.async packet. Checked for
// N = 8, 16, 32; N=8 reproduces the frozen body's
// (piece>>1)*256 + (piece&1)*128 + member*16 formula exactly. N is capped
// at 32 because the SFB scale block is one 32x128b UTCCP tile (32 rows).
#if defined(DSPARK_W13_GROUP_PROBE)

constexpr int kDgProbeStages = 4;
constexpr int kDgProbeChunks = 4;
constexpr int kDgProbeScratchStride = kHidden;  // >= any policy output_base

template <int kCols>
struct DgProbeTraits {
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Traits<
          cute::SM100_MMA_F8F6F4_SS,
          TcgenWeightType,
          TcgenInputType,
          float,
          cute::C<128>,
          cute::C<kCols>,
          cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
          cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
          cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
          cute::integral_constant<
              cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>{}));
  using ShapeB = decltype(cute::partition_shape_B(
      TiledMma{}, cute::make_shape(cute::Int<kCols>{}, cute::Int<32>{})));
  using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<TcgenInputType>{}, ShapeB{}));
  // accumulator slots 0..2*kCols-1, SFA at 2*kCols, SFB at 2*kCols+4
  static constexpr int kTmemNeeded = 2 * kCols + 8;
  static constexpr int kTmemColumns =
      kTmemNeeded <= 32 ? 32 : (kTmemNeeded <= 64 ? 64 : 128);
};

template <int kCols>
struct DgProbeSharedStorage {
  alignas(128) cute::ArrayEngine<
      uint8_t, kDgProbeChunks * cute::cosize_v<TcgenSmemLayoutA>>
      a[kDgProbeStages];
  alignas(128) cute::ArrayEngine<
      TcgenInputType,
      kDgProbeChunks
          * cute::cosize_v<typename DgProbeTraits<kCols>::SmemLayoutB>>
      b[kDgProbeStages];
  alignas(128) uint32_t sfa[kDgProbeStages][kOutputTile];
  alignas(128) uint32_t sfb[kDgProbeStages][kOutputTile];
  alignas(16) float partial[2][kOutputTile * kCols];
  alignas(16) cute::uint64_t stage_full[kDgProbeStages];
  alignas(16) cute::uint64_t stage_free[kDgProbeStages];
  alignas(16) cute::uint64_t accumulator_full[2];
  alignas(16) cute::uint64_t accumulator_free[2];
  alignas(16) cute::uint32_t tmem_base_ptr;
  alignas(16) int32_t route_indices[kRouteCacheSlots];
};

template <class Policy, int kCols>
__device__ inline void execute_dg_group_probe(
    const typename Policy::Args& args, __nv_bfloat16* scratch) {
  using namespace cute;
  using Traits = DgProbeTraits<kCols>;
  using Storage = DgProbeSharedStorage<kCols>;
  constexpr int kKBlocks = Policy::kKBlocks;
  constexpr int kStagesPerItem = kKBlocks / kDgProbeChunks;
  constexpr int kProbeBlockK = kDgProbeChunks * kFp4Block;
  constexpr int kKWidth = kKBlocks * kFp4Block;
  constexpr int kTmemColumns = Traits::kTmemColumns;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kBChunkBytes = 32 * kCols;
  static_assert(kCols % 8 == 0 && kCols >= 8 && kCols <= 32,
                "SFB rides one 32x128b UTCCP tile, so N is capped at 32");
  static_assert(kKBlocks % kDgProbeChunks == 0);
  static_assert(kStagesPerItem >= kDgProbeStages);
  static_assert(sizeof(Storage) <= kSharedBytesBudget);
  static_assert(cute::cosize_v<typename Traits::SmemLayoutB> == kBChunkBytes);

  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<Storage*>(dynamic_shared_memory);

  for (int r = static_cast<int>(threadIdx.x); r < Policy::kRouteRows;
       r += static_cast<int>(blockDim.x)) {
    storage.route_indices[r] = args.indices[r];
  }
  __syncthreads();
  {
    bool any_leader = false;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      if (chunk_selected(storage.route_indices, Policy::route_row(item))) {
        any_leader = true;
        break;
      }
    }
    if (!any_leader) {
      return;
    }
  }

  typename Traits::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<128>{}, Int<kCols>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = threadIdx.x / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kDgProbeStages>(
        storage.stage_full, kStagingThreads);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kDgProbeStages>(storage.stage_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(
        storage.accumulator_free, kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

  const uint32_t tmem_sfa = storage.tmem_base_ptr + 2 * kCols;
  const uint32_t tmem_sfb = storage.tmem_base_ptr + 2 * kCols + 4;

  uint64_t weight_descriptor[kDgProbeStages];
  uint64_t input_descriptor[kDgProbeStages];
  CUTE_UNROLL
  for (int s = 0; s < kDgProbeStages; ++s) {
    weight_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.a[s].begin()),
            layout<0>(TcgenSmemLayoutA{}))).desc_;
    input_descriptor[s] = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(
            make_smem_ptr(storage.b[s].begin()),
            layout<0>(typename Traits::SmemLayoutB{}))).desc_;
  }
  const UMMA::InstrDescriptorBlockScaled base_instr_desc =
      UMMA::make_instr_desc_block_scaled<
          TcgenWeightType, TcgenInputType, float, cutlass::float_ue8m0_t,
          128, kCols, UMMA::Major::K, UMMA::Major::K>();

  if (warp >= 5) {
    const int lane = static_cast<int>(threadIdx.x) - 5 * 32;
    int round = 0;
    GroupCache group_cache;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      const GroupContext ctx =
          group_plan<Policy>(args, item, storage.route_indices, group_cache);
      if (ctx.member_count == 0) {
        continue;
      }
      const uint8_t* const weight = ctx.weight;
      const uint8_t* const weight_scales = ctx.weight_scales;
      const int output_base = ctx.output_base;
      const int base_row = Policy::route_row(item);
      for (int stage = 0; stage < kStagesPerItem; ++stage, ++round) {
        const int s = round % kDgProbeStages;
        if (round >= kDgProbeStages) {
          wait_barrier(
              storage.stage_free[s], (round / kDgProbeStages - 1) & 1);
        }
        uint8_t* stage_a = reinterpret_cast<uint8_t*>(storage.a[s].begin());
        for (int ci = lane; ci < kOutputTile * 8; ci += kStagingThreads) {
          const int row = ci >> 3;
          const int g = ci & 7;
          const uint8_t* source = weight
              + static_cast<int64_t>(output_base + row) * (kKWidth / 2)
              + stage * 64 + g * 8;
          cutlass::arch::cp_async<8, cutlass::arch::CacheOperation::Always>(
              stage_a + (g >> 1) * 4096 + (g & 1) * 2048 + row * 16, source);
        }
        // kCols activation columns: 8 sixteen-byte pieces each. Chunk j at
        // j*32*kCols, K-quarter h at h*16*kCols, column m at m*16.
        uint8_t* stage_b = reinterpret_cast<uint8_t*>(storage.b[s].begin());
        for (int ci = lane; ci < kCols * 8; ci += kStagingThreads) {
          const int member = ci >> 3;
          const int piece = ci & 7;
          const int member_row = member < ctx.member_count
              ? group_member(ctx.members, member)
              : (base_row + member) % Policy::kRouteRows;
          const uint8_t* source = reinterpret_cast<const uint8_t*>(
              Policy::input_row(args, member_row))
              + stage * kProbeBlockK + piece * 16;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              stage_b + (piece >> 1) * kBChunkBytes
                  + (piece & 1) * (16 * kCols) + member * 16,
              source);
        }
        uint8_t* stage_sfa = reinterpret_cast<uint8_t*>(storage.sfa[s]);
        for (int ci = lane; ci < kOutputTile; ci += kStagingThreads) {
          const int m = (ci & 3) * 32 + (ci >> 2);
          const uint8_t* source = weight_scales
              + (output_base + m) * kKBlocks + stage * 4;
          cutlass::arch::cp_async<4, cutlass::arch::CacheOperation::Always>(
              stage_sfa + ci * 4, source);
        }
        uint8_t* stage_sfb = reinterpret_cast<uint8_t*>(storage.sfb[s]);
        for (int ci = lane; ci < kCols; ci += kStagingThreads) {
          const int member_row = ci < ctx.member_count
              ? group_member(ctx.members, ci)
              : (base_row + ci) % Policy::kRouteRows;
          const uint8_t* source =
              Policy::input_scales_row(args, member_row) + (stage & ~3);
          cutlass::arch::cp_async<4, cutlass::arch::CacheOperation::Always>(
              stage_sfb + ci * 16, source);
        }
        cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
      }
    }
  } else if (warp == 4) {
    int round = 0;
    int leader = 0;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      if (!chunk_selected(storage.route_indices, Policy::route_row(item))) {
        continue;
      }
      const int e = leader & 1;
      for (int stage = 0; stage < kStagesPerItem; ++stage, ++round) {
        const int s = round % kDgProbeStages;
        wait_barrier(storage.stage_full[s], (round / kDgProbeStages) & 1);
        if (stage == 0 && leader >= 2) {
          wait_barrier(storage.accumulator_free[e], (leader / 2 - 1) & 1);
        }
        if (elect_one_sync()) {
          dg_utccp_32x128b_warpx4(
              dg_make_sf_desc(storage.sfa[s]), tmem_sfa);
          if ((stage & 3) == 0) {
            dg_utccp_32x128b_warpx4(
                dg_make_sf_desc(storage.sfb[s]), tmem_sfb);
          }
          CUTE_UNROLL
          for (int j = 0; j < kDgProbeChunks; ++j) {
            dg_mma_mxf8f6f4(
                weight_descriptor[s] + static_cast<uint64_t>(j) * 256,
                input_descriptor[s]
                    + static_cast<uint64_t>(j) * (2 * kCols),
                storage.tmem_base_ptr + e * kCols,
                (stage > 0 || j > 0) ? 1u : 0u,
                dg_runtime_instr_desc(
                    base_instr_desc,
                    static_cast<uint32_t>(j),
                    static_cast<uint32_t>(stage & 3)),
                tmem_sfa, tmem_sfb);
          }
        }
        __syncwarp();
        cutlass::arch::umma_arrive(&storage.stage_free[s]);
        if (stage == kStagesPerItem - 1) {
          cutlass::arch::umma_arrive(&storage.accumulator_full[e]);
        }
      }
      ++leader;
    }
  } else {
    auto tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto thread_source = [&](int e) {
      auto slot = tmem_accumulator;
      slot.data() = storage.tmem_base_ptr + e * kCols;
      return thread_copy.partition_S(slot);
    };
    auto thread_tmem_0 = thread_source(0);
    auto thread_tmem_1 = thread_source(1);
    auto partial_layout = make_layout(
        make_shape(Int<128>{}, Int<kCols>{}),
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

    int leader = 0;
    GroupCache group_cache;
    for (uint32_t item = args.begin; item < args.end; ++item) {
      const GroupContext ctx =
          group_plan<Policy>(args, item, storage.route_indices, group_cache);
      if (ctx.member_count == 0) {
        continue;
      }
      const int e = leader & 1;
      wait_barrier(storage.accumulator_full[e], (leader / 2) & 1);
      if (e == 0) {
        copy(tmem_to_register, thread_tmem_0, register_accumulator);
      } else {
        copy(tmem_to_register, thread_tmem_1, register_accumulator);
      }
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[e]);
      if (e == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      CUTE_UNROLL
      for (int m = 0; m < kCols; ++m) {
        __nv_bfloat16* store =
            m < ctx.member_count
                ? Policy::store_row(args, ctx, group_member(ctx.members, m))
                : scratch + m * kDgProbeScratchStride;
        store[ctx.output_base + static_cast<int>(threadIdx.x)] =
            __float2bfloat16_rn(
                storage.partial[e][m * 128 + threadIdx.x]);
      }
      ++leader;
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
}

#endif  // DSPARK_W13_GROUP_PROBE
#endif  // __CUDA_ARCH__ >= 1000

}  // namespace dspark_w13
