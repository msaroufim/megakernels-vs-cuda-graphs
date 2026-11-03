// Self-contained front/epilogue phase bodies (main projection, shared-expert
// W2, routed/shared SwiGLU, expert combine) for the DeepSeek-V4 DSpark
// megakernel, compiled by both the production kernel and the phase
// microbenchmark (see dspark_w13_phase.cuh for the pattern rationale).
//
// Five bands:
//   main_proj   FP8 GEMM 12288 -> 4096, split-K = 8, ONE row (the anchor
//               token), FP32 split partials; item = (output_tile, split)
//               pair (256 items, runs ONCE at the front of the kernel).
//   shared_w2   FP8 GEMM 2048 -> 4096, 5 rows, BF16 outputs; item = one
//               128-output tile (32/layer). The shared expert's W2 sibling
//               of dspark_shared (W13); scalar in production until now.
//   routed_swiglu / shared_swiglu
//               elementwise clamp/mul + per-128-group amax requant,
//               120 / 20 tiles x 3 layers (diagnostic-first: measure, only
//               optimize if the reference exceeds ~0.05 ms isolated).
//   combine     scatter-sum of top-6 routed partials + shared output + HC
//               residual mix, 160 tiles x 3 layers (diagnostic-first).
//
// The execute_*_reference bodies are verbatim extractions of the production
// bodies from dspark_v4_kernel.cu (pure code motion; stage./task. became
// args., and shared_w2's arena-slot indirection is resolved by the wrapper
// into pre-offset pointers): each is its own oracle in the bench lane.
#pragma once

#include "dspark_tmem.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

// TCGen05 candidate precedent: the FP8 E4M3 x E4M3 UMMA pipeline with
// per-128-block scale chaining (dspark_shared_phase.cuh) and its validated
// split-K generalization in dspark_small_phase.cuh (iter-7). main_proj and
// shared_w2 are the same pipeline at K = 12288 (split 8) and K = 2048
// (no split) respectively.
#include "dspark_shared_phase.cuh"
#include "dspark_batch.h"

namespace dspark_epi {

constexpr int kDraftBlock = 5;
// Flat activation rows over the (batch, block, ...) workspace.
constexpr int kDraftRows = dspark_batch::kRows;
constexpr int kQuantBlock = 128;             // == kMainQuantBlock
constexpr int kOutputTile = 128;             // == kMainOutputTile
constexpr int kHidden = 4096;                // == kMainHidden
constexpr int kMainFeature = 12288;          // == kMainFeatureWidth
constexpr int kMainSplits = 8;               // == kMainSplitK
constexpr int kIntermediate = 2048;          // == kExpertIntermediate
constexpr int kHcStreams = 4;
constexpr int kActivatedExperts = 6;
constexpr int kRoutedExperts = 256;
constexpr float kSwiGluLimit = 10.0f;
constexpr int kSharedBytesBudget = dspark_batch::kDynamicSharedBytes;
// R10: every call site launches 256 threads/CTA (generated::kThreads in the
// megakernel, kBenchThreads in the phase bench; both TUs static_assert the
// match). blockDim.x is a runtime special register, so a loop strided by it
// has an unknown trip count and nvcc cannot unroll it -- one outstanding
// load per trip. Substituting the constant is bitwise-neutral: same
// addresses, same order.
constexpr int kEpiThreads = 256;

// Bit-exact E8M0 decode, verbatim from dspark_v4_kernel.cu (#71/#72).
__device__ __forceinline__ float decode_e8m0(uint8_t bits) {
  const uint32_t exponent = bits;
  const uint32_t float_bits =
      exponent == 0 ? 0x00400000U : exponent << 23;
  return __uint_as_float(float_bits);
}

// Verbatim copy of quantize_bf16_blocks from dspark_v4_kernel.cu (the
// kernel keeps its own copy for the other quantizing bands; this one serves
// the extracted SwiGLU bodies so the header stays self-contained).
__device__ inline void quantize_bf16_blocks(
    const __nv_bfloat16* input,
    cutlass::float_e4m3_t* output,
    uint8_t* scales,
    int blocks,
    float* shared) {
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  for (int block = 0; block < blocks; ++block) {
    const int base = block * kQuantBlock;
    float value = 0.0f;
    if (threadIdx.x < kQuantBlock) {
      value = __bfloat162float(input[base + threadIdx.x]);
      shared[threadIdx.x] = fabsf(value);
    } else {
      shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int delta = kEpiThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + delta]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float amax = fmaxf(shared[0], 1.0e-4f);
      const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
      const int raw_encoded = exponent + 127;
      scales[block] = static_cast<uint8_t>(
          raw_encoded < 0 ? 0 : (raw_encoded > 254 ? 254 : raw_encoded));
      shared[0] = ldexpf(1.0f, exponent);
    }
    __syncthreads();
    if (threadIdx.x < kQuantBlock) {
      output[base + threadIdx.x] = convert(value / shared[0]);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// main_proj: front FP8 projection, 12288 -> 4096, split-K = 8, ONE row.
// ---------------------------------------------------------------------------

struct MainProjArgs {
  // 1 x 12288 E4M3 activations + 96 E8M0 scales (single anchor row).
  const cutlass::float_e4m3_t* quantized_input;
  const uint8_t* input_scales;
  // 4096 x 12288 E4M3 weights, per-(128-output, 128-K) E8M0 block scales.
  const cutlass::float_e4m3_t* weight;
  const uint8_t* weight_scales;
  // FP32 split partials, output_column * 8 + split.
  float* partials;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_main_projection from dspark_v4_kernel.cu
// (pure code motion; stage./task. became args.).
__device__ inline void execute_mainproj_reference(const MainProjArgs& args) {
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int split = static_cast<int>(item) % kMainSplits;
    const int output_tile = static_cast<int>(item) / kMainSplits;
    const int output_column = output_tile * kOutputTile + threadIdx.x;
    if (threadIdx.x < kOutputTile && output_column < kHidden) {
      constexpr int kBlocksPerSplit =
          kMainFeature / kQuantBlock / kMainSplits;
      float result = 0.0f;
      for (int block = 0; block < kBlocksPerSplit; ++block) {
        const int k_block = split * kBlocksPerSplit + block;
        float inner = 0.0f;
        const int input_base = k_block * kQuantBlock;
        const int weight_base = output_column * kMainFeature + input_base;
#pragma unroll
        for (int offset = 0; offset < kQuantBlock; ++offset) {
          inner = fmaf(
              static_cast<float>(args.quantized_input[input_base + offset]),
              static_cast<float>(args.weight[weight_base + offset]),
              inner);
        }
        const float input_scale = decode_e8m0(args.input_scales[k_block]);
        const float weight_scale = decode_e8m0(
            args.weight_scales[(output_column / kQuantBlock) * 96 + k_block]);
        result = fmaf(inner, input_scale * weight_scale, result);
      }
      args.partials[output_column * kMainSplits + split] = result;
    }
  }
}

// ---------------------------------------------------------------------------
// shared_w2: shared-expert down projection, FP8 2048 -> 4096, 5 rows.
// The wrapper resolves the arena weight/scale slots into raw pointers.
// ---------------------------------------------------------------------------

struct SharedW2Args {
  // 5 x 2048 E4M3 SwiGLU activations + 5 x 16 E8M0 scales.
  const cutlass::float_e4m3_t* swiglu_quantized;
  const uint8_t* swiglu_scales;
  // 4096 x 2048 E4M3 weights, per-(128-output, 128-K) E8M0 block scales.
  const cutlass::float_e4m3_t* weight;
  const uint8_t* weight_scales;
  // 5 x 4096 BF16 outputs.
  __nv_bfloat16* shared_output;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_shared_w2 from dspark_v4_kernel.cu
// (pure code motion; stage./task. became args. and the shared_expert_weight
// slot lookups became the wrapper's pre-offset pointers).
__device__ inline void execute_shared_w2_reference(const SharedW2Args& args) {
  constexpr int kOutputTiles = kHidden / kOutputTile;
  constexpr int kBlocks = kIntermediate / kQuantBlock;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int output_tile = static_cast<int>(item) % kOutputTiles;
    const int output = output_tile * kOutputTile + threadIdx.x;
    if (threadIdx.x >= kOutputTile || output >= kHidden) {
      continue;
    }
    float result[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = 0; block < kBlocks; ++block) {
      float inner[kDraftBlock] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      const int weight_base = output * kIntermediate + block * kQuantBlock;
#pragma unroll
      for (int offset = 0; offset < kQuantBlock; ++offset) {
        const float weight_value =
            static_cast<float>(args.weight[weight_base + offset]);
#pragma unroll
        for (int row = 0; row < kDraftBlock; ++row) {
          inner[row] = fmaf(
              static_cast<float>(args.swiglu_quantized[
                  row * kIntermediate + block * kQuantBlock + offset]),
              weight_value,
              inner[row]);
        }
      }
      const float weight_scale =
          decode_e8m0(args.weight_scales[(output / kQuantBlock) * kBlocks + block]);
#pragma unroll
      for (int row = 0; row < kDraftBlock; ++row) {
        const float input_scale = decode_e8m0(
            args.swiglu_scales[row * kBlocks + block]);
        result[row] = fmaf(inner[row], input_scale * weight_scale, result[row]);
      }
    }
#pragma unroll
    for (int row = 0; row < kDraftBlock; ++row) {
      args.shared_output[row * kHidden + output] =
          __float2bfloat16_rn(result[row]);
    }
  }
}

// ---------------------------------------------------------------------------
// routed / shared SwiGLU: clamp/mul + per-128-group amax requant.
// ---------------------------------------------------------------------------

struct SwigluArgs {
  // Gate/up pairs: route_rows (30) x 2 x 2048 BF16 for routed, 5 x 2 x 2048
  // for shared.
  const __nv_bfloat16* w13;
  // Per-route-row weights (routed band only; unused by the shared body).
  const float* route_weights;
  // BF16 SwiGLU outputs + E4M3 requant + E8M0 scales.
  __nv_bfloat16* swiglu;
  cutlass::float_e4m3_t* swiglu_quantized;
  uint8_t* swiglu_scales;
  uint32_t begin;
  uint32_t end;
};

// Verbatim extraction of execute_layer0_routed_swiglu from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_routed_swiglu_reference(
    const SwigluArgs& args, float* shared) {
  constexpr int kVectorTile = 512;
  constexpr int kElements = kDraftRows * kActivatedExperts * kIntermediate;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int base = static_cast<int>(item) * kVectorTile;
    #pragma unroll
    for (int lane_item = threadIdx.x; lane_item < kVectorTile; lane_item += kEpiThreads) {
      const int index = base + lane_item;
      if (index >= kElements) {
        continue;
      }
      const int route_row = index / kIntermediate;
      const int feature = index % kIntermediate;
      const int w13_base = route_row * 2 * kIntermediate + feature;
      float gate = __bfloat162float(args.w13[w13_base]);
      float up = __bfloat162float(args.w13[w13_base + kIntermediate]);
      gate = fminf(gate, kSwiGluLimit);
      up = fminf(fmaxf(up, -kSwiGluLimit), kSwiGluLimit);
      const float value =
          gate / (1.0f + expf(-gate)) * up * args.route_weights[route_row];
      args.swiglu[index] = __float2bfloat16_rn(value);
    }
    __syncthreads();
    quantize_bf16_blocks(
        args.swiglu + base,
        args.swiglu_quantized + base,
        args.swiglu_scales + base / kQuantBlock,
        kVectorTile / kQuantBlock,
        shared);
  }
}

// Verbatim extraction of execute_layer0_shared_swiglu from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_shared_swiglu_reference(
    const SwigluArgs& args, float* shared) {
  constexpr int kVectorTile = 512;
  constexpr int kElements = kDraftRows * kIntermediate;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int base = static_cast<int>(item) * kVectorTile;
    #pragma unroll
    for (int lane_item = threadIdx.x; lane_item < kVectorTile; lane_item += kEpiThreads) {
      const int index = base + lane_item;
      if (index >= kElements) {
        continue;
      }
      const int row = index / kIntermediate;
      const int feature = index % kIntermediate;
      const int w13_base = row * 2 * kIntermediate + feature;
      float gate = __bfloat162float(args.w13[w13_base]);
      float up = __bfloat162float(args.w13[w13_base + kIntermediate]);
      gate = fminf(gate, kSwiGluLimit);
      up = fminf(fmaxf(up, -kSwiGluLimit), kSwiGluLimit);
      const float value = gate / (1.0f + expf(-gate)) * up;
      args.swiglu[index] = __float2bfloat16_rn(value);
    }
    __syncthreads();
    quantize_bf16_blocks(
        args.swiglu + base,
        args.swiglu_quantized + base,
        args.swiglu_scales + base / kQuantBlock,
        kVectorTile / kQuantBlock,
        shared);
  }
}

// ---------------------------------------------------------------------------
// combine: scatter-sum of routed partials + shared output + HC residual mix.
// ---------------------------------------------------------------------------

struct CombineArgs {
  // 30 route-row expert ids (row-major, top-6 per draft row).
  const int32_t* indices;
  // 30 x 4096 BF16 routed partials, 5 x 4096 BF16 shared output.
  const __nv_bfloat16* routed_output_partials;
  const __nv_bfloat16* shared_output;
  // 5 x 4 x 4096 BF16 residual streams; per-row 4x4 comb + 4 post scales.
  const __nv_bfloat16* residual_streams;
  const float* comb;
  const float* post;
  // 5 x 4096 FP32 routed sum + 5 x 4 x 4096 BF16 updated streams.
  float* routed_output;
  __nv_bfloat16* streams;
  uint32_t begin;
  uint32_t end;
};

__device__ __forceinline__ void combine_sort_pair(int& key_a, int& key_b) {
  if (key_b < key_a) {
    const int key = key_a;
    key_a = key_b;
    key_b = key;
  }
}

// Verbatim extraction of execute_layer0_expert_combine from
// dspark_v4_kernel.cu (pure code motion; stage./task. became args.).
__device__ inline void execute_combine_reference(const CombineArgs& args) {
  constexpr int kCombineTile = 128;
  constexpr int kTilesPerCta = kEpiThreads / kCombineTile;
  constexpr int kElements = kDraftRows * kHidden;
  constexpr int kRowStreamElements = kHcStreams * kHidden;
  static_assert(kEpiThreads % kCombineTile == 0);
  // The relaxed GB300 schedule deliberately assigns two independent tiles to
  // each claim so this 160-tile phase retires in one persistent-grid wave.
  // Map the second tile onto the block's previously idle upper 128 threads
  // instead of serializing both tiles through the lower half.  Each element
  // still executes the identical expert-order accumulation and HC update.
  for (uint32_t item_base = args.begin;
       item_base < args.end;
       item_base += kTilesPerCta) {
    const uint32_t item =
        item_base + static_cast<uint32_t>(threadIdx.x / kCombineTile);
    const int lane = threadIdx.x % kCombineTile;
    const int index = static_cast<int>(item) * kCombineTile + lane;
    if (item >= args.end || index >= kElements) {
      continue;
    }
    const int row = index / kHidden;
    const int column = index % kHidden;
    float routed = 0.0f;
#ifdef DSPARK_V4_RELAXED_DAG
    // routed_output_partials is indexed by top-k slot, while the frozen
    // reference accumulates those six values in ascending expert order.  A
    // six-input sorting network reproduces that exact order with 12 compares
    // instead of scanning all 256 experts and testing all six slots at each
    // one.  The low three key bits retain the reference's top-slot tie order.
    int key0 = args.indices[row * kActivatedExperts + 0] * 8 + 0;
    int key1 = args.indices[row * kActivatedExperts + 1] * 8 + 1;
    int key2 = args.indices[row * kActivatedExperts + 2] * 8 + 2;
    int key3 = args.indices[row * kActivatedExperts + 3] * 8 + 3;
    int key4 = args.indices[row * kActivatedExperts + 4] * 8 + 4;
    int key5 = args.indices[row * kActivatedExperts + 5] * 8 + 5;
    combine_sort_pair(key1, key2);
    combine_sort_pair(key4, key5);
    combine_sort_pair(key0, key2);
    combine_sort_pair(key3, key5);
    combine_sort_pair(key0, key1);
    combine_sort_pair(key3, key4);
    combine_sort_pair(key2, key5);
    combine_sort_pair(key0, key3);
    combine_sort_pair(key1, key4);
    combine_sort_pair(key2, key4);
    combine_sort_pair(key1, key3);
    combine_sort_pair(key2, key3);
    const int partial_base = row * kActivatedExperts * kHidden + column;
    routed += __bfloat162float(
        args.routed_output_partials[partial_base + (key0 & 7) * kHidden]);
    routed += __bfloat162float(
        args.routed_output_partials[partial_base + (key1 & 7) * kHidden]);
    routed += __bfloat162float(
        args.routed_output_partials[partial_base + (key2 & 7) * kHidden]);
    routed += __bfloat162float(
        args.routed_output_partials[partial_base + (key3 & 7) * kHidden]);
    routed += __bfloat162float(
        args.routed_output_partials[partial_base + (key4 & 7) * kHidden]);
    routed += __bfloat162float(
        args.routed_output_partials[partial_base + (key5 & 7) * kHidden]);
#else
    for (int expert = 0; expert < kRoutedExperts; ++expert) {
#pragma unroll
      for (int top = 0; top < kActivatedExperts; ++top) {
        if (args.indices[row * kActivatedExperts + top] == expert) {
          routed += __bfloat162float(
              args.routed_output_partials[
                  (row * kActivatedExperts + top) * kHidden + column]);
        }
      }
    }
#endif
    args.routed_output[index] = routed;
    const __nv_bfloat16 expert_output = __float2bfloat16_rn(
        routed + __bfloat162float(args.shared_output[index]));
    const float expert_value = __bfloat162float(expert_output);
    float residual[kHcStreams];
#pragma unroll
    for (int input_stream = 0; input_stream < kHcStreams; ++input_stream) {
      residual[input_stream] = __bfloat162float(
          args.residual_streams[
              row * kRowStreamElements + input_stream * kHidden + column]);
    }
    __nv_bfloat16 updated[kHcStreams];
#pragma unroll
    for (int output_stream = 0; output_stream < kHcStreams; ++output_stream) {
      float residual_sum = 0.0f;
#pragma unroll
      for (int input_stream = 0; input_stream < kHcStreams; ++input_stream) {
        residual_sum = fmaf(
            args.comb[
                (row * kHcStreams + input_stream) * kHcStreams + output_stream],
            residual[input_stream],
            residual_sum);
      }
      updated[output_stream] = __float2bfloat16_rn(
          args.post[row * kHcStreams + output_stream] * expert_value + residual_sum);
    }
#pragma unroll
    for (int output_stream = 0; output_stream < kHcStreams; ++output_stream) {
      args.streams[
          row * kRowStreamElements + output_stream * kHidden + column] =
          updated[output_stream];
    }
  }
}

// ---------------------------------------------------------------------------
// CANDIDATE bodies (relaxed program stage 3, front/epilogue iteration).
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// Generalized FP8 E4M3 x E4M3 UMMA split-K pipeline: the dspark_small
// iter-7 template (itself the dspark_shared pipeline via the iter-5 wo_b
// clone) with the K extent, split count, row count, output width, and
// epilogue parameterized. One item is one (output_tile, split) pair, so
// both instantiations are DROP-IN for the existing phase tilings:
//   main_proj  <12288, 8, 1, 4096, false>  (256 items, FP32 split partials)
//   shared_w2  <2048, 1, 5, 4096, true>    (32 items, BF16 rounded outputs)
// An 8-deep cp.async ring stages 128x32 E4M3 weight slots; warp 4 issues
// one K=32 UMMA per slot with four consecutive MMAs chained into one of
// four TMEM slots (scaleC = 0,1,1,1) = one tensor-core inner product per
// 128-element quantization block; warps 0-3 drain each block through the
// frozen outer FP32 fmaf scale chain. kRows = 1 leaves B columns 1..7 zero.
// Numerics: K=32-tree order inside a block vs the serial 128-FMA chain ->
// low-bit FP32 reorder noise on the split-partial band (wo_b/iter-7 class,
// rel_mean ~1.3e-7) and <=1 BF16 ULP after rounding on the BF16 band (the
// shared-expert precedent, oracle #91); the bench reports the drift.
// kBatchElements > 1 tiles the item space OVER BATCH: one item becomes one
// (batch element, output tile, split) triple, element-major. Because every
// workspace region is (batch, block, ...) contiguous, element e's operands
// and results are exactly e * kRows rows further on. kBatchElements == 1
// folds every added term to zero, so the batch-1 instantiation is the frozen
// body verbatim.
template <
    int kFeature,
    int kSplits,
    int kRows,
    int kWidth,
    bool kBf16Out,
    int kBatchElements = 1>
__device__ inline void execute_fp8_tcgen(
    const cutlass::float_e4m3_t* input,
    const uint8_t* input_scales,
    const cutlass::float_e4m3_t* weight,
    const uint8_t* weight_scales,
    float* partials,
    __nv_bfloat16* bf16_output,
    uint32_t begin,
    uint32_t end) {
  using namespace cute;
  constexpr int kTmemColumns = 32;
  constexpr int kStagingThreads = 3 * 32;
  constexpr int kStages = dspark_shared::kStages;                     // 8
  constexpr int kAccumulatorSlots = dspark_shared::kAccumulatorSlots; // 4
  constexpr int kSplitWidth = kFeature / kSplits;      // K columns per item
  constexpr int kMmaBlocks = kSplitWidth / 32;         // K=32 blocks per item
  constexpr int kGroupBlocks = 4;
  constexpr int kScaleBlocks = kFeature / kQuantBlock; // scale blocks per row
  constexpr int kBlocksPerSplit = kSplitWidth / kQuantBlock;
  constexpr int kOutputTiles = kWidth / kOutputTile;
  constexpr int kInputStageBytes =
      static_cast<int>(cute::cosize_v<dspark_shared::TcgenSmemLayoutB>);
  static_assert(kRows >= 1 && kRows <= 8);
  static_assert(kSplitWidth % kQuantBlock == 0);
  static_assert(kMmaBlocks % kGroupBlocks == 0);
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
          input + static_cast<int64_t>(element) * kRows * kFeature;
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
      if (lane < 2 * kRows) {
        const int row = lane >> 1;
        const int chunk = lane & 1;
        const cutlass::float_e4m3_t* source =
            element_input + row * kFeature + column_base + chunk * 16;
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
    constexpr int kGroupsPerItem = kMmaBlocks / kGroupBlocks;
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
          weight_scales[output_tile * kScaleBlocks + k_block]);
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        const float input_scale = decode_e8m0(
            input_scales[(element * kRows + row) * kScaleBlocks + k_block]);
        result[row] = fmaf(
            storage.partial[parity][threadIdx.x + 128 * row],
            input_scale * weight_scale,
            result[row]);
      }
      if (block == kGroupsPerItem - 1) {
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          if constexpr (kBf16Out) {
            bf16_output[
                (element * kRows + row) * kWidth + output_tile * kOutputTile
                + static_cast<int>(threadIdx.x)] =
                __float2bfloat16_rn(result[row]);
          } else {
            partials[
                ((element * kRows + row) * kWidth + output_tile * kOutputTile
                 + static_cast<int>(threadIdx.x)) * kSplits + split] =
                result[row];
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

// main.projection is WEIGHT-BOUND (4096 x 12288 FP8 = 50 MiB) and its item
// count is batch-invariant, so the batch's rows ride the UMMA N mode: kRows
// goes 1 -> batch inside the atom's existing 8 columns, and the 50 MiB stream
// is read once for the whole batch.
__device__ inline void execute_mainproj_tcgen(const MainProjArgs& args) {
  execute_fp8_tcgen<
      kMainFeature, kMainSplits, dspark_batch::kBatch, kHidden, false>(
      args.quantized_input,
      args.input_scales,
      args.weight,
      args.weight_scales,
      args.partials,
      nullptr,
      args.begin,
      args.end);
}

// The shared expert's W2 is 8 MiB per layer: cheap enough that tiling over
// batch (item count x batch, weights re-read per element) beats retemplating
// the atom.
// R4 band 6: batch 1 keeps the frozen tiling; batch > 1 N-widens onto the
// batch-invariant 32-item tiling.
__device__ inline void execute_shared_w2_tcgen(const SharedW2Args& args) {
#if DSPARK_V4_BATCH == 1
  execute_fp8_tcgen<
      kIntermediate, 1, kDraftBlock, kHidden, true, dspark_batch::kBatch>(
      args.swiglu_quantized,
      args.swiglu_scales,
      args.weight,
      args.weight_scales,
      nullptr,
      args.shared_output,
      args.begin,
      args.end);
#else
  dspark_shared::execute_fp8_tcgen_wide<
      kIntermediate, 1, dspark_batch::kWideRows, dspark_batch::kRows,
      kHidden, true>(
      args.swiglu_quantized,
      args.swiglu_scales,
      args.weight,
      args.weight_scales,
      nullptr,
      args.shared_output,
      args.begin,
      args.end);
#endif
}

#endif  // __CUDA_ARCH__ >= 1000

}  // namespace dspark_epi
