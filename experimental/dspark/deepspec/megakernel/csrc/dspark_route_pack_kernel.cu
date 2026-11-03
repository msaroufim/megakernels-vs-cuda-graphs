#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <climits>
#include <cfloat>

namespace {

constexpr int kMarkovVocab = 129280;
// Unlike SGLang's one-input partial argmax, this leaf reads two freshly
// produced rows.  Sixty-four CTAs expose enough cold-row parallelism on the
// 152-SM GB300; the former 16-CTA/8192-value geometry was fast only when both
// rows were already resident in L2 in the standalone microbenchmark.
constexpr int kMarkovArgmaxTile = 2048;
constexpr int kMarkovArgmaxSplits =
    (kMarkovVocab + kMarkovArgmaxTile - 1) / kMarkovArgmaxTile;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ unsigned long long pack_argmax(
    float value,
    int index) {
  return (static_cast<unsigned long long>(ordered_float_bits(value)) << 32)
      | static_cast<unsigned long long>(
          0xFFFFFFFFu - static_cast<uint32_t>(index));
}

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* pointer, int index);

template <>
__device__ __forceinline__ float load_float<float>(
    const float* pointer,
    int index) {
  return pointer[index];
}

template <>
__device__ __forceinline__ float load_float<__nv_bfloat16>(
    const __nv_bfloat16* pointer,
    int index) {
  return __bfloat162float(pointer[index]);
}

template <typename base_t, typename bias_t>
__global__ __launch_bounds__(256, 1) void markov_add_argmax_kernel(
    const base_t* base_logits,
    const bias_t* bias,
    unsigned long long* argmax_slot) {
  __shared__ unsigned long long warp_best[8];
  const int split_begin = static_cast<int>(blockIdx.x) * kMarkovArgmaxTile;
  const int split_end = min(split_begin + kMarkovArgmaxTile, kMarkovVocab);
  float best_value = -FLT_MAX;
  int best_index = kMarkovVocab;
  for (int index = split_begin + static_cast<int>(threadIdx.x);
       index < split_end;
       index += static_cast<int>(blockDim.x)) {
    const float value =
        load_float(base_logits, index) + load_float(bias, index);
    if (value > best_value || (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  unsigned long long packed = pack_argmax(best_value, best_index);
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const unsigned long long other =
        __shfl_down_sync(0xFFFFFFFFu, packed, delta);
    packed = other > packed ? other : packed;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_best[warp] = packed;
  }
  __syncthreads();
  if (warp == 0) {
    packed = lane < 8 ? warp_best[lane] : 0ULL;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      const unsigned long long other =
          __shfl_down_sync(0xFFFFFFFFu, packed, delta);
      packed = other > packed ? other : packed;
    }
    if (lane == 0) {
      atomicMax(argmax_slot, packed);
    }
  }
}

__global__ void markov_argmax_finalize_kernel(
    long* draft_tokens,
    int draft_token_column,
    unsigned long long* argmax_slot) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const unsigned long long packed = argmax_slot[0];
    const long index = static_cast<long>(
        0xFFFFFFFFu - static_cast<uint32_t>(packed));
    draft_tokens[draft_token_column] = index;
    argmax_slot[0] = 0ULL;
  }
}

__global__ void route_pack_kernel(
    const int* ids,
    const float* weights,
    int* output,
    int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    const unsigned int weight_bits = __bfloat16_as_ushort(
        __float2bfloat16_rn(weights[index]));
    output[index] = (ids[index] << 16) | static_cast<int>(weight_bits);
  }
}

__global__ void target_tap_mean_concat_kernel(
    const __nv_bfloat16* tap0,
    const __nv_bfloat16* tap1,
    const __nv_bfloat16* tap2,
    __nv_bfloat16* output) {
  constexpr int kRows = 6;
  constexpr int kStreams = 4;
  constexpr int kWidth = 4096;
  constexpr int kTaps = 3;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= kRows * kTaps * kWidth) {
    return;
  }
  const int feature = index % kWidth;
  const int tap_id = (index / kWidth) % kTaps;
  const int row = index / (kWidth * kTaps);
  const __nv_bfloat16* tap = tap_id == 0 ? tap0 : (tap_id == 1 ? tap1 : tap2);
  const int base = row * kStreams * kWidth + feature;
  float sum = 0.0f;
#pragma unroll
  for (int stream = 0; stream < kStreams; ++stream) {
    sum += __bfloat162float(tap[base + stream * kWidth]);
  }
  output[index] = __float2bfloat16_rn(sum * 0.25f);
}

__global__ void embedding_hc_expand_kernel(
    const long* token_ids,
    const __nv_bfloat16* embedding,
    __nv_bfloat16* output,
    int vocab_size) {
  constexpr int kRows = 6;
  constexpr int kStreams = 4;
  constexpr int kWidth = 4096;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= kRows * kStreams * kWidth) {
    return;
  }
  const int feature = index % kWidth;
  const int row = index / (kStreams * kWidth);
  const long token = token_ids[row];
  if (token >= 0 && token < vocab_size) {
    output[index] = embedding[token * kWidth + feature];
  }
}

__global__ void prepare_graft_inputs_kernel(
    const long* bonus,
    long* input_ids,
    long* candidates,
    long mask_token_id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const long value = bonus[0];
#pragma unroll
    for (int row = 0; row < 5; ++row) {
      input_ids[row] = row == 0 ? value : mask_token_id;
    }
    candidates[0] = value;
  }
}

__global__ void prepare_target_inject_layout_kernel(
    const int* commit_len,
    const long* prefix_len,
    int* swa_loc,
    long* positions) {
  const int row = static_cast<int>(threadIdx.x);
  if (row < 6) {
    swa_loc[row] = row < commit_len[0] ? (prefix_len[0] + row) & 127 : -1;
    positions[row] = prefix_len[0] + row;
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__global__ __launch_bounds__(256, 1) void q_lora_rmsnorm_quant_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* norm_weight,
    uint8_t* quantized,
    uint8_t* scales,
    float norm_eps) {
  constexpr int kRows = 6;
  constexpr int kWidth = 1024;
  constexpr int kGroup = 128;
  constexpr int kGroups = kWidth / kGroup;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  __shared__ float warp_sums[kWarps];
  __shared__ float inverse_rms;
  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kWidth / kThreads; ++item) {
    const int feature = thread + item * kThreads;
    const float value = __bfloat162float(input[row * kWidth + feature]);
    sum += value * value;
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      inverse_rms = rsqrtf(total / kWidth + norm_eps);
    }
  }
  __syncthreads();

  const int group = warp;
  float values[kGroup / 32];
  float amax = 0.0f;
#pragma unroll
  for (int item = 0; item < kGroup / 32; ++item) {
    const int feature = group * kGroup + item * 32 + lane;
    const __nv_bfloat16 rounded = __float2bfloat16_rn(
        __bfloat162float(input[row * kWidth + feature])
        * inverse_rms
        * __bfloat162float(norm_weight[feature]));
    values[item] = __bfloat162float(rounded);
    amax = fmaxf(amax, fabsf(values[item]));
  }
  amax = warp_max(amax);
  amax = __shfl_sync(0xffffffff, amax, 0);
  amax = fmaxf(amax, 1.0e-10f);
  const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
  const int encoded = max(0, min(254, exponent + 127));
  if (lane == 0) {
    // SGLang/DeepGEMM packs four K-group UE8M0 bytes per int32, with
    // the padded M dimension as the inner physical stride.
    scales[(group >> 2) * 8 * 4 + row * 4 + (group & 3)] =
        static_cast<uint8_t>(encoded);
  }
  const float scale = ldexpf(1.0f, exponent);
#pragma unroll
  for (int item = 0; item < kGroup / 32; ++item) {
    const int feature = group * kGroup + item * 32 + lane;
    const __nv_fp8_e4m3 value(values[item] / scale);
    quantized[row * kWidth + feature] = *reinterpret_cast<const uint8_t*>(&value);
  }
}

__global__ __launch_bounds__(1024, 1) void target_hc_head_rmsnorm_kernel(
    const __nv_bfloat16* streams,
    const float* hc_fn,
    const float* hc_scale,
    const float* hc_base,
    const __nv_bfloat16* norm_weight,
    __nv_bfloat16* output,
    float norm_eps,
    float hc_eps) {
  constexpr int kStreams = 4;
  constexpr int kWidth = 4096;
  constexpr int kTotal = kStreams * kWidth;
  constexpr int kWarps = 32;
  __shared__ float partial[5][kWarps];
  __shared__ float pre[kStreams];
  __shared__ float y_partial[kWarps];
  __shared__ float inverse_rms;
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int row = static_cast<int>(blockIdx.x);
  const __nv_bfloat16* row_streams = streams + row * kTotal;
  float accumulators[5] = {};
  for (int index = thread; index < kTotal; index += blockDim.x) {
    const float value = __bfloat162float(row_streams[index]);
    accumulators[0] += value * value;
#pragma unroll
    for (int mix = 0; mix < kStreams; ++mix) {
      accumulators[mix + 1] += hc_fn[mix * kTotal + index] * value;
    }
  }
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    accumulators[component] = warp_sum(accumulators[component]);
    if (lane == 0) {
      partial[component][warp] = accumulators[component];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = lane < kWarps ? partial[component][lane] : 0.0f;
      value = warp_sum(value);
      if (lane == 0) {
        partial[component][0] = value;
      }
    }
  }
  __syncthreads();
  if (thread == 0) {
    const float stream_rms = rsqrtf(partial[0][0] / kTotal + norm_eps);
#pragma unroll
    for (int mix = 0; mix < kStreams; ++mix) {
      const float argument =
          partial[mix + 1][0] * stream_rms * hc_scale[0] + hc_base[mix];
      pre[mix] = 1.0f / (1.0f + expf(-argument)) + hc_eps;
    }
  }
  __syncthreads();
  float output_squares = 0.0f;
  for (int feature = thread; feature < kWidth; feature += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
      value += pre[stream]
          * __bfloat162float(row_streams[stream * kWidth + feature]);
    }
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    output[row * kWidth + feature] = rounded;
    const float rounded_float = __bfloat162float(rounded);
    output_squares += rounded_float * rounded_float;
  }
  output_squares = warp_sum(output_squares);
  if (lane == 0) {
    y_partial[warp] = output_squares;
  }
  __syncthreads();
  if (warp == 0) {
    float value = lane < kWarps ? y_partial[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) {
      inverse_rms = rsqrtf(value / kWidth + norm_eps);
    }
  }
  __syncthreads();
  for (int feature = thread; feature < kWidth; feature += blockDim.x) {
    const int index = row * kWidth + feature;
    output[index] = __float2bfloat16_rn(
        __bfloat162float(output[index])
        * inverse_rms
        * __bfloat162float(norm_weight[feature]));
  }
}

__global__ void fused_moe_hc_post_kernel(
    const __nv_bfloat16* routed,
    const __nv_bfloat16* shared,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    __nv_bfloat16* output) {
  constexpr int kTiles = 8;
  constexpr int kTileWidth = 4096 / kTiles;
  const int row = static_cast<int>(blockIdx.x) / kTiles;
  const int tile = static_cast<int>(blockIdx.x) % kTiles;
  const int feature_end = (tile + 1) * kTileWidth;
  for (int feature = tile * kTileWidth + static_cast<int>(threadIdx.x);
       feature < feature_end;
       feature += static_cast<int>(blockDim.x)) {
    const int moe_index = row * 4096 + feature;
    const __nv_bfloat16 moe = __float2bfloat16_rn(
        __bfloat162float(routed[moe_index])
        + (shared == nullptr ? 0.0f : __bfloat162float(shared[moe_index])));
    float residual_values[4];
#pragma unroll
    for (int input = 0; input < 4; ++input) {
      residual_values[input] = __bfloat162float(
          residual[(row * 4 + input) * 4096 + feature]);
    }
#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      float value = post[row * 4 + stream] * __bfloat162float(moe);
#pragma unroll
      for (int input = 0; input < 4; ++input) {
        value += comb[(row * 4 + input) * 4 + stream]
            * residual_values[input];
      }
      output[(row * 4 + stream) * 4096 + feature] =
          __float2bfloat16_rn(value);
    }
  }
}

__global__ void fused_moe_finalize_hc_post_kernel(
    const __nv_bfloat16* gemm2,
    int gemm2_stride,
    const __nv_bfloat16* expert_weights,
    const int* expanded_to_permuted,
    const __nv_bfloat16* shared,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    __nv_bfloat16* output) {
  constexpr int kTopK = 6;
  constexpr int kTiles = 8;
  constexpr int kTileWidth = 4096 / kTiles;
  const int row = static_cast<int>(blockIdx.x) / kTiles;
  const int tile = static_cast<int>(blockIdx.x) % kTiles;
  const int pair_begin = tile * (kTileWidth / 2);
  const int pair_end = (tile + 1) * (kTileWidth / 2);
  const auto* gemm2_pairs =
      reinterpret_cast<const __nv_bfloat162*>(gemm2);
  const auto* shared_pairs =
      reinterpret_cast<const __nv_bfloat162*>(shared);
  const auto* residual_pairs =
      reinterpret_cast<const __nv_bfloat162*>(residual);
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output);
  for (int pair = pair_begin + static_cast<int>(threadIdx.x);
       pair < pair_end;
       pair += static_cast<int>(blockDim.x)) {
    float routed_lo = 0.0f;
    float routed_hi = 0.0f;
#pragma unroll
    for (int expert = 0; expert < kTopK; ++expert) {
      const int expanded = row * kTopK + expert;
      const int permuted = expanded_to_permuted[expanded];
      if (permuted >= 0) {
        const float weight = __bfloat162float(expert_weights[expanded]);
        const float2 value = __bfloat1622float2(
            gemm2_pairs[permuted * (gemm2_stride / 2) + pair]);
        routed_lo += weight * value.x;
        routed_hi += weight * value.y;
      }
    }
    // Preserve both BF16 boundaries from FlashInfer finalize followed by the
    // existing routed/shared combine.  This is a graph-boundary fusion, not a
    // reassociation of the model's numerical program.
    const __nv_bfloat162 routed =
        __floats2bfloat162_rn(routed_lo, routed_hi);
    const int row_pair = row * (4096 / 2) + pair;
    const float2 routed_float = __bfloat1622float2(routed);
    const float2 shared_float =
        __bfloat1622float2(shared_pairs[row_pair]);
    const __nv_bfloat162 moe = __floats2bfloat162_rn(
        routed_float.x + shared_float.x,
        routed_float.y + shared_float.y);
    float2 residual_values[4];
#pragma unroll
    for (int input = 0; input < 4; ++input) {
      residual_values[input] = __bfloat1622float2(
          residual_pairs[(row * 4 + input) * (4096 / 2) + pair]);
    }
    const float2 moe_float = __bfloat1622float2(moe);
#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      float value_lo = post[row * 4 + stream] * moe_float.x;
      float value_hi = post[row * 4 + stream] * moe_float.y;
#pragma unroll
      for (int input = 0; input < 4; ++input) {
        const float coefficient = comb[(row * 4 + input) * 4 + stream];
        value_lo += coefficient * residual_values[input].x;
        value_hi += coefficient * residual_values[input].y;
      }
      output_pairs[(row * 4 + stream) * (4096 / 2) + pair] =
          __floats2bfloat162_rn(value_lo, value_hi);
    }
  }
}

}  // namespace

template <typename base_t, typename bias_t>
void launch_markov_add_argmax(
    const torch::Tensor& base_logits,
    const torch::Tensor& bias,
    const torch::Tensor& draft_tokens,
    const torch::Tensor& argmax_slot,
    int draft_token_column) {
  auto stream = at::cuda::getCurrentCUDAStream();
  markov_add_argmax_kernel<<<kMarkovArgmaxSplits, 256, 0, stream>>>(
      reinterpret_cast<const base_t*>(base_logits.const_data_ptr()),
      reinterpret_cast<const bias_t*>(bias.const_data_ptr()),
      reinterpret_cast<unsigned long long*>(argmax_slot.data_ptr<int64_t>()));
  markov_argmax_finalize_kernel<<<1, 1, 0, stream>>>(
      draft_tokens.data_ptr<long>(),
      draft_token_column,
      reinterpret_cast<unsigned long long*>(argmax_slot.data_ptr<int64_t>()));
}

void dspark_markov_add_argmax_cuda(
    const torch::Tensor& base_logits,
    const torch::Tensor& bias,
    const torch::Tensor& draft_tokens,
    const torch::Tensor& argmax_slot,
    long draft_token_column) {
  TORCH_CHECK(
      base_logits.is_cuda() && bias.is_cuda() && draft_tokens.is_cuda()
          && argmax_slot.is_cuda(),
      "Markov add/argmax tensors must be CUDA");
  TORCH_CHECK(
      base_logits.is_contiguous() && bias.is_contiguous()
          && draft_tokens.is_contiguous() && argmax_slot.is_contiguous(),
      "Markov add/argmax tensors must be contiguous");
  TORCH_CHECK(
      base_logits.sizes() == torch::IntArrayRef({1, kMarkovVocab})
          && bias.sizes() == base_logits.sizes(),
      "Markov logits and bias must have shape [1, 129280]");
  TORCH_CHECK(
      (base_logits.scalar_type() == torch::kBFloat16
       || base_logits.scalar_type() == torch::kFloat32)
          && (bias.scalar_type() == torch::kBFloat16
              || bias.scalar_type() == torch::kFloat32),
      "Markov logits and bias must be BF16 or FP32");
  TORCH_CHECK(
      draft_tokens.dim() == 2 && draft_tokens.size(0) == 1
          && draft_tokens.scalar_type() == torch::kInt64,
      "draft tokens must be int64 [1, gamma]");
  TORCH_CHECK(
      draft_token_column >= 0 && draft_token_column < draft_tokens.size(1),
      "draft token column is out of range");
  TORCH_CHECK(
      argmax_slot.numel() == 1 && argmax_slot.scalar_type() == torch::kInt64,
      "argmax slot must be one int64 word");
  const c10::cuda::CUDAGuard device_guard(base_logits.device());
  const int column = static_cast<int>(draft_token_column);
  if (base_logits.scalar_type() == torch::kBFloat16) {
    if (bias.scalar_type() == torch::kBFloat16) {
      launch_markov_add_argmax<__nv_bfloat16, __nv_bfloat16>(
          base_logits, bias, draft_tokens, argmax_slot, column);
    } else {
      launch_markov_add_argmax<__nv_bfloat16, float>(
          base_logits, bias, draft_tokens, argmax_slot, column);
    }
  } else if (bias.scalar_type() == torch::kBFloat16) {
    launch_markov_add_argmax<float, __nv_bfloat16>(
        base_logits, bias, draft_tokens, argmax_slot, column);
  } else {
    launch_markov_add_argmax<float, float>(
        base_logits, bias, draft_tokens, argmax_slot, column);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_route_pack_cuda(
    const torch::Tensor& ids,
    const torch::Tensor& weights,
    const torch::Tensor& output) {
  TORCH_CHECK(ids.is_cuda() && weights.is_cuda() && output.is_cuda(),
              "route tensors must be CUDA");
  TORCH_CHECK(ids.is_contiguous() && weights.is_contiguous() && output.is_contiguous(),
              "route tensors must be contiguous");
  TORCH_CHECK(ids.scalar_type() == torch::kInt32, "ids must be int32");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kInt32, "output must be int32");
  TORCH_CHECK(ids.sizes() == weights.sizes() && ids.sizes() == output.sizes(),
              "route tensor shapes must match");
  TORCH_CHECK(ids.numel() <= INT_MAX, "route tensor is too large");
  const c10::cuda::CUDAGuard device_guard(ids.device());
  const int count = static_cast<int>(ids.numel());
  route_pack_kernel<<<1, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(),
      weights.data_ptr<float>(),
      output.data_ptr<int>(),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_target_tap_mean_concat_cuda(
    const torch::Tensor& tap0,
    const torch::Tensor& tap1,
    const torch::Tensor& tap2,
    const torch::Tensor& output) {
  for (const auto* tensor : {&tap0, &tap1, &tap2, &output}) {
    TORCH_CHECK(tensor->is_cuda(), "target tap tensors must be CUDA");
    TORCH_CHECK(tensor->is_contiguous(), "target tap tensors must be contiguous");
    TORCH_CHECK(tensor->scalar_type() == torch::kBFloat16,
                "target tap tensors must be BF16");
  }
  TORCH_CHECK(tap0.sizes() == tap1.sizes() && tap0.sizes() == tap2.sizes(),
              "target tap shapes must match");
  TORCH_CHECK(tap0.numel() == 6 * 4 * 4096,
              "target taps must have shape [6, 4, 4096]");
  TORCH_CHECK(output.numel() == 6 * 3 * 4096,
              "target hidden output must have shape [6, 12288]");
  const c10::cuda::CUDAGuard device_guard(tap0.device());
  constexpr int count = 6 * 3 * 4096;
  target_tap_mean_concat_kernel<<<(count + 255) / 256, 256, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(tap0.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(tap1.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(tap2.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_q_lora_rmsnorm_quant_cuda(
    const torch::Tensor& input,
    const torch::Tensor& norm_weight,
    const torch::Tensor& quantized,
    const torch::Tensor& scales,
    double norm_eps) {
  TORCH_CHECK(input.is_cuda() && norm_weight.is_cuda()
                  && quantized.is_cuda() && scales.is_cuda(),
              "Q-LoRA tensors must be CUDA");
  TORCH_CHECK(input.is_contiguous() && norm_weight.is_contiguous()
                  && quantized.is_contiguous(),
              "Q-LoRA input, weight, and output must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16
                  && norm_weight.scalar_type() == torch::kBFloat16,
              "Q-LoRA input and norm weight must be BF16");
  TORCH_CHECK(quantized.scalar_type() == torch::kUInt8,
              "Q-LoRA quantized output must be uint8");
  TORCH_CHECK(scales.scalar_type() == torch::kInt32,
              "Q-LoRA scales must use the packed int32 UE8M0 ABI");
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({6, 1024}),
              "Q-LoRA input must have shape [6, 1024]");
  TORCH_CHECK(norm_weight.numel() == 1024,
              "Q-LoRA norm weight must have 1024 elements");
  TORCH_CHECK(quantized.numel() == input.numel(),
              "Q-LoRA quantized output shape mismatch");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({6, 2}),
              "Q-LoRA scales must have logical shape [6, 2]");
  TORCH_CHECK(scales.stride(0) == 1 && scales.stride(1) == 8,
              "Q-LoRA scales must use the TMA-aligned column-major layout");
  TORCH_CHECK(norm_eps > 0.0, "Q-LoRA RMSNorm epsilon must be positive");
  const c10::cuda::CUDAGuard device_guard(input.device());
  q_lora_rmsnorm_quant_kernel<<<6, 256, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(norm_weight.const_data_ptr()),
      quantized.data_ptr<uint8_t>(),
      reinterpret_cast<uint8_t*>(scales.data_ptr<int>()),
      static_cast<float>(norm_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_embedding_hc_expand_cuda(
    const torch::Tensor& token_ids,
    const torch::Tensor& embedding,
    const torch::Tensor& output) {
  TORCH_CHECK(token_ids.is_cuda() && embedding.is_cuda() && output.is_cuda(),
              "embedding expansion tensors must be CUDA");
  TORCH_CHECK(token_ids.is_contiguous() && embedding.is_contiguous()
                  && output.is_contiguous(),
              "embedding expansion tensors must be contiguous");
  TORCH_CHECK(token_ids.scalar_type() == torch::kInt64,
              "embedding token ids must be int64");
  TORCH_CHECK(embedding.scalar_type() == torch::kBFloat16
                  && output.scalar_type() == torch::kBFloat16,
              "embedding expansion data must be BF16");
  TORCH_CHECK(token_ids.numel() == 6, "embedding expansion requires six ids");
  TORCH_CHECK(embedding.dim() == 2 && embedding.size(1) == 4096,
              "embedding table must have width 4096");
  TORCH_CHECK(output.numel() == 6 * 4 * 4096,
              "embedding output must have shape [6, 4, 4096]");
  const c10::cuda::CUDAGuard device_guard(token_ids.device());
  constexpr int count = 6 * 4 * 4096;
  embedding_hc_expand_kernel<<<(count + 255) / 256, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      token_ids.const_data_ptr<long>(),
      reinterpret_cast<const __nv_bfloat16*>(embedding.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int>(embedding.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_prepare_graft_inputs_cuda(
    const torch::Tensor& bonus,
    const torch::Tensor& input_ids,
    const torch::Tensor& candidates,
    long mask_token_id) {
  TORCH_CHECK(bonus.is_cuda() && input_ids.is_cuda() && candidates.is_cuda(),
              "graft input tensors must be CUDA");
  TORCH_CHECK(bonus.is_contiguous() && input_ids.is_contiguous()
                  && candidates.is_contiguous(),
              "graft input tensors must be contiguous");
  TORCH_CHECK(bonus.scalar_type() == torch::kInt64
                  && input_ids.scalar_type() == torch::kInt64
                  && candidates.scalar_type() == torch::kInt64,
              "graft input tensors must be int64");
  TORCH_CHECK(bonus.numel() >= 1 && input_ids.numel() == 5
                  && candidates.numel() == 6,
              "graft input shapes are invalid");
  const c10::cuda::CUDAGuard device_guard(bonus.device());
  prepare_graft_inputs_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      bonus.const_data_ptr<long>(),
      input_ids.data_ptr<long>(),
      candidates.data_ptr<long>(),
      mask_token_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_prepare_target_inject_layout_cuda(
    const torch::Tensor& commit_len,
    const torch::Tensor& prefix_len,
    const torch::Tensor& swa_loc,
    const torch::Tensor& positions) {
  TORCH_CHECK(commit_len.is_cuda() && prefix_len.is_cuda()
                  && swa_loc.is_cuda() && positions.is_cuda(),
              "target injection layout tensors must be CUDA");
  TORCH_CHECK(commit_len.is_contiguous() && prefix_len.is_contiguous()
                  && swa_loc.is_contiguous() && positions.is_contiguous(),
              "target injection layout tensors must be contiguous");
  TORCH_CHECK(commit_len.scalar_type() == torch::kInt32
                  && swa_loc.scalar_type() == torch::kInt32,
              "target injection commit/location tensors must be int32");
  TORCH_CHECK(prefix_len.scalar_type() == torch::kInt64
                  && positions.scalar_type() == torch::kInt64,
              "target injection prefix/position tensors must be int64");
  TORCH_CHECK(commit_len.numel() == 1 && prefix_len.numel() == 1
                  && swa_loc.numel() == 6 && positions.numel() == 6,
              "target injection layout shapes are invalid");
  const c10::cuda::CUDAGuard device_guard(commit_len.device());
  prepare_target_inject_layout_kernel<<<1, 32, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      commit_len.const_data_ptr<int>(),
      prefix_len.const_data_ptr<long>(),
      swa_loc.data_ptr<int>(),
      positions.data_ptr<long>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_target_hc_head_rmsnorm_cuda(
    const torch::Tensor& streams,
    const torch::Tensor& hc_fn,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    const torch::Tensor& norm_weight,
    const torch::Tensor& output,
    double norm_eps,
    double hc_eps) {
  for (const auto* tensor : {&streams, &hc_fn, &hc_scale, &hc_base, &norm_weight, &output}) {
    TORCH_CHECK(tensor->is_cuda(), "target HC-head tensors must be CUDA");
    TORCH_CHECK(tensor->is_contiguous(), "target HC-head tensors must be contiguous");
  }
  TORCH_CHECK(streams.scalar_type() == torch::kBFloat16
                  && norm_weight.scalar_type() == torch::kBFloat16
                  && output.scalar_type() == torch::kBFloat16,
              "target HC-head data tensors must be BF16");
  TORCH_CHECK(hc_fn.scalar_type() == torch::kFloat32
                  && hc_scale.scalar_type() == torch::kFloat32
                  && hc_base.scalar_type() == torch::kFloat32,
              "target HC-head coefficient tensors must be FP32");
  TORCH_CHECK(streams.dim() == 3 && streams.size(0) >= 1 && streams.size(0) <= 6
                  && streams.size(1) == 4 && streams.size(2) == 4096,
              "HC-head streams must have shape [1..6, 4, 4096]");
  TORCH_CHECK(hc_fn.numel() == 4 * 4 * 4096
                  && hc_scale.numel() == 1 && hc_base.numel() == 4,
              "target HC-head coefficient shapes are invalid");
  TORCH_CHECK(norm_weight.numel() == 4096
                  && output.numel() == streams.size(0) * 4096,
              "HC-head output/norm shapes are invalid");
  const c10::cuda::CUDAGuard device_guard(streams.device());
  target_hc_head_rmsnorm_kernel<<<streams.size(0), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(streams.const_data_ptr()),
      hc_fn.const_data_ptr<float>(),
      hc_scale.const_data_ptr<float>(),
      hc_base.const_data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(norm_weight.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<float>(norm_eps),
      static_cast<float>(hc_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_fused_moe_hc_post_cuda(
    const torch::Tensor& routed,
    const torch::Tensor& shared,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output) {
  TORCH_CHECK(
      routed.is_cuda() && shared.is_cuda() && residual.is_cuda()
          && post.is_cuda() && comb.is_cuda() && output.is_cuda(),
      "MHC tensors must be CUDA");
  TORCH_CHECK(
      routed.is_contiguous() && shared.is_contiguous() && residual.is_contiguous()
          && post.is_contiguous() && comb.is_contiguous() && output.is_contiguous(),
      "MHC tensors must be contiguous");
  TORCH_CHECK(
      routed.scalar_type() == torch::kBFloat16
          && shared.scalar_type() == torch::kBFloat16
          && residual.scalar_type() == torch::kBFloat16
          && output.scalar_type() == torch::kBFloat16,
      "MoE and stream tensors must be BF16");
  TORCH_CHECK(
      post.scalar_type() == torch::kFloat32
          && comb.scalar_type() == torch::kFloat32,
      "MHC coefficients must be FP32");
  TORCH_CHECK(routed.numel() == 6 * 4096 && shared.numel() == routed.numel(),
              "MoE output shape is invalid");
  TORCH_CHECK(residual.numel() == 6 * 4 * 4096 && output.numel() == residual.numel(),
              "MHC stream shape is invalid");
  TORCH_CHECK(post.numel() == 6 * 4 && comb.numel() == 6 * 4 * 4,
              "MHC coefficient shape is invalid");
  const c10::cuda::CUDAGuard device_guard(routed.device());
  fused_moe_hc_post_kernel<<<6 * 8, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(routed.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(shared.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
      post.const_data_ptr<float>(),
      comb.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_expert_hc_post_cuda(
    const torch::Tensor& expert_output,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output) {
  TORCH_CHECK(
      expert_output.is_cuda() && residual.is_cuda() && post.is_cuda()
          && comb.is_cuda() && output.is_cuda(),
      "expert MHC tensors must be CUDA");
  TORCH_CHECK(
      expert_output.is_contiguous() && residual.is_contiguous()
          && post.is_contiguous() && comb.is_contiguous() && output.is_contiguous(),
      "expert MHC tensors must be contiguous");
  TORCH_CHECK(
      expert_output.scalar_type() == torch::kBFloat16
          && residual.scalar_type() == torch::kBFloat16
          && output.scalar_type() == torch::kBFloat16,
      "expert and stream tensors must be BF16");
  TORCH_CHECK(
      post.scalar_type() == torch::kFloat32
          && comb.scalar_type() == torch::kFloat32,
      "MHC coefficients must be FP32");
  TORCH_CHECK(
      expert_output.dim() == 2 && expert_output.size(0) >= 1
          && expert_output.size(0) <= 6 && expert_output.size(1) == 4096,
      "expert output must have shape [1..6, 4096]");
  const int64_t rows = expert_output.size(0);
  TORCH_CHECK(
      residual.dim() == 3 && residual.size(0) == rows && residual.size(1) == 4
          && residual.size(2) == 4096 && output.sizes() == residual.sizes(),
      "MHC streams must have shape [rows, 4, 4096]");
  TORCH_CHECK(
      post.dim() == 2 && post.size(0) == rows && post.size(1) == 4
          && comb.dim() == 3 && comb.size(0) == rows && comb.size(1) == 4
          && comb.size(2) == 4,
      "MHC coefficients must have shape [rows, 4] and [rows, 4, 4]");
  const c10::cuda::CUDAGuard device_guard(expert_output.device());
  fused_moe_hc_post_kernel<<<
      rows * 8, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_output.const_data_ptr()),
      nullptr,
      reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
      post.const_data_ptr<float>(),
      comb.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_fused_moe_finalize_hc_post_cuda(
    const torch::Tensor& gemm2,
    const torch::Tensor& expert_weights,
    const torch::Tensor& expanded_to_permuted,
    const torch::Tensor& shared,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output) {
  TORCH_CHECK(
      gemm2.is_cuda() && expert_weights.is_cuda()
          && expanded_to_permuted.is_cuda() && shared.is_cuda()
          && residual.is_cuda() && post.is_cuda() && comb.is_cuda()
          && output.is_cuda(),
      "fused finalize tensors must be CUDA");
  TORCH_CHECK(
      gemm2.is_contiguous() && expert_weights.is_contiguous()
          && expanded_to_permuted.is_contiguous() && shared.is_contiguous()
          && residual.is_contiguous() && post.is_contiguous()
          && comb.is_contiguous() && output.is_contiguous(),
      "fused finalize tensors must be contiguous");
  TORCH_CHECK(
      gemm2.scalar_type() == torch::kBFloat16
          && expert_weights.scalar_type() == torch::kBFloat16
          && shared.scalar_type() == torch::kBFloat16
          && residual.scalar_type() == torch::kBFloat16
          && output.scalar_type() == torch::kBFloat16,
      "fused finalize data tensors must be BF16");
  TORCH_CHECK(
      expanded_to_permuted.scalar_type() == torch::kInt32,
      "expanded-to-permuted map must be int32");
  TORCH_CHECK(
      post.scalar_type() == torch::kFloat32
          && comb.scalar_type() == torch::kFloat32,
      "MHC coefficients must be FP32");
  TORCH_CHECK(gemm2.dim() == 2 && gemm2.size(1) >= 4096,
              "gemm2 output must be [padded_tokens, >=4096]");
  TORCH_CHECK(expert_weights.numel() == 6 * 6
                  && expanded_to_permuted.numel() == 6 * 6,
              "fused finalize requires batch6/topk6 routing metadata");
  TORCH_CHECK(shared.numel() == 6 * 4096,
              "shared expert output shape is invalid");
  TORCH_CHECK(residual.numel() == 6 * 4 * 4096
                  && output.numel() == residual.numel(),
              "MHC stream shape is invalid");
  TORCH_CHECK(post.numel() == 6 * 4 && comb.numel() == 6 * 4 * 4,
              "MHC coefficient shape is invalid");
  TORCH_CHECK(gemm2.size(1) <= INT_MAX, "gemm2 stride is too large");
  const c10::cuda::CUDAGuard device_guard(gemm2.device());
  fused_moe_finalize_hc_post_kernel<<<
      6 * 8, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2.const_data_ptr()),
      static_cast<int>(gemm2.size(1)),
      reinterpret_cast<const __nv_bfloat16*>(expert_weights.const_data_ptr()),
      expanded_to_permuted.const_data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(shared.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
      post.const_data_ptr<float>(),
      comb.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
