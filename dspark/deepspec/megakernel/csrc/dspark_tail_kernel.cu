#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

namespace cg = cooperative_groups;

constexpr int kThreads = 256;
constexpr int kTraceColumns = 12;

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void grid_barrier(
    int*,
    int*,
    int) {
  cg::this_grid().sync();
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

#ifdef DSPARK_ENABLE_DEVICE_TRACE
__device__ __forceinline__ void record_timestamp(
    int64_t* trace,
    int block,
    int step,
    int steps,
    int column) {
  if (threadIdx.x == 0) {
    trace[(static_cast<int64_t>(block) * steps + step) * kTraceColumns + column] =
        static_cast<int64_t>(globaltimer_ns());
  }
}
#else
__device__ __forceinline__ void record_timestamp(
    int64_t*, int, int, int, int) {}
#endif

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void dspark_tail_kernel(
    const scalar_t* __restrict__ base_logits,
    const scalar_t* __restrict__ hidden_states,
    const int64_t* __restrict__ first_prev_token,
    const scalar_t* __restrict__ markov_w1,
    const scalar_t* __restrict__ markov_w2,
    const scalar_t* __restrict__ confidence_weight,
    const scalar_t* __restrict__ confidence_bias,
    const float* __restrict__ uniforms,
    float temperature,
    int steps,
    int vocab_size,
    int hidden_size,
    int rank,
    int* __restrict__ barrier_state,
    float* __restrict__ block_max,
    int* __restrict__ block_index,
    float* __restrict__ block_sum,
    float* __restrict__ global_values,
    int64_t* __restrict__ token_ids,
    float* __restrict__ corrected_logits,
    float* __restrict__ draft_probs,
    float* __restrict__ confidence_logits,
    int64_t* __restrict__ trace) {
  __shared__ float reduce_values[kThreads];
  __shared__ int reduce_indices[kThreads];
  const int block = blockIdx.x;
  const int num_blocks = gridDim.x;
  const int global_thread = block * blockDim.x + threadIdx.x;
  const int global_stride = num_blocks * blockDim.x;
  int64_t previous_token = first_prev_token[0];

  for (int step = 0; step < steps; ++step) {
    record_timestamp(trace, block, step, steps, 0);
    float local_max = -3.402823466e+38F;
    int local_index = vocab_size;
    for (int vocab = global_thread; vocab < vocab_size; vocab += global_stride) {
      float dot = 0.0f;
      const int64_t embedding_offset = previous_token * static_cast<int64_t>(rank);
      const int64_t weight_offset = static_cast<int64_t>(vocab) * rank;
      for (int column = 0; column < rank; ++column) {
        dot = fmaf(
            load_as_float(markov_w1, embedding_offset + column),
            load_as_float(markov_w2, weight_offset + column),
            dot);
      }
      // Match the released BF16 linear followed by BF16 add: the projected
      // bias and corrected logit each cross an activation-dtype boundary.
      const scalar_t rounded_bias = static_cast<scalar_t>(dot);
      const scalar_t rounded_logit = static_cast<scalar_t>(
          load_as_float(base_logits, static_cast<int64_t>(step) * vocab_size + vocab) +
          static_cast<float>(rounded_bias));
      const float value = static_cast<float>(rounded_logit);
      corrected_logits[static_cast<int64_t>(step) * vocab_size + vocab] = value;
      if (value > local_max || (value == local_max && vocab < local_index)) {
        local_max = value;
        local_index = vocab;
      }
    }
    reduce_values[threadIdx.x] = local_max;
    reduce_indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        const float other_value = reduce_values[threadIdx.x + offset];
        const int other_index = reduce_indices[threadIdx.x + offset];
        if (other_value > reduce_values[threadIdx.x] ||
            (other_value == reduce_values[threadIdx.x] &&
             other_index < reduce_indices[threadIdx.x])) {
          reduce_values[threadIdx.x] = other_value;
          reduce_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      block_max[block] = reduce_values[0];
      block_index[block] = reduce_indices[0];
    }
    record_timestamp(trace, block, step, steps, 1);

    record_timestamp(trace, block, step, steps, 2);
    grid_barrier(barrier_state, barrier_state + 1, num_blocks);
    if (block == 0 && threadIdx.x == 0) {
      float best = -3.402823466e+38F;
      int best_index = vocab_size;
      for (int candidate = 0; candidate < num_blocks; ++candidate) {
        const float value = block_max[candidate];
        const int index = block_index[candidate];
        if (value > best || (value == best && index < best_index)) {
          best = value;
          best_index = index;
        }
      }
      global_values[0] = best;
      global_values[1] = static_cast<float>(best_index);
    }
    grid_barrier(barrier_state, barrier_state + 1, num_blocks);
    record_timestamp(trace, block, step, steps, 3);

    record_timestamp(trace, block, step, steps, 4);
    float local_sum = 0.0f;
    if (temperature >= 1.0e-5f) {
      const float maximum = global_values[0];
      for (int vocab = global_thread; vocab < vocab_size; vocab += global_stride) {
        const int64_t offset = static_cast<int64_t>(step) * vocab_size + vocab;
        const float value = expf((corrected_logits[offset] - maximum) / temperature);
        draft_probs[offset] = value;
        local_sum += value;
      }
    } else {
      const int chosen = static_cast<int>(global_values[1]);
      for (int vocab = global_thread; vocab < vocab_size; vocab += global_stride) {
        const int64_t offset = static_cast<int64_t>(step) * vocab_size + vocab;
        draft_probs[offset] = vocab == chosen ? 1.0f : 0.0f;
      }
    }
    reduce_values[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduce_values[threadIdx.x] += reduce_values[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      block_sum[block] = reduce_values[0];
    }
    record_timestamp(trace, block, step, steps, 5);

    record_timestamp(trace, block, step, steps, 6);
    grid_barrier(barrier_state, barrier_state + 1, num_blocks);
    if (block == 0 && threadIdx.x == 0 && temperature >= 1.0e-5f) {
      float total = 0.0f;
      for (int candidate = 0; candidate < num_blocks; ++candidate) {
        total += block_sum[candidate];
      }
      global_values[0] = total;
    }
    grid_barrier(barrier_state, barrier_state + 1, num_blocks);
    record_timestamp(trace, block, step, steps, 7);

    record_timestamp(trace, block, step, steps, 8);
    if (temperature >= 1.0e-5f) {
      const float inverse_sum = 1.0f / global_values[0];
      for (int vocab = global_thread; vocab < vocab_size; vocab += global_stride) {
        const int64_t offset = static_cast<int64_t>(step) * vocab_size + vocab;
        draft_probs[offset] *= inverse_sum;
      }
    }
    record_timestamp(trace, block, step, steps, 9);

    record_timestamp(trace, block, step, steps, 10);
    grid_barrier(barrier_state, barrier_state + 1, num_blocks);
    if (block == 0 && threadIdx.x == 0) {
      int chosen = static_cast<int>(global_values[1]);
      if (temperature >= 1.0e-5f) {
        const float threshold = uniforms[step];
        float cumulative = 0.0f;
        chosen = vocab_size - 1;
        for (int vocab = 0; vocab < vocab_size; ++vocab) {
          cumulative += draft_probs[static_cast<int64_t>(step) * vocab_size + vocab];
          if (cumulative >= threshold) {
            chosen = vocab;
            break;
          }
        }
      }
      token_ids[step] = chosen;

      float confidence = 0.0f;
      const int64_t hidden_offset = static_cast<int64_t>(step) * hidden_size;
      for (int column = 0; column < hidden_size; ++column) {
        confidence = fmaf(
            load_as_float(hidden_states, hidden_offset + column),
            load_as_float(confidence_weight, column),
            confidence);
      }
      const int64_t embedding_offset = previous_token * static_cast<int64_t>(rank);
      for (int column = 0; column < rank; ++column) {
        confidence = fmaf(
            load_as_float(markov_w1, embedding_offset + column),
            load_as_float(confidence_weight, hidden_size + column),
            confidence);
      }
      confidence += load_as_float(confidence_bias, 0);
      confidence_logits[step] = static_cast<float>(static_cast<scalar_t>(confidence));
      global_values[1] = static_cast<float>(chosen);
    }
    grid_barrier(barrier_state, barrier_state + 1, num_blocks);
    // `global_values[1]` is reused for the following step's argmax. Reading
    // the committed token from its immutable per-step slot avoids a reuse
    // race between blocks leaving this barrier at slightly different times.
    previous_token = token_ids[step];
    record_timestamp(trace, block, step, steps, 11);
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> dspark_tail_cuda(
    const torch::Tensor& base_logits,
    const torch::Tensor& hidden_states,
    const torch::Tensor& first_prev_token,
    const torch::Tensor& markov_w1,
    const torch::Tensor& markov_w2,
    const torch::Tensor& confidence_weight,
    const torch::Tensor& confidence_bias,
    const torch::Tensor& uniforms,
    double temperature,
    const torch::Tensor& barrier_state,
    const torch::Tensor& block_max,
    const torch::Tensor& block_index,
    const torch::Tensor& block_sum,
    const torch::Tensor& global_values,
    const torch::Tensor& token_ids,
    const torch::Tensor& corrected_logits,
    const torch::Tensor& draft_probs,
    const torch::Tensor& confidence_logits,
    const torch::Tensor& trace) {
  check_cuda_contiguous(base_logits, "base_logits");
  check_cuda_contiguous(hidden_states, "hidden_states");
  check_cuda_contiguous(first_prev_token, "first_prev_token");
  check_cuda_contiguous(markov_w1, "markov_w1");
  check_cuda_contiguous(markov_w2, "markov_w2");
  check_cuda_contiguous(confidence_weight, "confidence_weight");
  check_cuda_contiguous(confidence_bias, "confidence_bias");
  check_cuda_contiguous(uniforms, "uniforms");
  TORCH_CHECK(base_logits.dim() == 2, "base_logits must be [steps, vocab]");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [steps, hidden]");
  TORCH_CHECK(markov_w1.dim() == 2, "markov_w1 must be [vocab, rank]");
  TORCH_CHECK(markov_w2.sizes() == markov_w1.sizes(), "markov_w2 shape mismatch");
  TORCH_CHECK(base_logits.size(0) == hidden_states.size(0), "step mismatch");
  TORCH_CHECK(base_logits.size(1) == markov_w1.size(0), "vocab mismatch");
  TORCH_CHECK(confidence_weight.numel() == hidden_states.size(1) + markov_w1.size(1),
              "confidence weight width mismatch");
  TORCH_CHECK(first_prev_token.scalar_type() == torch::kInt64, "token must be int64");
  TORCH_CHECK(uniforms.scalar_type() == torch::kFloat32, "uniforms must be float32");
  TORCH_CHECK(temperature >= 0.0, "temperature must be non-negative");
  TORCH_CHECK(base_logits.scalar_type() == hidden_states.scalar_type(), "dtype mismatch");
  TORCH_CHECK(base_logits.scalar_type() == markov_w1.scalar_type(), "dtype mismatch");
  TORCH_CHECK(base_logits.scalar_type() == markov_w2.scalar_type(), "dtype mismatch");
  TORCH_CHECK(base_logits.scalar_type() == confidence_weight.scalar_type(), "dtype mismatch");
  TORCH_CHECK(base_logits.scalar_type() == confidence_bias.scalar_type(), "dtype mismatch");

  const c10::cuda::CUDAGuard device_guard(base_logits.device());
  int device = base_logits.get_device();
  int blocks = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(blocks > 0, "GPU reports no multiprocessors");
  int cooperative = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative != 0, "DSpark tail requires cooperative launch support");
  TORCH_CHECK(block_max.numel() >= blocks, "block_max workspace too small");
  TORCH_CHECK(block_index.numel() >= blocks, "block_index workspace too small");
  TORCH_CHECK(block_sum.numel() >= blocks, "block_sum workspace too small");
  TORCH_CHECK(trace.numel() >= static_cast<int64_t>(blocks) * base_logits.size(0) * kTraceColumns,
              "trace workspace too small");

  const auto stream = at::cuda::getCurrentCUDAStream(device);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      base_logits.scalar_type(),
      "dspark_tail_kernel",
      [&] {
        int blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            dspark_tail_kernel<scalar_t>,
            kThreads,
            0));
        TORCH_CHECK(blocks_per_sm >= 1, "DSpark tail kernel cannot reside on the GPU");
        auto* base_ptr = base_logits.data_ptr<scalar_t>();
        auto* hidden_ptr = hidden_states.data_ptr<scalar_t>();
        auto* previous_ptr = first_prev_token.data_ptr<int64_t>();
        auto* w1_ptr = markov_w1.data_ptr<scalar_t>();
        auto* w2_ptr = markov_w2.data_ptr<scalar_t>();
        auto* confidence_weight_ptr = confidence_weight.data_ptr<scalar_t>();
        auto* confidence_bias_ptr = confidence_bias.data_ptr<scalar_t>();
        auto* uniforms_ptr = uniforms.data_ptr<float>();
        float temperature_value = static_cast<float>(temperature);
        int steps = static_cast<int>(base_logits.size(0));
        int vocab = static_cast<int>(base_logits.size(1));
        int hidden = static_cast<int>(hidden_states.size(1));
        int rank = static_cast<int>(markov_w1.size(1));
        auto* barrier_ptr = barrier_state.data_ptr<int>();
        auto* block_max_ptr = block_max.data_ptr<float>();
        auto* block_index_ptr = block_index.data_ptr<int>();
        auto* block_sum_ptr = block_sum.data_ptr<float>();
        auto* global_ptr = global_values.data_ptr<float>();
        auto* tokens_ptr = token_ids.data_ptr<int64_t>();
        auto* corrected_ptr = corrected_logits.data_ptr<float>();
        auto* probabilities_ptr = draft_probs.data_ptr<float>();
        auto* confidence_ptr = confidence_logits.data_ptr<float>();
        auto* trace_ptr = trace.data_ptr<int64_t>();
        void* arguments[] = {
            &base_ptr,
            &hidden_ptr,
            &previous_ptr,
            &w1_ptr,
            &w2_ptr,
            &confidence_weight_ptr,
            &confidence_bias_ptr,
            &uniforms_ptr,
            &temperature_value,
            &steps,
            &vocab,
            &hidden,
            &rank,
            &barrier_ptr,
            &block_max_ptr,
            &block_index_ptr,
            &block_sum_ptr,
            &global_ptr,
            &tokens_ptr,
            &corrected_ptr,
            &probabilities_ptr,
            &confidence_ptr,
            &trace_ptr,
        };
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(dspark_tail_kernel<scalar_t>),
            dim3(blocks),
            dim3(kThreads),
            arguments,
            0,
            stream.stream()));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {token_ids, corrected_logits, draft_probs, confidence_logits, trace};
}
