#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <vector>

namespace {

struct alignas(16) CompressPlan {
  uint32_t seq_len;
  uint16_t ragged_id;
  uint16_t buffer_len;
  int32_t read_page_0;
  int32_t read_page_1;
};

struct alignas(8) WritePlan {
  uint32_t ragged_id;
  int32_t write_loc;
};

static_assert(sizeof(CompressPlan) == 16);
static_assert(sizeof(WritePlan) == 8);

__device__ __forceinline__ CompressPlan invalid_compress_plan() {
  return CompressPlan{UINT32_MAX, 0, 0, -1, -1};
}

__device__ __forceinline__ WritePlan invalid_write_plan() {
  return WritePlan{UINT32_MAX, -1};
}

__global__ void tail_relaunch_kernel(int* iteration, int iterations) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int completed = atomicAdd(iteration, 1) + 1;
    if (completed < iterations) {
      cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    }
  }
}

__device__ __forceinline__ void execute_full_loop_state_update(
    const int64_t* new_seq_len,
    int64_t* prefix_len,
    int32_t* positions,
    int32_t* window_locations,
    CompressPlan* c4_plan_c,
    WritePlan* c4_plan_w,
    CompressPlan* c128_plan_c,
    WritePlan* c128_plan_w,
    int64_t* c4_target_out_locations,
    int64_t* c4_index_out_locations,
    int32_t* c4_index_context_lens,
    int64_t* c128_target_out_locations,
    int32_t* c128_extra_topk_lengths,
    int64_t* graft_seq_lens,
    int64_t* graft_positions,
    int64_t* graft_out_cache_loc) {
  const int row = static_cast<int>(threadIdx.x);
  const int64_t prefix = new_seq_len[0];
  if (row == 0) {
    prefix_len[0] = prefix;
    graft_seq_lens[0] = prefix;
  }
  if (row < 6) {
    const int32_t position = static_cast<int32_t>(prefix + row);
    positions[row] = position;
    window_locations[row] = position & 127;
    if (row < 5) {
      graft_positions[row] = prefix + row;
      graft_out_cache_loc[row] = (prefix + row) & 127;
    }
    c4_target_out_locations[row] = 128 + position / 4;
    c4_index_out_locations[row] = position / 4;
    c4_index_context_lens[row] = max((position + 1) / 4, 1);
    c128_target_out_locations[row] = 128 + position / 128;
    c128_extra_topk_lengths[row] = (position + 1) / 128;

    c4_plan_c[row] = invalid_compress_plan();
    c4_plan_w[row] = invalid_write_plan();
    c128_plan_c[row] = invalid_compress_plan();
    c128_plan_w[row] = invalid_write_plan();
  }
  __syncthreads();

  if (row < 6) {
    const int32_t position = static_cast<int32_t>(prefix + row);
    if ((position + 1) % 4 == 0) {
      int slot = 0;
#pragma unroll
      for (int previous = 0; previous < row; ++previous) {
        slot += ((prefix + previous + 1) % 4 == 0);
      }
      const int32_t position_0 = max(position - 4, 0);
      c4_plan_c[slot] = CompressPlan{
          static_cast<uint32_t>(position + 1),
          static_cast<uint16_t>(row),
          static_cast<uint16_t>(8 - min(row + 1, 8)),
          (position_0 / 4) & 1,
          (position / 4) & 1};
    }
    const int32_t c4_seq_len = static_cast<int32_t>(prefix + 6);
    const int32_t c4_last_compressed = (c4_seq_len / 4) * 4;
    const int32_t c4_first_write = min(c4_last_compressed - 4, c4_seq_len - 4);
    const bool c4_write = position >= c4_first_write;
    if (c4_write) {
      int slot = 0;
#pragma unroll
      for (int previous = 0; previous < row; ++previous) {
        slot += static_cast<int32_t>(prefix + previous) >= c4_first_write;
      }
      c4_plan_w[slot] = WritePlan{
          static_cast<uint32_t>(row),
          position & 7};
    }

    if ((position + 1) % 128 == 0) {
      int slot = 0;
#pragma unroll
      for (int previous = 0; previous < row; ++previous) {
        slot += ((prefix + previous + 1) % 128 == 0);
      }
      c128_plan_c[slot] = CompressPlan{
          static_cast<uint32_t>(position + 1),
          static_cast<uint16_t>(row),
          static_cast<uint16_t>(128 - min(row + 1, 128)),
          0,
          0};
    }
    const int32_t c128_seq_len = static_cast<int32_t>(prefix + 6);
    const int32_t c128_first_write = (c128_seq_len / 128) * 128;
    const bool c128_write = position >= c128_first_write;
    if (c128_write) {
      int slot = 0;
#pragma unroll
      for (int previous = 0; previous < row; ++previous) {
        slot += static_cast<int32_t>(prefix + previous) >= c128_first_write;
      }
      c128_plan_w[slot] = WritePlan{
          static_cast<uint32_t>(row),
          position & 127};
    }
  }
}

__global__ void full_loop_state_update_only_kernel(
    const int64_t* new_seq_len,
    int64_t* prefix_len,
    int32_t* positions,
    int32_t* window_locations,
    CompressPlan* c4_plan_c,
    WritePlan* c4_plan_w,
    CompressPlan* c128_plan_c,
    WritePlan* c128_plan_w,
    int64_t* c4_target_out_locations,
    int64_t* c4_index_out_locations,
    int32_t* c4_index_context_lens,
    int64_t* c128_target_out_locations,
    int32_t* c128_extra_topk_lengths,
    int64_t* graft_seq_lens,
    int64_t* graft_positions,
    int64_t* graft_out_cache_loc) {
  execute_full_loop_state_update(
      new_seq_len, prefix_len, positions, window_locations,
      c4_plan_c, c4_plan_w, c128_plan_c, c128_plan_w,
      c4_target_out_locations, c4_index_out_locations,
      c4_index_context_lens, c128_target_out_locations,
      c128_extra_topk_lengths, graft_seq_lens, graft_positions,
      graft_out_cache_loc);
}

template <bool Vectorized>
__global__ void prepare_proposal_megakernel_kernel(
    const int64_t* bonus,
    const int32_t* commit_len,
    const int64_t* new_seq_len,
    const __nv_bfloat16* target_hidden,
    const float* freqs_real,
    int32_t* anchor,
    __nv_bfloat16* main_hidden,
    float* rope,
    int64_t* start_pos,
    int64_t* epoch) {
  const int commit_row = max(0, min(commit_len[0] - 1, 5));
  const int64_t position = new_seq_len[0] - 1;
  if (threadIdx.x == 0) {
    anchor[0] = static_cast<int32_t>(bonus[0]);
    start_pos[0] = position;
    epoch[0] += 1;
  }
  if constexpr (Vectorized) {
    // All row strides are multiples of 16 bytes. Host validation below also
    // checks storage offsets, since contiguous tensor views can be unaligned.
    const auto* source = reinterpret_cast<const uint4*>(
        target_hidden + commit_row * 3 * 4096);
    auto* destination = reinterpret_cast<uint4*>(main_hidden);
#pragma unroll
    for (int packet = threadIdx.x; packet < 3 * 4096 / 8; packet += 256) {
      destination[packet] = source[packet];
    }
    if (threadIdx.x < 6 * 32 * 2 / 4) {
      reinterpret_cast<uint4*>(rope)[threadIdx.x] =
          reinterpret_cast<const uint4*>(freqs_real + position * 64)[threadIdx.x];
    }
  } else {
  for (int column = threadIdx.x; column < 3 * 4096; column += blockDim.x) {
    main_hidden[column] = target_hidden[commit_row * 3 * 4096 + column];
  }
  for (int element = threadIdx.x; element < 6 * 32 * 2; element += blockDim.x) {
    const int row = element / (32 * 2);
    const int feature = element % (32 * 2);
    rope[element] = freqs_real[(position + row) * 32 * 2 + feature];
  }
  }
}

__global__ void publish_proposal_candidates_kernel(
    const int32_t* output_ids,
    int64_t* candidates) {
  if (threadIdx.x < 6) {
    candidates[threadIdx.x] = static_cast<int64_t>(output_ids[threadIdx.x]);
  }
}

__global__ void full_loop_state_update_relaunch_kernel(
    int* iteration,
    int iterations,
    const int64_t* new_seq_len,
    int64_t* prefix_len,
    int32_t* positions,
    int32_t* window_locations,
    CompressPlan* c4_plan_c,
    WritePlan* c4_plan_w,
    CompressPlan* c128_plan_c,
    WritePlan* c128_plan_w,
    int64_t* c4_target_out_locations,
    int64_t* c4_index_out_locations,
    int32_t* c4_index_context_lens,
    int64_t* c128_target_out_locations,
    int32_t* c128_extra_topk_lengths,
    int64_t* graft_seq_lens,
    int64_t* graft_positions,
    int64_t* graft_out_cache_loc) {
  execute_full_loop_state_update(
      new_seq_len, prefix_len, positions, window_locations,
      c4_plan_c, c4_plan_w, c128_plan_c, c128_plan_w,
      c4_target_out_locations, c4_index_out_locations,
      c4_index_context_lens, c128_target_out_locations,
      c128_extra_topk_lengths, graft_seq_lens, graft_positions,
      graft_out_cache_loc);
  __syncthreads();
  if (threadIdx.x == 0) {
    const int completed = atomicAdd(iteration, 1) + 1;
    if (completed < iterations) {
      // Tail launch is ordered after the whole parent graph. Issuing it from
      // this earlier controller overlaps launch preparation with the graft.
      cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    }
  }
}

cudaGraph_t as_graph(int64_t handle) {
  TORCH_CHECK(handle != 0, "CUDA graph handle must be nonzero");
  return reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(handle));
}

cudaGraphExec_t as_executable(int64_t handle) {
  TORCH_CHECK(handle != 0, "CUDA graph executable handle must be nonzero");
  return reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(handle));
}

}  // namespace

int64_t dspark_instantiate_device_graph_cuda(int64_t graph_handle) {
  cudaGraphInstantiateParams parameters{};
  parameters.flags =
      cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
  parameters.uploadStream = at::cuda::getCurrentCUDAStream();
  cudaGraphExec_t executable = nullptr;
  C10_CUDA_CHECK(cudaGraphInstantiateWithParams(
      &executable, as_graph(graph_handle), &parameters));
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(executable));
}

namespace {

int64_t instantiate_composed_graph(
    const std::vector<int64_t>& graph_handles,
    bool device_launch) {
  TORCH_CHECK(!graph_handles.empty(), "composed CUDA graph requires child graphs");
  cudaGraph_t parent = nullptr;
  cudaGraphExec_t executable = nullptr;
  C10_CUDA_CHECK(cudaGraphCreate(&parent, 0));
  try {
    cudaGraphNode_t previous = nullptr;
    for (int64_t handle : graph_handles) {
      cudaGraphNode_t child = nullptr;
      const cudaGraphNode_t* dependencies = previous == nullptr ? nullptr : &previous;
      const size_t dependency_count = previous == nullptr ? 0 : 1;
      C10_CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child,
          parent,
          dependencies,
          dependency_count,
          as_graph(handle)));
      previous = child;
    }
    cudaGraphInstantiateParams parameters{};
    parameters.flags = cudaGraphInstantiateFlagUpload;
    if (device_launch) {
      parameters.flags |= cudaGraphInstantiateFlagDeviceLaunch;
    }
    parameters.uploadStream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaGraphInstantiateWithParams(&executable, parent, &parameters));
    C10_CUDA_CHECK(cudaGraphDestroy(parent));
    parent = nullptr;
  } catch (...) {
    // Preserve the original error while releasing resources already acquired.
    if (executable != nullptr) {
      cudaGraphExecDestroy(executable);
    }
    if (parent != nullptr) {
      cudaGraphDestroy(parent);
    }
    throw;
  }
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(executable));
}

}  // namespace

int64_t dspark_instantiate_composed_device_graph_cuda(
    const std::vector<int64_t>& graph_handles) {
  return instantiate_composed_graph(graph_handles, true);
}

int64_t dspark_instantiate_composed_host_graph_cuda(
    const std::vector<int64_t>& graph_handles) {
  return instantiate_composed_graph(graph_handles, false);
}

void dspark_launch_device_graph_cuda(int64_t executable_handle) {
  C10_CUDA_CHECK(cudaGraphLaunch(
      as_executable(executable_handle), at::cuda::getCurrentCUDAStream()));
}

void dspark_destroy_device_graph_cuda(int64_t executable_handle) {
  C10_CUDA_CHECK(cudaGraphExecDestroy(as_executable(executable_handle)));
}

void dspark_launch_tail_marker_cuda(
    const torch::Tensor& iteration,
    int64_t iterations) {
  TORCH_CHECK(iteration.is_cuda(), "iteration must be CUDA");
  TORCH_CHECK(iteration.is_contiguous(), "iteration must be contiguous");
  TORCH_CHECK(iteration.scalar_type() == torch::kInt32, "iteration must be int32");
  TORCH_CHECK(iteration.numel() == 1, "iteration must contain one value");
  TORCH_CHECK(iterations > 0 && iterations <= INT_MAX, "iterations out of range");
  const c10::cuda::CUDAGuard device_guard(iteration.device());
  tail_relaunch_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      iteration.data_ptr<int>(), static_cast<int>(iterations));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_launch_full_loop_state_update_cuda(
    const torch::Tensor& iteration,
    int64_t iterations,
    const torch::Tensor& new_seq_len,
    const torch::Tensor& prefix_len,
    const torch::Tensor& positions,
    const torch::Tensor& window_locations,
    const torch::Tensor& c4_plan_c,
    const torch::Tensor& c4_plan_w,
    const torch::Tensor& c128_plan_c,
    const torch::Tensor& c128_plan_w,
    const torch::Tensor& c4_target_out_locations,
    const torch::Tensor& c4_index_out_locations,
    const torch::Tensor& c4_index_context_lens,
    const torch::Tensor& c128_target_out_locations,
    const torch::Tensor& c128_extra_topk_lengths,
    const torch::Tensor& graft_seq_lens,
    const torch::Tensor& graft_positions,
    const torch::Tensor& graft_out_cache_loc) {
  const auto require = [&](const torch::Tensor& tensor,
                           torch::ScalarType dtype,
                           int64_t elements,
                           const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.numel() == elements, name, " has wrong size");
    TORCH_CHECK(tensor.get_device() == iteration.get_device(), name, " is on another device");
  };
  require(iteration, torch::kInt32, 1, "iteration");
  require(new_seq_len, torch::kInt64, 1, "new_seq_len");
  require(prefix_len, torch::kInt64, 1, "prefix_len");
  require(positions, torch::kInt32, 6, "positions");
  require(window_locations, torch::kInt32, 6, "window_locations");
  require(c4_plan_c, torch::kUInt8, 6 * 16, "c4_plan_c");
  require(c4_plan_w, torch::kUInt8, 6 * 8, "c4_plan_w");
  require(c128_plan_c, torch::kUInt8, 6 * 16, "c128_plan_c");
  require(c128_plan_w, torch::kUInt8, 6 * 8, "c128_plan_w");
  require(c4_target_out_locations, torch::kInt64, 6, "c4_target_out_locations");
  require(c4_index_out_locations, torch::kInt64, 6, "c4_index_out_locations");
  require(c4_index_context_lens, torch::kInt32, 6, "c4_index_context_lens");
  require(c128_target_out_locations, torch::kInt64, 6, "c128_target_out_locations");
  require(c128_extra_topk_lengths, torch::kInt32, 6, "c128_extra_topk_lengths");
  require(graft_seq_lens, torch::kInt64, 1, "graft_seq_lens");
  require(graft_positions, torch::kInt64, 5, "graft_positions");
  require(graft_out_cache_loc, torch::kInt64, 5, "graft_out_cache_loc");
  TORCH_CHECK(iterations >= 0 && iterations <= INT_MAX, "iterations out of range");
  TORCH_CHECK(new_seq_len.data_ptr<int64_t>() != prefix_len.data_ptr<int64_t>(),
              "new_seq_len and prefix_len must not alias");
  const c10::cuda::CUDAGuard device_guard(iteration.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
#define DSPARK_STATE_ARGS \
      new_seq_len.data_ptr<int64_t>(), prefix_len.data_ptr<int64_t>(), \
      positions.data_ptr<int32_t>(), window_locations.data_ptr<int32_t>(), \
      reinterpret_cast<CompressPlan*>(c4_plan_c.data_ptr<uint8_t>()), \
      reinterpret_cast<WritePlan*>(c4_plan_w.data_ptr<uint8_t>()), \
      reinterpret_cast<CompressPlan*>(c128_plan_c.data_ptr<uint8_t>()), \
      reinterpret_cast<WritePlan*>(c128_plan_w.data_ptr<uint8_t>()), \
      c4_target_out_locations.data_ptr<int64_t>(), \
      c4_index_out_locations.data_ptr<int64_t>(), \
      c4_index_context_lens.data_ptr<int32_t>(), \
      c128_target_out_locations.data_ptr<int64_t>(), \
      c128_extra_topk_lengths.data_ptr<int32_t>(), \
      graft_seq_lens.data_ptr<int64_t>(), graft_positions.data_ptr<int64_t>(), \
      graft_out_cache_loc.data_ptr<int64_t>()
  if (iterations == 0) {
    full_loop_state_update_only_kernel<<<1, 32, 0, stream>>>(
        DSPARK_STATE_ARGS);
  } else {
    full_loop_state_update_relaunch_kernel<<<1, 32, 0, stream>>>(
        iteration.data_ptr<int>(), static_cast<int>(iterations),
        DSPARK_STATE_ARGS);
  }
#undef DSPARK_STATE_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_prepare_proposal_megakernel_cuda(
    const torch::Tensor& bonus,
    const torch::Tensor& commit_len,
    const torch::Tensor& new_seq_len,
    const torch::Tensor& target_hidden,
    const torch::Tensor& freqs_real,
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& rope,
    const torch::Tensor& start_pos,
    const torch::Tensor& epoch,
    bool vectorized) {
  const auto require = [&](const torch::Tensor& tensor,
                           torch::ScalarType dtype,
                           int64_t elements,
                           const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous CUDA");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.numel() == elements, name, " has wrong size");
    TORCH_CHECK(tensor.get_device() == bonus.get_device(), name, " is on another device");
  };
  require(bonus, torch::kInt64, 1, "bonus");
  require(commit_len, torch::kInt32, 1, "commit_len");
  require(new_seq_len, torch::kInt64, 1, "new_seq_len");
  require(target_hidden, torch::kBFloat16, 6 * 3 * 4096, "target_hidden");
  TORCH_CHECK(freqs_real.is_cuda() && freqs_real.is_contiguous(), "freqs_real must be contiguous CUDA");
  TORCH_CHECK(freqs_real.scalar_type() == torch::kFloat32, "freqs_real has wrong dtype");
  TORCH_CHECK(freqs_real.numel() >= 6 * 32 * 2, "freqs_real is too small");
  TORCH_CHECK(freqs_real.get_device() == bonus.get_device(), "freqs_real is on another device");
  require(anchor, torch::kInt32, 1, "anchor");
  require(main_hidden, torch::kBFloat16, 3 * 4096, "main_hidden");
  require(rope, torch::kFloat32, 6 * 32 * 2, "rope");
  require(start_pos, torch::kInt64, 1, "start_pos");
  require(epoch, torch::kInt64, 1, "epoch");
  const c10::cuda::CUDAGuard device_guard(bonus.device());
  const auto aligned = [](const torch::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
  };
  const bool use_vectorized = vectorized && aligned(target_hidden)
      && aligned(main_hidden) && aligned(freqs_real) && aligned(rope);
  auto kernel = use_vectorized ? prepare_proposal_megakernel_kernel<true>
                              : prepare_proposal_megakernel_kernel<false>;
  kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      bonus.data_ptr<int64_t>(), commit_len.data_ptr<int32_t>(),
      new_seq_len.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(target_hidden.data_ptr<at::BFloat16>()),
      freqs_real.data_ptr<float>(), anchor.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(main_hidden.data_ptr<at::BFloat16>()),
      rope.data_ptr<float>(), start_pos.data_ptr<int64_t>(), epoch.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dspark_publish_proposal_candidates_cuda(
    const torch::Tensor& output_ids,
    const torch::Tensor& candidates) {
  TORCH_CHECK(output_ids.is_cuda() && output_ids.is_contiguous(), "output_ids must be contiguous CUDA");
  TORCH_CHECK(candidates.is_cuda() && candidates.is_contiguous(), "candidates must be contiguous CUDA");
  TORCH_CHECK(output_ids.scalar_type() == torch::kInt32 && output_ids.numel() == 6,
              "output_ids must be int32 [1,6]");
  TORCH_CHECK(candidates.scalar_type() == torch::kInt64 && candidates.numel() == 6,
              "candidates must be int64 [6]");
  TORCH_CHECK(output_ids.get_device() == candidates.get_device(), "candidate tensors are on different devices");
  const c10::cuda::CUDAGuard device_guard(output_ids.device());
  publish_proposal_candidates_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      output_ids.data_ptr<int32_t>(), candidates.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> dspark_device_graph_node_types_cuda(int64_t graph_handle) {
  size_t count = 0;
  cudaGraph_t graph = as_graph(graph_handle);
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
  std::vector<int64_t> types;
  types.reserve(count);
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    types.push_back(static_cast<int64_t>(type));
  }
  return types;
}
