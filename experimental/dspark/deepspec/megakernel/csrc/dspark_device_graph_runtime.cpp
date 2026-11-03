#include <torch/extension.h>

#include <cstdint>
#include <vector>

int64_t dspark_instantiate_device_graph_cuda(int64_t graph_handle);
int64_t dspark_instantiate_composed_device_graph_cuda(
    const std::vector<int64_t>& graph_handles);
int64_t dspark_instantiate_composed_host_graph_cuda(
    const std::vector<int64_t>& graph_handles);
void dspark_launch_device_graph_cuda(int64_t executable_handle);
void dspark_destroy_device_graph_cuda(int64_t executable_handle);
void dspark_launch_tail_marker_cuda(
    const torch::Tensor& iteration,
    int64_t iterations);
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
    const torch::Tensor& graft_out_cache_loc);
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
    bool vectorized);
void dspark_publish_proposal_candidates_cuda(
    const torch::Tensor& output_ids,
    const torch::Tensor& candidates);
std::vector<int64_t> dspark_device_graph_node_types_cuda(int64_t graph_handle);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("instantiate", &dspark_instantiate_device_graph_cuda);
  module.def("instantiate_composed", &dspark_instantiate_composed_device_graph_cuda);
  module.def("instantiate_composed_host", &dspark_instantiate_composed_host_graph_cuda);
  module.def("launch", &dspark_launch_device_graph_cuda);
  module.def("destroy", &dspark_destroy_device_graph_cuda);
  module.def("tail_marker", &dspark_launch_tail_marker_cuda);
  module.def("full_loop_state_update", &dspark_launch_full_loop_state_update_cuda);
  module.def("prepare_proposal_megakernel", &dspark_prepare_proposal_megakernel_cuda);
  module.def("publish_proposal_candidates", &dspark_publish_proposal_candidates_cuda);
  module.def("node_types", &dspark_device_graph_node_types_cuda);
}
