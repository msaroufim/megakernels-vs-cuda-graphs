#include <torch/extension.h>

std::vector<torch::Tensor> dspark_full_cuda(
    const torch::Tensor& draft_input_ids,
    const torch::Tensor& first_prev_token,
    const torch::Tensor& target_hidden_states,
    const torch::Tensor& position_ids,
    const torch::Tensor& uniforms,
    double temperature,
    int64_t past_length,
    double rms_epsilon,
    double rope_theta,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    const std::vector<torch::Tensor>& top_weights,
    const std::vector<torch::Tensor>& layer_weights,
    const std::vector<torch::Tensor>& workspace);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "full",
      &dspark_full_cuda,
      "Single-launch complete DSpark Qwen proposal");
}
