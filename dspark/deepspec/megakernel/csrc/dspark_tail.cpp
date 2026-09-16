#include <torch/extension.h>

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
    const torch::Tensor& trace);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "tail",
      &dspark_tail_cuda,
      "Single-launch DSpark Markov, softmax, sampling, and confidence tail");
}
