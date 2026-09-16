#include <torch/extension.h>

void gather_target_kv_cuda(
    const torch::Tensor& swa0,
    const torch::Tensor& swa1,
    const torch::Tensor& swa2,
    const torch::Tensor& req_to_token,
    const torch::Tensor& full_to_swa,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& seq_lens,
    const c10::optional<torch::Tensor>& expected_pos,
    const torch::Tensor& out,
    const torch::Tensor& status,
    bool allow_zero_locations);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "gather_target_kv",
      &gather_target_kv_cuda,
      pybind11::arg("swa0"),
      pybind11::arg("swa1"),
      pybind11::arg("swa2"),
      pybind11::arg("req_to_token"),
      pybind11::arg("full_to_swa"),
      pybind11::arg("req_pool_indices"),
      pybind11::arg("seq_lens"),
      pybind11::arg("expected_pos"),
      pybind11::arg("out"),
      pybind11::arg("status"),
      pybind11::arg("allow_zero_locations") = false,
      "Gather and dequantize the authoritative DeepSeek-V4 SWA KV ring");
}
