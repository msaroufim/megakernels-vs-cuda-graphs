#include <torch/extension.h>

void dspark_route_pack_cuda(
    const torch::Tensor& ids,
    const torch::Tensor& weights,
    const torch::Tensor& output);

void dspark_target_tap_mean_concat_cuda(
    const torch::Tensor& tap0,
    const torch::Tensor& tap1,
    const torch::Tensor& tap2,
    const torch::Tensor& output);

void dspark_embedding_hc_expand_cuda(
    const torch::Tensor& token_ids,
    const torch::Tensor& embedding,
    const torch::Tensor& output);

void dspark_prepare_graft_inputs_cuda(
    const torch::Tensor& bonus,
    const torch::Tensor& input_ids,
    const torch::Tensor& candidates,
    long mask_token_id);

void dspark_prepare_target_inject_layout_cuda(
    const torch::Tensor& commit_len,
    const torch::Tensor& prefix_len,
    const torch::Tensor& swa_loc,
    const torch::Tensor& positions);

void dspark_target_hc_head_rmsnorm_cuda(
    const torch::Tensor& streams,
    const torch::Tensor& hc_fn,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    const torch::Tensor& norm_weight,
    const torch::Tensor& output,
    double norm_eps,
    double hc_eps);

void dspark_q_lora_rmsnorm_quant_cuda(
    const torch::Tensor& input,
    const torch::Tensor& norm_weight,
    const torch::Tensor& quantized,
    const torch::Tensor& scales,
    double norm_eps);

void dspark_fused_moe_hc_post_cuda(
    const torch::Tensor& routed,
    const torch::Tensor& shared,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output);

void dspark_expert_hc_post_cuda(
    const torch::Tensor& expert_output,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output);

void dspark_fused_moe_finalize_hc_post_cuda(
    const torch::Tensor& gemm2,
    const torch::Tensor& expert_weights,
    const torch::Tensor& expanded_to_permuted,
    const torch::Tensor& shared,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output);

void dspark_markov_add_argmax_cuda(
    const torch::Tensor& base_logits,
    const torch::Tensor& bias,
    const torch::Tensor& draft_tokens,
    const torch::Tensor& argmax_slot,
    long draft_token_column);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "pack",
      &dspark_route_pack_cuda,
      "Pack top-k ids and BF16 route weights for FlashInfer");
  module.def(
      "target_tap_mean_concat",
      &dspark_target_tap_mean_concat_cuda,
      "Mean three target HC taps and concatenate them");
  module.def(
      "embedding_hc_expand",
      &dspark_embedding_hc_expand_cuda,
      "Gather six checkpoint embedding rows and expand four HC streams");
  module.def(
      "prepare_graft_inputs",
      &dspark_prepare_graft_inputs_cuda,
      "Prepare graft masks and accepted bonus token");
  module.def(
      "prepare_target_inject_layout",
      &dspark_prepare_target_inject_layout_cuda,
      "Mask uncommitted target rows and assign their decode positions");
  module.def(
      "target_hc_head_rmsnorm",
      &dspark_target_hc_head_rmsnorm_cuda,
      "Fuse the fixed DSpark HC head and final RMSNorm");
  module.def(
      "q_lora_rmsnorm_quant",
      &dspark_q_lora_rmsnorm_quant_cuda,
      "Fuse the fixed target Q-LoRA RMSNorm and UE8M0 quantization");
  module.def(
      "moe_hc_post",
      &dspark_fused_moe_hc_post_cuda,
      "Fuse target routed/shared combine into MHC post");
  module.def(
      "expert_hc_post",
      &dspark_expert_hc_post_cuda,
      "Fuse one complete expert output into MHC post");
  module.def(
      "moe_finalize_hc_post",
      &dspark_fused_moe_finalize_hc_post_cuda,
      "Fuse FlashInfer MoE finalize, shared combine, and MHC post");
  module.def(
      "markov_add_argmax",
      &dspark_markov_add_argmax_cuda,
      "Fuse DSpark Markov base-logit addition with greedy argmax");
}
