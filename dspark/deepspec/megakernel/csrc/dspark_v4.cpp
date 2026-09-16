#include <torch/extension.h>

#include <string>
#include <vector>

void dspark_v4_scheduler_smoke_cuda(
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    int64_t proposal_epoch);

void dspark_v4_main_stage_cuda(
    const torch::Tensor& main_hidden,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& output,
    int64_t proposal_epoch);

void dspark_v4_front_stage_cuda(
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& main_output,
    const torch::Tensor& embedding_output,
    int64_t proposal_epoch);

void dspark_v4_front_kv_stage_cuda(
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& rope_cos_sin,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& main_output,
    const torch::Tensor& embedding_output,
    const torch::Tensor& kv_cache,
    const torch::Tensor& output_ids,
    const torch::Tensor& corrected_logits,
    const torch::Tensor& probabilities,
    const torch::Tensor& confidence_logits,
    const torch::Tensor& calibrated_confidences,
    const torch::Tensor& scheduled_prefix_lengths,
    const torch::Tensor& scheduler_read_mask,
    const torch::Tensor& scheduler_summary,
    const torch::Tensor& uniforms,
    const torch::Tensor& sts_temperatures,
    const torch::Tensor& steps_per_second,
    double sampling_temperature,
    bool calibration_enabled,
    bool prefix_enabled,
    int64_t start_pos,
    int64_t draft_layer_count,
    int64_t proposal_epoch);

void dspark_v4_front_kv_stage_masked_cuda(
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& rope_cos_sin,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& main_output,
    const torch::Tensor& embedding_output,
    const torch::Tensor& kv_cache,
    const torch::Tensor& output_ids,
    const torch::Tensor& corrected_logits,
    const torch::Tensor& probabilities,
    const torch::Tensor& confidence_logits,
    const torch::Tensor& calibrated_confidences,
    const torch::Tensor& scheduled_prefix_lengths,
    const torch::Tensor& scheduler_read_mask,
    const torch::Tensor& scheduler_summary,
    const torch::Tensor& uniforms,
    const torch::Tensor& sts_temperatures,
    const torch::Tensor& steps_per_second,
    double sampling_temperature,
    bool calibration_enabled,
    bool prefix_enabled,
    int64_t start_pos,
    int64_t draft_layer_count,
    int64_t proposal_epoch,
    int64_t execution_mask,
    int64_t draft_layer_mask);

#ifdef DSPARK_V4_STATIC_TP2_TAIL
pybind11::dict dspark_v4_tp2_tail_create_owner(
    int64_t rank,
    int64_t device,
    int64_t peer_device);
void dspark_v4_tp2_tail_open_peer(const pybind11::dict& peer_descriptor);
void dspark_v4_tp2_tail_bind_rank0_weights(
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets);
void dspark_v4_tp2_tail_enable_serving();
void dspark_v4_tp2_tail_bind_input_status(
    const torch::Tensor& input_status);
void dspark_v4_tp2_tail_inject_test_fault(
    const std::string& fault,
    int64_t proposal_epoch);
void dspark_v4_tp2_tail_set_payload_validation_epochs(int64_t epoch_count);
void dspark_v4_tp2_tail_arm_rank1(
    const torch::Tensor& lm_head_bf16,
    const torch::Tensor& markov_w1_bf16,
    const torch::Tensor& markov_w2_bf16,
    const torch::Tensor& anchor,
    int64_t proposal_epoch);
void dspark_v4_tp2_tail_publish_payload(
    const torch::Tensor& payload,
    int64_t proposal_epoch);
void dspark_v4_tp2_tail_consume_payload(
    const torch::Tensor& out,
    int64_t proposal_epoch);
void dspark_v4_tp2_tail_stage_and_publish_rank0(
    const torch::Tensor& output_ids,
    const torch::Tensor& calibrated_confidences,
    const torch::Tensor& anchor,
    const torch::Tensor& draft_block_ids,
    const c10::optional<torch::Tensor>& folded_tokens,
    const torch::Tensor& confidence_out,
    int64_t proposal_epoch,
    int64_t mask_token_id);
void dspark_v4_tp2_tail_consume_and_stage_rank1(
    const torch::Tensor& out,
    const torch::Tensor& draft_block_ids,
    const c10::optional<torch::Tensor>& folded_tokens,
    const torch::Tensor& confidence_out,
    int64_t proposal_epoch,
    int64_t mask_token_id);
pybind11::dict dspark_v4_tp2_tail_epoch_audit(int64_t proposal_epoch);
pybind11::dict dspark_v4_tp2_tail_allreduce_epoch(
    const torch::Tensor& local,
    torch::Tensor& combined,
    const c10::optional<torch::Tensor>& local_out,
    int64_t proposal_epoch,
    int64_t layers,
    int64_t parts,
    int64_t publish_ticket_skew,
    bool skip_publish);
void dspark_v4_tp2_tail_allreduce_reset_epochs();
pybind11::dict dspark_v4_tp2_tail_serving_audit();
void dspark_v4_tp2_tail_quiesce();
void dspark_v4_tp2_tail_close_peer();
void dspark_v4_tp2_tail_destroy_owner();
#endif

void dspark_v4_set_scheduler_policy(
    int64_t idle_sleep_ns,
    int64_t idle_sleep_long_ns,
    int64_t idle_sleep_after,
    int64_t flags,
    int64_t deep_period,
    int64_t express_workers,
    int64_t prefetch_period);
std::vector<int64_t> dspark_v4_get_scheduler_policy();
#if defined(DSPARK_V4_TP2_ABLATE)
void dspark_v4_set_tp2_ablation(int64_t rank, int64_t bands);
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
std::vector<torch::Tensor> dspark_v4_scheduler_probe_dump();
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "set_scheduler_policy",
      &dspark_v4_set_scheduler_policy,
      pybind11::arg("idle_sleep_ns"),
      pybind11::arg("idle_sleep_long_ns"),
      pybind11::arg("idle_sleep_after"),
      pybind11::arg("flags"),
      pybind11::arg("deep_period"),
      pybind11::arg("express_workers"),
      pybind11::arg("prefetch_period"),
      "Select the idle-discovery scheduling policy (no numerical effect)");
  module.def(
      "get_scheduler_policy",
      &dspark_v4_get_scheduler_policy,
      "Current idle-discovery scheduling policy");
#if defined(DSPARK_V4_TP2_ABLATE)
  module.def(
      "set_tp2_ablation",
      &dspark_v4_set_tp2_ablation,
      pybind11::arg("rank"),
      pybind11::arg("bands"),
      "TP2 band-ownership ablation: run only this rank's half of the "
      "selected bands (partial result by construction; timing only)");
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  module.def(
      "scheduler_probe_dump",
      &dspark_v4_scheduler_probe_dump,
      "Per-worker and per-phase scheduler attribution probes");
#endif
  module.def(
      "scheduler_smoke",
      &dspark_v4_scheduler_smoke_cuda,
      "One-launch DeepSeek-V4 persistent GPU scheduler smoke test");
  module.def(
      "main_stage",
      &dspark_v4_main_stage_cuda,
      "One-launch DeepSeek-V4 main FP8 projection debug stage");
  module.def(
      "front_stage",
      &dspark_v4_front_stage_cuda,
      "One-launch DeepSeek-V4 main projection and HC embedding debug stage");
  module.def(
      "front_kv_stage",
      &dspark_v4_front_kv_stage_cuda,
      "One-launch DeepSeek-V4 front plus three target-KV updates debug stage");
  module.def(
      "front_kv_stage_masked",
      &dspark_v4_front_kv_stage_masked_cuda,
      "Experimental phase-masked DeepSeek-V4 persistent stage");
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  module.def(
      "tp2_tail_create_owner",
      &dspark_v4_tp2_tail_create_owner,
      pybind11::arg("rank"),
      pybind11::arg("device"),
      pybind11::arg("peer_device"));
  module.def(
      "tp2_tail_open_peer",
      &dspark_v4_tp2_tail_open_peer,
      pybind11::arg("peer_descriptor"));
  module.def(
      "tp2_tail_bind_rank0_weights",
      &dspark_v4_tp2_tail_bind_rank0_weights,
      pybind11::arg("weight_arena"),
      pybind11::arg("weight_offsets"));
  module.def("tp2_tail_enable_serving", &dspark_v4_tp2_tail_enable_serving);
  module.def(
      "tp2_tail_bind_input_status",
      &dspark_v4_tp2_tail_bind_input_status,
      pybind11::arg("status"));
  module.def(
      "tp2_tail_inject_test_fault",
      &dspark_v4_tp2_tail_inject_test_fault,
      pybind11::arg("fault"),
      pybind11::arg("proposal_epoch"));
  module.def(
      "tp2_tail_set_payload_validation_epochs",
      &dspark_v4_tp2_tail_set_payload_validation_epochs,
      pybind11::arg("epoch_count"));
  module.def(
      "tp2_tail_arm_rank1",
      &dspark_v4_tp2_tail_arm_rank1,
      pybind11::arg("lm_head_bf16"),
      pybind11::arg("markov_w1_bf16"),
      pybind11::arg("markov_w2_bf16"),
      pybind11::arg("anchor"),
      pybind11::arg("proposal_epoch"));
  module.def(
      "tp2_tail_publish_payload",
      &dspark_v4_tp2_tail_publish_payload,
      pybind11::arg("payload"),
      pybind11::arg("proposal_epoch"));
  module.def(
      "tp2_tail_consume_payload",
      &dspark_v4_tp2_tail_consume_payload,
      pybind11::arg("out"),
      pybind11::arg("proposal_epoch"));
  module.def(
      "tp2_tail_stage_and_publish_rank0",
      &dspark_v4_tp2_tail_stage_and_publish_rank0,
      pybind11::arg("output_ids"),
      pybind11::arg("calibrated_confidences"),
      pybind11::arg("anchor"),
      pybind11::arg("draft_block_ids"),
      pybind11::arg("folded_tokens"),
      pybind11::arg("confidence_out"),
      pybind11::arg("proposal_epoch"),
      pybind11::arg("mask_token_id"));
  module.def(
      "tp2_tail_consume_and_stage_rank1",
      &dspark_v4_tp2_tail_consume_and_stage_rank1,
      pybind11::arg("out"),
      pybind11::arg("draft_block_ids"),
      pybind11::arg("folded_tokens"),
      pybind11::arg("confidence_out"),
      pybind11::arg("proposal_epoch"),
      pybind11::arg("mask_token_id"));
  module.def(
      "tp2_tail_epoch_audit",
      &dspark_v4_tp2_tail_epoch_audit,
      pybind11::arg("proposal_epoch"));
  module.def(
      "tp2_tail_allreduce_epoch",
      &dspark_v4_tp2_tail_allreduce_epoch,
      pybind11::arg("local"),
      pybind11::arg("combined"),
      pybind11::arg("local_out"),
      pybind11::arg("proposal_epoch"),
      pybind11::arg("layers"),
      pybind11::arg("parts"),
      pybind11::arg("publish_ticket_skew") = 0,
      pybind11::arg("skip_publish") = false);
  module.def(
      "tp2_tail_allreduce_reset_epochs",
      &dspark_v4_tp2_tail_allreduce_reset_epochs);
  module.def("tp2_tail_serving_audit", &dspark_v4_tp2_tail_serving_audit);
  module.def("tp2_tail_quiesce", &dspark_v4_tp2_tail_quiesce);
  module.def("tp2_tail_close_peer", &dspark_v4_tp2_tail_close_peer);
  module.def("tp2_tail_destroy_owner", &dspark_v4_tp2_tail_destroy_owner);
#endif
}
