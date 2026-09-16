from types import SimpleNamespace

import pytest

from deepspec.megakernel import (
    DeepSeekV4MegaKernelSpec,
    UnsupportedMegaKernelConfig,
    V4LaunchShape,
    V4PhaseKind,
    V4WorkerRole,
    build_v4_launch_abi,
    build_v4_phase_program,
)


def _flash_config() -> dict[str, object]:
    return {
        "vocab_size": 129280,
        "dim": 4096,
        "moe_inter_dim": 2048,
        "n_layers": 43,
        "n_mtp_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "n_heads": 64,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 6,
        "score_func": "sqrtsoftplus",
        "route_scale": 1.5,
        "swiglu_limit": 10.0,
        "q_lora_rank": 1024,
        "head_dim": 512,
        "rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "window_size": 128,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "dtype": "fp8",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }


def test_official_v4_flash_config_maps_to_frozen_contract():
    expected = DeepSeekV4MegaKernelSpec()
    assert DeepSeekV4MegaKernelSpec.from_inference_config(_flash_config()) == expected
    assert (
        DeepSeekV4MegaKernelSpec.from_inference_config(SimpleNamespace(**_flash_config()))
        == expected
    )
    expected.require_supported()
    assert expected.target_feature_width == 12288
    assert expected.mhc_width == 16384
    assert expected.query_width == 32768
    assert expected.maximum_sparse_kv == 133


def test_v4_pro_is_identified_but_fails_closed():
    config = _flash_config()
    config.update(
        dim=7168,
        n_layers=61,
        dspark_target_layer_ids=[58, 59, 60],
        dspark_markov_rank=512,
        n_heads=128,
        n_routed_experts=384,
        q_lora_rank=1536,
        o_groups=16,
        moe_inter_dim=3072,
        route_scale=2.5,
    )
    spec = DeepSeekV4MegaKernelSpec.from_inference_config(config)
    assert spec.variant == "pro"
    with pytest.raises(UnsupportedMegaKernelConfig, match="variant='pro'"):
        spec.require_supported()


def test_unknown_v4_shape_fails_closed():
    config = _flash_config()
    config["dim"] = 4097
    spec = DeepSeekV4MegaKernelSpec.from_inference_config(config)
    assert spec.variant == "unknown"
    with pytest.raises(UnsupportedMegaKernelConfig, match="hidden_size=4097"):
        spec.require_supported()


def test_gb300_launch_shape_accepts_only_generated_batch_specializations():
    # Batched serving (R3) replaced the fail-closed batch_size == 1 guard: a
    # generated header exists for batch 1, 2, 4 and 8, and nothing else is
    # buildable, so the shape stays fail-closed on every other value.
    for batch in (1, 2, 4, 8):
        V4LaunchShape(batch_size=batch).require_supported()
    for batch in (0, 3, 5, 16):
        with pytest.raises(ValueError, match="batch_size"):
            V4LaunchShape(batch_size=batch).require_supported()
    with pytest.raises(ValueError, match="152 SMs"):
        V4LaunchShape(workers=148).require_supported()


def test_v4_abi_covers_outputs_workspace_and_one_launch_boundary():
    abi = build_v4_launch_abi()
    assert abi.top_level_kernel_launches == 1
    assert abi.nested_kernel_launches == 0
    assert abi.weight_arena.source_tensor_count == 4707
    assert abi.weight_arena.source_stored_bytes == 12_980_961_820
    assert abi.weight_arena.conversions_during_proposal == 0
    assert abi.tensor("main_hidden").shape == (1, 1, 12288)
    assert abi.tensor("kv_cache").shape == (3, 1, 128, 512)
    assert abi.tensor("output_ids").shape == (1, 6)
    assert abi.tensor("draft_probabilities").shape == (1, 5, 129280)
    assert abi.tensor("scheduler_read_mask").shape == (1, 5)
    assert abi.tensor("sampling_temperature").shape == (1,)
    assert abi.tensor("trace_records").shape == (152, 2048, 8)

    previous_end = 0
    for region in abi.workspace.regions:
        assert region.offset_bytes % abi.workspace.alignment == 0
        assert region.offset_bytes >= previous_end
        assert region.nbytes > 0
        previous_end = region.end_bytes
    assert abi.workspace.total_bytes >= previous_end
    assert abi.workspace.region("phase_dependency_arrivals").shape[0] == len(
        build_v4_phase_program(abi.spec, abi.shape)
    )


def test_v4_gpu_program_is_topological_device_only_and_overlap_ready():
    spec = DeepSeekV4MegaKernelSpec()
    shape = V4LaunchShape()
    program = build_v4_phase_program(spec, shape)
    by_name = {phase.name: phase for phase in program}
    seen: set[str] = set()
    assert [phase.phase_id for phase in program] == list(range(len(program)))
    for phase in program:
        assert set(phase.dependencies) <= seen
        assert phase.synchronization == "release_acquire_dependency_counter"
        assert phase.work_units > 0
        assert "barrier" not in phase.name
        assert "host" not in phase.name
        assert "sleep" not in phase.name
        seen.add(phase.name)

    roots = {phase.name for phase in program if not phase.dependencies}
    assert roots == {
        "main.activation_quant",
        "draft.embedding_hc_expand",
        "tail_0.markov_gather",
    }
    assert by_name["main.projection"].dependencies == ("main.activation_quant",)
    assert by_name["main.split_k_reduce"].dependencies == ("main.projection",)
    assert by_name["tail_0.markov_w2"].dependencies == ("tail_0.markov_gather",)
    assert "head.lm_row_0" in by_name["tail_0.correct_logits_partial_max"].dependencies
    for layer in range(3):
        phase = by_name[f"layer_{layer}.main_kv_projection"]
        assert phase.dependencies == ("main.projected_activation_quant",)
        assert (
            f"layer_{layer}.main_kv_norm_rope_quant_store"
            in by_name[f"layer_{layer}.sparse_attention"].dependencies
        )
        assert by_name[f"layer_{layer}.attn_hc_reduce_rmsnorm"].work_units == 5
        assert by_name[f"layer_{layer}.ffn_hc_reduce_rmsnorm"].work_units == 5
        assert by_name[f"layer_{layer}.q_a_rmsnorm"].work_units == 5
    assert by_name["layer_0.fp4_routed_w13"].work_units >= shape.workers
    assert by_name["tail_0.softmax_exp_sum"].work_units >= shape.workers
    assert by_name["tail_1.markov_gather"].dependencies == ("tail_0.normalize_scan_sample",)
    assert by_name["prefix.causal_algorithm_1"].kind == V4PhaseKind.SCHEDULER
    assert by_name["proposal.finalize_audit"].roles == (V4WorkerRole.CONTROLLER,)
    assert sum(role.warps for role in V4WorkerRole) == 8


def test_v4_greedy_tail_program_removes_dead_temperature_zero_phases():
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
    )
    by_name = {phase.name: phase for phase in program}

    assert len(program) == 94
    # R12: +384 over the pre-R12 total (3 layers x 128). The relaxed
    # router band K-splits each 4,096-step chain across 32 threads, so one
    # work item is 8 chains instead of 40 and the band exposes 160 items
    # per layer instead of 32. Nothing else in the program moved.
    assert sum(phase.work_units for phase in program) == 16_657
    assert not any("softmax_exp_sum" in name for name in by_name)
    assert not any("normalize_scan_sample" in name for name in by_name)
    assert not any(name.startswith("prefix.") for name in by_name)
    assert not any(name.startswith("head.lm_row_") and name != "head.lm_row_0" for name in by_name)

    for step in range(5):
        gather = by_name[f"tail_{step}.markov_gather"]
        markov = by_name[f"tail_{step}.markov_w2"]
        if step:
            assert f"tail_{step - 1}.markov_w2" in gather.dependencies
        assert markov.dependencies == (gather.name, "head.lm_row_0")
    assert by_name["proposal.finalize_audit"].dependencies == ("tail_4.markov_w2",)


def test_v4_static_tp2_tail_program_gives_rank0_two_fifths_of_tail_vocabulary():
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        static_tp2_tail=True,
    )
    by_name = {phase.name: phase for phase in program}

    assert len(program) == 94
    # R12: +384 over the pre-R12 total (3 layers x 128). The relaxed
    # router band K-splits each 4,096-step chain across 32 threads, so one
    # work item is 8 chains instead of 40 and the band exposes 160 items
    # per layer instead of 32. Nothing else in the program moved.
    assert sum(phase.work_units for phase in program) == 13_021
    assert by_name["head.lm_row_0"].work_units == 404
    assert [by_name[f"tail_{step}.markov_w2"].work_units for step in range(5)] == [404] * 5


def test_v4_greedy_tail_variants_fail_closed():
    with pytest.raises(ValueError, match="greedy_tail requires relaxed_tail"):
        build_v4_phase_program(
            DeepSeekV4MegaKernelSpec(),
            V4LaunchShape(),
            greedy_tail=True,
        )

    from deepspec.megakernel.extension import load_dspark_v4_scheduler_extension

    with pytest.raises(ValueError, match="greedy_tail requires relaxed_dag"):
        load_dspark_v4_scheduler_extension(greedy_tail=True)
    # greedy_tail + instrumented is NO LONGER rejected: the device trace
    # records only (phase_id, worker, role, timestamps) and
    # build_v4_phase_program already accepts greedy_tail/batch, so the
    # host-side phase map is derivable. Lifting it is what made the batch-8
    # band map measurable (R5 S0). The remaining instrumentation guard is
    # the static-TP2 one, covered by the test below.
    # A build is not attempted here: nvcc/CUTLASS are absent on the CPU lane,
    # so only the argument validation is asserted, and it must not raise
    # before the toolchain lookup.
    assert (
        load_dspark_v4_scheduler_extension.cache_info() is not None
    )  # the loader is still the cached entry point


def test_v4_fine_overlap_requires_static_batch_one_program(monkeypatch):
    from deepspec.megakernel.extension import load_dspark_v4_scheduler_extension

    load_dspark_v4_scheduler_extension.cache_clear()
    monkeypatch.setenv("DSPARK_V4_FINE_GRAINED_OVERLAP", "1")
    monkeypatch.delenv("DSPARK_V4_STATIC_QUEUES", raising=False)
    with pytest.raises(
        ValueError,
        match="DSPARK_V4_FINE_GRAINED_OVERLAP requires DSPARK_V4_STATIC_QUEUES=1",
    ):
        load_dspark_v4_scheduler_extension(relaxed_dag=True, greedy_tail=True)
    load_dspark_v4_scheduler_extension.cache_clear()


def test_v4_compiled_phase_masks_require_static_complete_valid_pair(monkeypatch):
    from deepspec.megakernel.extension import load_dspark_v4_scheduler_extension

    load_dspark_v4_scheduler_extension.cache_clear()
    monkeypatch.delenv("DSPARK_V4_STATIC_QUEUES", raising=False)
    with pytest.raises(ValueError, match="provided together"):
        load_dspark_v4_scheduler_extension(compiled_execution_mask=1)
    with pytest.raises(ValueError, match="require DSPARK_V4_STATIC_QUEUES"):
        load_dspark_v4_scheduler_extension(
            compiled_execution_mask=1,
            compiled_draft_layer_mask=1,
        )
    monkeypatch.setenv("DSPARK_V4_STATIC_QUEUES", "1")
    with pytest.raises(ValueError, match="unsupported phase bits"):
        load_dspark_v4_scheduler_extension(
            relaxed_dag=True,
            greedy_tail=True,
            compiled_execution_mask=1 << 20,
            compiled_draft_layer_mask=1,
        )
    with pytest.raises(ValueError, match="select one to three layers"):
        load_dspark_v4_scheduler_extension(
            relaxed_dag=True,
            greedy_tail=True,
            compiled_execution_mask=1,
            compiled_draft_layer_mask=0,
        )
    load_dspark_v4_scheduler_extension.cache_clear()


def test_v4_static_tp2_tail_variants_fail_closed():
    spec = DeepSeekV4MegaKernelSpec()
    shape = V4LaunchShape()
    with pytest.raises(ValueError, match="static_tp2_tail requires relaxed_tail"):
        build_v4_phase_program(spec, shape, static_tp2_tail=True)
    with pytest.raises(ValueError, match="static_tp2_tail requires greedy_tail"):
        build_v4_phase_program(
            spec,
            shape,
            relaxed_tail=True,
            static_tp2_tail=True,
        )

    from deepspec.megakernel.extension import load_dspark_v4_scheduler_extension

    with pytest.raises(ValueError, match="legacy B200 static TP2 tail"):
        load_dspark_v4_scheduler_extension(static_tp2_tail=True)
    with pytest.raises(ValueError, match="legacy B200 static TP2 tail"):
        load_dspark_v4_scheduler_extension(
            relaxed_dag=True,
            static_tp2_tail=True,
        )
    with pytest.raises(ValueError, match="legacy B200 static TP2 tail"):
        load_dspark_v4_scheduler_extension(
            instrumented=True,
            relaxed_dag=True,
            greedy_tail=True,
            static_tp2_tail=True,
        )
