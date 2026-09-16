from collections import Counter
from pathlib import Path

from deepspec.megakernel.full_loop_compiler import (
    FullLoopExecutor,
    GB300GraftLowering,
    GB300LoopLowering,
    GB300TargetHeadLowering,
    GB300TargetLowering,
    GraftExecutor,
    TargetLayerVariant,
    compile_full_loop_program,
    compile_gb300_graft_lowering,
    compile_gb300_loop_lowering,
    compile_gb300_target_head_lowering,
    compile_gb300_target_lowering,
)
from deepspec.megakernel.full_loop_schedule import build_full_loop_program


def test_compiler_preserves_semantics_but_emits_small_physical_program():
    program = compile_full_loop_program()
    assert program.semantic_phases == 898
    assert len(program.bands) == 51
    assert program.semantic_work_units == sum(band.work_units for band in program.bands)
    assert program.workers == 152
    assert program.launch_backend == "aot_composed_cuda_graph"
    assert all(
        band.dependency == (None if band.band_id == 0 else band.band_id - 1)
        for band in program.bands
    )
    assert [phase_id for band in program.bands for phase_id in band.semantic_phase_ids] == list(
        range(program.semantic_phases)
    )

    semantic = build_full_loop_program()
    assert all(
        band.work_units
        == sum(semantic[phase_id].work_units for phase_id in band.semantic_phase_ids)
        for band in program.bands
    )


def test_compiler_selects_specialized_graphs_instead_of_one_mega_symbol():
    program = compile_full_loop_program()
    executors = Counter(band.executor for band in program.bands)
    assert executors == {
        FullLoopExecutor.INPUT_CONTROLLER: 1,
        FullLoopExecutor.TARGET_BOOTSTRAP: 1,
        FullLoopExecutor.TARGET_LAYER_GRAPH: 43,
        FullLoopExecutor.TARGET_HEAD_GRAPH: 1,
        FullLoopExecutor.TARGET_INJECT_GRAPH: 1,
        FullLoopExecutor.LOOP_STATE_UPDATE: 1,
        FullLoopExecutor.GRAFT_INPUT_GRAPH: 1,
        FullLoopExecutor.GRAFT_GRAPH: 1,
        FullLoopExecutor.TAIL_RELAUNCH: 1,
    }


def test_target_variants_are_frozen_at_compile_time():
    target = compile_full_loop_program().target_bands
    assert [band.layer for band in target] == list(range(43))
    assert Counter(band.compress_ratio for band in target) == {0: 2, 4: 21, 128: 20}
    assert [band.layer for band in target if band.router == "hash"] == [0, 1, 2]
    assert [band.layer for band in target if band.router == "score"] == list(range(3, 43))
    assert [band.layer for band in target if band.captures_tap] == [40, 41, 42]
    assert Counter(band.target_variant for band in target) == {
        TargetLayerVariant.HASH_DENSE: 2,
        TargetLayerVariant.HASH_COMPRESSED_R4: 1,
        TargetLayerVariant.SCORE_COMPRESSED_R4: 20,
        TargetLayerVariant.SCORE_COMPRESSED_R128: 20,
    }
    assert all(band.target_lowering is not None for band in target)
    assert {band.target_lowering.router for band in target[:3]} == {"hash_fused"}
    assert {band.target_lowering.router for band in target[3:]} == {"sglang_fused_gate"}


def test_gb300_lowering_freezes_profiled_geometry_without_autotuning():
    lowering = compile_gb300_target_lowering(
        layer=42,
        variant=TargetLayerVariant.SCORE_COMPRESSED_R4,
    )
    assert lowering == GB300TargetLowering(variant=TargetLayerVariant.SCORE_COMPRESSED_R4)
    assert lowering.mhc_pre_splits == 64
    assert lowering.fused_mhc_splits == 8
    assert lowering.fused_mhc_tile_mix_outputs == 2
    assert lowering.moe_max_num_tokens == 8
    assert lowering.moe_enable_pdl is True
    assert lowering.overlap_query_kv
    assert lowering.q_lora_norm_quant == "sm103_fused"
    assert lowering.overlap_target_cache_pack
    assert lowering.overlap_routed_shared
    assert lowering.keep_wo_a_fp8
    assert lowering.fuse_moe_finalize
    assert lowering.moe_hc_post == "sm103_fused"
    assert lowering.compressor == "sglang_c4"
    assert lowering.compressed_cache_store == "sglang_fused"


def test_gb300_lowering_selects_compressor_from_semantic_variant():
    compiled = compile_full_loop_program()
    by_layer = {band.layer: band.target_lowering for band in compiled.target_bands}
    assert by_layer[0] is not None and by_layer[0].compressor == "none"
    assert by_layer[2] is not None and by_layer[2].compressor == "sglang_c4"
    assert by_layer[3] is not None and by_layer[3].compressor == "sglang_c128"
    assert by_layer[42] is not None and by_layer[42].compressor == "sglang_c4"


def test_gb300_graft_lowering_is_compiler_owned_and_autotune_free():
    lowering = compile_gb300_graft_lowering()
    assert lowering == GB300GraftLowering()
    assert lowering.greedy_argmax == "sglang_split_topk1"
    assert lowering.hc_head == "sm103_fused_hc_rmsnorm"
    assert lowering.lm_head_layout == "kn_contiguous"
    assert lowering.direct_proposal_store
    assert lowering.target_inject == "dspark_swa_kv"
    assert lowering.target_tap_reduce == "sm103_mean_concat"
    assert lowering.target_inject_layout == "sm103_commit_mask"
    assert lowering.input_prepare == "sm103_fill_bonus"
    assert lowering.routed_expert == "flashinfer_mxfp4"
    assert lowering.autotune is False
    graft = next(
        band
        for band in compile_full_loop_program().bands
        if band.executor is FullLoopExecutor.GRAFT_GRAPH
    )
    assert graft.graft_lowering == lowering


def test_proposal_megakernel_is_an_isolated_full_loop_executor_choice():
    lowering = compile_gb300_graft_lowering(GraftExecutor.PROPOSAL_MEGAKERNEL)
    assert lowering.executor == "proposal_megakernel"
    assert lowering.attention == "dspark_persistent"
    assert lowering.routed_expert == "dspark_persistent_fp4"
    assert lowering.greedy_argmax == "dspark_greedy_tail"
    assert lowering.target_inject == "dspark_swa_kv"
    assert lowering.autotune is False
    program = compile_full_loop_program(graft_executor=GraftExecutor.PROPOSAL_MEGAKERNEL)
    graft_lowerings = {
        band.graft_lowering for band in program.bands if band.graft_lowering is not None
    }
    assert graft_lowerings == {lowering}


def test_gb300_target_head_lowering_is_compiler_owned_and_autotune_free():
    lowering = compile_gb300_target_head_lowering()
    assert lowering == GB300TargetHeadLowering()
    assert lowering.greedy_argmax == "sglang_split_topk1"
    assert lowering.hc_head == "sm103_fused_hc_rmsnorm"
    assert lowering.norm == "sm103_fused_hc_rmsnorm"
    assert lowering.lm_head == "torch_bf16_mm_kn"
    assert lowering.lm_head_layout == "kn_contiguous"
    assert lowering.accept_commit == "sm103_fused"
    assert lowering.autotune is False
    head = next(
        band
        for band in compile_full_loop_program().bands
        if band.executor is FullLoopExecutor.TARGET_HEAD_GRAPH
    )
    assert head.target_head_lowering == lowering


def test_only_commit_band_owns_the_logical_loop_backedge():
    program = compile_full_loop_program()
    tail = [band for band in program.bands if band.executor is FullLoopExecutor.TAIL_RELAUNCH]
    assert len(tail) == 1
    assert tail[0].band_id == len(program.bands) - 1
    assert tail[0].name == "loop.commit_and_continue"
    assert tail[0].loop_lowering == compile_gb300_loop_lowering()
    assert tail[0].loop_lowering == GB300LoopLowering()
    assert tail[0].loop_lowering.metadata_scope == "shared_across_target_layers"
    assert tail[0].loop_lowering.relaunch == "aot_parent_unroll"
    assert tail[0].loop_lowering.autotune is False

    state = [band for band in program.bands if band.executor is FullLoopExecutor.LOOP_STATE_UPDATE]
    assert len(state) == 1
    assert state[0].name == "loop.advance_accepted_state"
    assert state[0].band_id < tail[0].band_id
    assert state[0].loop_lowering == tail[0].loop_lowering


def test_compiler_orders_inject_state_update_prepare_and_graft():
    program = compile_full_loop_program()
    executors = [band.executor for band in program.bands]
    expected = (
        FullLoopExecutor.TARGET_INJECT_GRAPH,
        FullLoopExecutor.LOOP_STATE_UPDATE,
        FullLoopExecutor.GRAFT_INPUT_GRAPH,
        FullLoopExecutor.GRAFT_GRAPH,
        FullLoopExecutor.TAIL_RELAUNCH,
    )
    start = executors.index(FullLoopExecutor.TARGET_INJECT_GRAPH)
    assert tuple(executors[start:]) == expected


def test_cuda13_runtime_supports_composed_graph_and_tail_probe_programs():
    root = Path(__file__).resolve().parents[1]
    source = (root / "deepspec/megakernel/csrc/dspark_device_graph_runtime_kernel.cu").read_text()
    assert "cudaGraphInstantiateFlagDeviceLaunch" in source
    assert "cudaGraphInstantiateFlagUpload" in source
    assert "cudaGetCurrentGraphExec()" in source
    assert "cudaStreamGraphTailLaunch" in source
    # One host launch plus the retained generic tail-marker path and the full
    # loop controller's early tail publication.
    assert source.count("cudaGraphLaunch(") == 3
