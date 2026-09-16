"""Prepare the retained prompt and real target taps outside proposal timing."""

from types import SimpleNamespace


def prepare(torch, args, arms, fixture, candidates, initial_candidates):
    """Seed shared KV and compose input preparation with each unchanged captured arm."""
    from deepspec.megakernel.device_graph_runtime import DeviceGraphExecutable
    from deepspec.megakernel.full_loop_graft import capture_target_inject_band
    from deepspec.megakernel.full_loop_moe_compiler import prepare_graft_inputs_cuda
    from deepspec.megakernel.prefill_snapshot import seed_graft_pool

    graft = arms["original"]
    seed_graft_pool(fixture, graft)
    prefix = int(fixture["seq_lens"][0])
    prefix_len = torch.tensor([prefix], dtype=torch.int64, device="cuda")
    commit_len = torch.ones(1, dtype=torch.int32, device="cuda")
    bonus = fixture["bonus_tokens"].to(device="cuda", dtype=torch.int64)
    saved = torch.load(args.target_taps, map_location="cpu", weights_only=True)
    if (
        saved["schema"] != "deepspec.compiled_target_taps.v1"
        or saved["checkpoint_revision"] != args.snapshot.name
        or saved["prefix_len"] != prefix
        or saved["target_layers"] != list(range(43))
        or not torch.equal(saved["candidate_ids"], initial_candidates.cpu())
    ):
        raise ValueError("Real target taps do not match the retained prompt/checkpoint")
    taps = tuple(saved["taps"][layer].to("cuda") for layer in (40, 41, 42))
    if any(t.shape != (6, 4, 4096) or t.dtype != torch.bfloat16 for t in taps):
        raise ValueError("Target taps must be three BF16 [6, 4, 4096] tensors")
    inject = capture_target_inject_band(graft, taps, commit_len=commit_len, prefix_len=prefix_len)
    inject.graph.replay()
    torch.cuda.synchronize()
    inputs, executables = {}, {}
    for name, arm in arms.items():
        arm.seq_lens.fill_(prefix + 1)
        arm.positions.copy_(torch.arange(prefix + 1, prefix + 6, device="cuda"))
        arm.out_cache_loc.copy_(arm.positions.remainder(128))
        if hasattr(arm.runner.buffers, "seq_lens_cpu"):
            arm.runner.buffers.seq_lens_cpu.fill_(prefix + 1)

        def operation(arm=arm):
            """Include the original device-side input preparation in each proposal."""
            prepare_graft_inputs_cuda(bonus, arm.input_ids, candidates, mask_token_id=128799)

        operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            operation()
        inputs[name] = graph
        executables[name] = DeviceGraphExecutable.compose((graph, arm.graph), device_launch=False)
    torch.cuda.synchronize()
    return SimpleNamespace(
        inputs=inputs,
        executables=executables,
        keepalive=(inject, taps, prefix_len, commit_len, bonus, candidates),
        metadata={
            "prefix_len": prefix,
            "proposal_seq_len": prefix + 1,
            "target_layers": [40, 41, 42],
            "initial_candidate_ids": initial_candidates.cpu().tolist(),
            "bonus_ids": bonus.cpu().tolist(),
            "input_preparation_timed": True,
            "target_injection_timed": False,
            "exec_flags": 0,
        },
    )
