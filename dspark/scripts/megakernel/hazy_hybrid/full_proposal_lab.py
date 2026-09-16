"""Compare the retained full-body scheduler and one-kernel proposal with original SGLang."""

import argparse
import json
import os
import statistics
import struct
import sys
import traceback
from pathlib import Path

from benchmark_utils import (
    capture_factory,
    emit,
    graph_record,
    load,
    save_tensors,
    sha,
    tensor_fields,
)
from fixture import prepare
from full_proposal_build import BASE_BINARY_SHA, BASE_RECORD_SHA


def error(torch, reference, value, *, allow_dtype_difference=False):
    """Report numerical differences without concealing a changed shape or nonfinite output."""
    if reference.shape != value.shape or (
        reference.dtype != value.dtype and not allow_dtype_difference
    ):
        raise ValueError("Output layout changed")
    left, right = reference.float(), value.float()
    difference = right - left
    return {
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(value.dtype),
        "bitwise_equal": bool(
            torch.equal(
                reference.contiguous().reshape(-1).view(torch.uint8),
                value.contiguous().reshape(-1).view(torch.uint8),
            )
        ),
        "different_elements": int((reference != value).sum().item()),
        "max_abs": float(difference.abs().max().item()),
        "rms": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(difference.norm().item() / max(left.norm().item(), 1e-30)),
        "finite": bool(torch.isfinite(right).all().item()),
    }


def capture_target_work(torch, graft, target_hidden, commit_len, new_seq_len):
    """Refresh one committed target row, matching the integrated proposal's input contract."""
    selected = torch.empty((1, 12288), dtype=torch.bfloat16, device=target_hidden.device)
    index = torch.empty_like(commit_len)
    position = torch.empty_like(new_seq_len)
    slot = torch.empty_like(commit_len)

    def operation():
        torch.sub(commit_len, 1, out=index)
        index.clamp_(0, 5)
        torch.index_select(target_hidden, 0, index, out=selected)
        torch.sub(new_seq_len, 1, out=position)
        torch.remainder(position, 128, out=slot)
        graft.bundle.draft_model.write_target_hidden_kv(
            main_hidden=selected,
            swa_loc=slot,
            positions=position,
            pool=graft.bundle.draft_model_runner.token_to_kv_pool,
        )

    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        operation()
    return graph, (selected, index, position, slot)


def run(args):
    """Validate exact IO integration before timing any complete-proposal comparison."""
    if not args.disjoint_draft_kv_fixture:
        raise ValueError("The retained comparison requires --disjoint-draft-kv-fixture")
    out, root = args.output, args.artifact_root
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "complete": False,
        "execution_complete": False,
        "numerically_qualified": None,
        "diagnostic_only": False,
        "status": "loading",
        "scope": "Complete fixed-state greedy proposal, including preparation and publication",
        "candidate_uses_one_kernel": False,
        "gpu_side_scheduler": True,
        "candidate_pdl": False,
        "normalized_sampling_qualified": False,
        "evolving_kv_qualified": False,
        "cluster_priority": "p50",
        "comparisons": [],
        "correctness": {},
        "equal_work": args.equal_work,
    }

    def save():
        """Preserve partial evidence before subsequent builds, captures or validation."""
        emit(out / "result.json", state)

    save()
    try:
        pins = json.loads(Path(__file__).with_name("pins.json").read_text())
        for key in ("prefill_snapshot", "target_taps"):
            if sha(getattr(args, key)) != pins["fixtures"][key]:
                raise ValueError(f"Changed real fixture: {key}")
        cap = load(
            "full_io_original_capture",
            root / "hazy-original-post-ffn-capture-draft-0911/capture.py",
        )
        source, _ = cap.admit(args.snapshot)
        sys.path.insert(0, str(root / source["host_directory"]))
        import torch

        from deepspec.megakernel.device_graph_runtime import DeviceGraphExecutable
        from deepspec.megakernel.extension import view_v4_workspace_region
        from deepspec.megakernel.full_loop_compiler import (
            GraftExecutor,
            compile_gb300_graft_lowering,
        )
        from deepspec.megakernel.full_loop_graft import capture_sglang_graft_band
        from deepspec.megakernel.prefill_snapshot import load_prefill_snapshot, verify_ids
        from scripts.megakernel.precompiled_experiments import load_plan, packing_environment

        torch.set_grad_enabled(False)
        gpu = torch.cuda.get_device_properties(0)
        if torch.cuda.device_count() != 1 or (gpu.major, gpu.minor, gpu.multi_processor_count) != (
            10,
            3,
            152,
        ):
            raise ValueError("Expected one 152-SM GB300")
        state["gpu"] = {"name": gpu.name, "uuid": str(gpu.uuid), "sms": gpu.multi_processor_count}
        state["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda}
        old_record = root.parent / "extensions-resumed-completion-red-0908/build-record.json"
        if sha(old_record) != BASE_RECORD_SHA:
            raise ValueError("Retained full-body build record changed")
        if json.loads(old_record.read_text())["binary_sha256"] != BASE_BINARY_SHA:
            raise ValueError("Retained full-body image changed")
        plan_path = out / "plan.json"
        image_plans = [
            ("proposal_megakernel", old_record),
            ("integrated", args.build_record),
        ]
        emit(
            plan_path,
            {
                "schema": "dspark.precompiled_plan.v1",
                "arms": [
                    {
                        "label": name,
                        "build_record": str(path),
                        "sha256": sha(path),
                    }
                    for name, path in image_plans
                ],
            },
        )
        plan = load_plan(plan_path)
        state["images"] = plan.identity()
        overlay = args.overlay
        import deepspec.megakernel as kernel_package

        kernel_package.proposal_io = load(
            "deepspec.megakernel.proposal_io", overlay / "proposal_io.py"
        )
        capture_module = load("full_io_capture", overlay / "full_loop_proposal.py")
        fixture = load_prefill_snapshot(args.prefill_snapshot)
        initial_candidates = verify_ids(fixture)
        candidates = initial_candidates.clone()
        if args.equal_work:
            from equal_work_head import install_head_export

            import deepspec.megakernel.full_loop_graft as graft_module

            head_hidden = torch.empty((5, 4096), dtype=torch.bfloat16, device="cuda")
            head_receipt = install_head_export(torch, graft_module, head_hidden)
            state["reference_head_export"] = head_receipt
        graft, _ = capture_factory(
            capture_sglang_graft_band, args.snapshot, args.dist_port, candidates[1:]
        )
        prepared = prepare(
            torch, args, {"original": graft}, fixture, candidates, initial_candidates
        )
        inject, _, _, commit_len, bonus, _ = prepared.keepalive
        new_seq_len = torch.tensor(
            [prepared.metadata["proposal_seq_len"]], dtype=torch.int64, device="cuda"
        )
        if args.equal_work:
            from equal_work import capture_confidence
            from full_proposal_disjoint_fixture import digest, prefix_bytes

            pools = graft.bundle.draft_model_runner.token_to_kv_pool.swa_kv_pool.kv_buffer
            old_prefix = tuple(prefix_bytes(value) for value in pools)
            target_graph, target_owners = capture_target_work(
                torch, graft, inject.target_hidden, commit_len, new_seq_len
            )
            confidence = capture_confidence(
                torch, graft, candidates, head_hidden=head_hidden, snapshot=args.snapshot
            )
            # The one-row target refresh establishes this run's common cache fixture.
            # The disjoint verifier snapshots it below, after all capture warmups.
            target_graph.replay()
            torch.cuda.synchronize()
            new_prefix = tuple(prefix_bytes(value) for value in pools)

            def untouched(data):
                """Exclude only slot2's 576 data bytes and eight scale bytes."""
                return data[:1152] + data[1728:73744] + data[73752:]

            if any(
                untouched(a) != untouched(b) for a, b in zip(old_prefix, new_prefix, strict=True)
            ):
                raise ValueError("One-row target refresh modified another live prefix slot")
            state["equal_work_contract"] = {
                "input": "BF16[6,12288] taps; GPU commit length, bonus, sequence length",
                "target_rows_projected": 1,
                "target_tap_mean_concat_timed": False,
                "target_work": (
                    "Select committed row; main projection/RMS; three target KV projections, "
                    "normalization, RoPE and cache writes"
                ),
                "confidence": confidence.record(),
                "prefix_refresh": [
                    {
                        "before_sha256": digest(a),
                        "after_sha256": digest(b),
                        "other_127_slots_unchanged": True,
                    }
                    for a, b in zip(old_prefix, new_prefix, strict=True)
                ],
                "scope": (
                    "Fixed-state greedy proposal; verifier, normalized sampling "
                    "and evolving cache excluded"
                ),
            }
        disjoint_fixture = None
        if getattr(args, "disjoint_draft_kv_fixture", False):
            from full_proposal_disjoint_fixture import apply

            disjoint_fixture = apply(torch, graft, prepared)
            state["disjoint_draft_kv_fixture"] = disjoint_fixture.record()
            state["fixture_integrity"] = {}
            emit(out / "disjoint-draft-kv-fixture.json", disjoint_fixture.record())
        fields = tensor_fields(graft.runner.buffers, torch)
        initial_fields = {name: value.clone() for name, value in fields.items()}
        kv = tuple(graft.bundle.draft_model_runner.token_to_kv_pool.swa_kv_pool.kv_buffer)
        initial_kv = tuple(value.clone() for value in kv)
        bands, executables, weight_cache = {}, dict(prepared.executables), {}
        reference_names = {"original"}
        if args.equal_work:
            reference_names.update(("target_prepared", "equal_work"))
            input_graph = prepared.inputs["original"]
            for name, pieces in (
                ("target_prepared", (target_graph, input_graph, graft.graph)),
                ("equal_work", (target_graph, input_graph, graft.graph, confidence.graph)),
            ):
                executables[name] = DeviceGraphExecutable.compose(pieces, device_launch=False)
        state["fixture"] = prepared.metadata
        if args.equal_work:
            state["timed_work_by_arm"] = {
                name: {
                    "target_refresh": name != "original",
                    "confidence_and_sts": name in ("equal_work", "retained", "integrated"),
                    "proposal_input_and_id_publication": True,
                }
                for name in (*reference_names, "retained", "integrated")
            }
        state["status"] = "capturing_full_proposals"
        save()
        for arm in plan.arms:
            name = "retained" if arm.label == "proposal_megakernel" else arm.label
            with packing_environment(arm):
                # The frozen loader predates per-arm geometry in its plan ABI.
                # Scope the existing host launch flag here and verify readback.
                os.environ["DSPARK_V4_CLUSTER_DIM"] = "2"
                band = capture_module.capture_proposal_megakernel_band(
                    args.snapshot,
                    graft=graft,
                    target_hidden=inject.target_hidden,
                    bonus=bonus,
                    commit_len=commit_len,
                    new_seq_len=new_seq_len,
                    candidates=candidates,
                    lowering=compile_gb300_graft_lowering(GraftExecutor.PROPOSAL_MEGAKERNEL),
                    weight_cache=weight_cache,
                    extension_module=arm.module,
                    integrated_io=name != "retained",
                )
            pieces = (band.input_graph, band.graph) if name == "retained" else (band.graph,)
            executable = DeviceGraphExecutable.compose(pieces, device_launch=False)
            bands[name], executables[name] = band, executable
        if args.equal_work:
            state["equal_work_contract"]["native_parameters"] = {
                name: confidence.verify_native_parameters(band) for name, band in bands.items()
            }
        nc = load("full_io_census", root / "fresh-w2-real-input-0909/comparison-r2/node_census.py")
        tg = load("full_io_graph", root / "hazy-original-fused64-timing-draft-0911/timed_graph.py")
        tg.nc = nc
        records = {"original": graph_record(graft.graph, nc, tg)}
        if args.equal_work:
            emit(out / "target-work-graph.json", graph_record(target_graph, nc, tg))
            emit(out / "confidence-graph.json", graph_record(confidence.graph, nc, tg))
        for name, band in bands.items():
            records[name] = graph_record(band.graph, nc, tg)
            emit(out / f"{name}-input-graph.json", graph_record(band.input_graph, nc, tg))
            nodes = [
                n
                for n in records[name]["nodes"]
                if "dspark_v4_scheduler_kernel" in n.get("name", "")
            ]
            if len(nodes) != 1:
                raise ValueError("Expected one full-model scheduler body")
            cluster = nodes[0]["attrs"]["CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION"]
            if cluster["status"] != 0 or struct.unpack_from(
                "<III", bytes.fromhex(cluster["hex"])
            ) != (2, 1, 1):
                raise ValueError("Actual full-proposal cluster geometry differs from two")
        for name, record in records.items():
            emit(out / f"{name}-graph.json", record)
        if records["original"]["pdl_edges"] <= 0:
            raise ValueError("The original graph lost its PDL dependencies")
        for name, candidate_graph in records.items():
            if name in ("original", "retained"):
                continue
            if (
                candidate_graph["node_count"] != 1
                or candidate_graph["pdl_edges"]
                or "dspark_v4_scheduler_kernel" not in candidate_graph["nodes"][0].get("name", "")
            ):
                raise ValueError("The integrated proposal is not one scheduler kernel")
        state["candidate_uses_one_kernel"] = True
        cu = nc.cu
        for name, executable in executables.items():
            flags = nc.checked(cu.cuGraphExecGetFlags(cu.CUgraphExec(executable.handle)))
            if int(flags) != 0:
                raise ValueError(f"{name} graph executor flags differ from zero")

        def reset(name):
            """Restore accepted external state; the proposal owns its internal epoch and scratch."""
            candidates.copy_(initial_candidates)
            for field_name, value in fields.items():
                value.copy_(initial_fields[field_name])
            for value, initial in zip(kv, initial_kv, strict=True):
                value.copy_(initial)

        def launch(name):
            """Use the identical CUDA driver launch API for all three executables."""
            nc.checked(
                cu.cuGraphLaunch(
                    cu.CUgraphExec(executables[name].handle),
                    cu.CUstream(torch.cuda.current_stream().cuda_stream),
                )
            )

        def evidence(name):
            """Read live semantic outputs before another arm can overwrite any shared storage."""
            if name in reference_names:
                values = {
                    "ids": candidates.clone(),
                    "head_normalized": graft.head_normalized.clone(),
                    "base_logits": graft.head_logits.clone(),
                }
                if args.equal_work:
                    values["head_hidden"] = head_hidden.clone()
                    if name == "equal_work":
                        values.update(
                            {key: value.clone() for key, value in confidence.outputs.items()}
                        )
                return values
            band = bands[name]
            workspace, outputs = band.keepalive[2:4]
            values = {
                "ids": candidates.clone(),
                "output_ids": outputs.output_ids.clone(),
                "corrected_logits": outputs.corrected_logits.clone(),
                "confidence_logits": outputs.confidence_logits.clone(),
                "calibrated_confidences": outputs.calibrated_confidences.clone(),
                "kv_cache": band.keepalive[1].out.clone(),
            }
            for key in (
                "head_normalized",
                "base_logits",
                "router_indices",
                "routed_output",
                "shared_output",
            ):
                values[key] = view_v4_workspace_region(workspace, key).clone()
            values["head_normalized"] = values["head_normalized"].reshape(5, 4096)
            values["base_logits"] = values["base_logits"].reshape(5, -1)
            if args.equal_work:
                values["head_hidden"] = (
                    view_v4_workspace_region(workspace, "head_hidden").clone().reshape(5, 4096)
                )
            return values

        def validate(tag):
            """Require exact integrated math and reference IDs before publishing timing."""
            saved = {}
            for name in executables:
                reset(name)
                if args.equal_work and name == "equal_work":
                    with torch.inference_mode():
                        for value in confidence.outputs.values():
                            value.fill_(float("nan"))
                if name not in reference_names:
                    bands[name].proposal_outputs.output_ids.fill_(-777)
                    bands[name].keepalive[1].out.fill_(float("nan"))
                launch(name)
                torch.cuda.synchronize()
                if disjoint_fixture is not None:
                    state["fixture_integrity"][f"{tag}/{name}"] = disjoint_fixture.verify()
                values = evidence(name)
                if args.equal_work and name == "equal_work":
                    last = graft.bundle.draft_model.stages[-1]
                    old_normalized = torch.empty_like(graft.head_normalized)
                    graft_module._equal_work_head_export.original(
                        graft.head_streams,
                        last.hc_head_fn,
                        last.hc_head_scale,
                        last.hc_head_base,
                        last.norm.weight,
                        output=old_normalized,
                        norm_eps=float(graft.bundle.draft_model.norm_eps),
                        hc_eps=float(graft.bundle.draft_model.hc_eps),
                    )
                    torch.testing.assert_close(
                        values["head_normalized"], old_normalized, rtol=0, atol=0
                    )
                    state.setdefault("head_export_parity", {})[tag] = True
                save_tensors(torch, out / tag / name, values)
                if (
                    name not in reference_names
                    and torch.count_nonzero(bands[name].keepalive[1].status).item()
                ):
                    raise ValueError("KV gather reported an invalid live mapping")
                if name not in reference_names | {"retained"}:
                    bands[name].io_binding.verify_resident()
                    emit(out / tag / name / "io-binding.json", bands[name].io_binding.record())
                saved[name] = values
            differences = {
                key: error(torch, value, saved["integrated"][key])
                for key, value in saved["retained"].items()
            }
            original_errors = {
                key: error(
                    torch,
                    value,
                    saved["integrated"][key],
                    allow_dtype_difference=key == "base_logits",
                )
                for key, value in saved["original"].items()
                if key in saved["integrated"]
            }
            state["correctness"][tag] = {
                "retained_vs_integrated": differences,
                "original_vs_integrated": original_errors,
            }
            if args.equal_work:
                equal_errors = {
                    key: error(torch, value, saved["integrated"][key], allow_dtype_difference=True)
                    for key, value in saved["equal_work"].items()
                    if key in saved["integrated"]
                }
                state["correctness"][tag]["equal_work_vs_integrated"] = equal_errors
                own_input_errors = {}
                for name in ("equal_work", "integrated"):
                    oracle = confidence.evaluate(saved[name]["head_hidden"], saved[name]["ids"][:5])
                    own_input_errors[name] = {
                        key: error(torch, oracle[key], saved[name][key])
                        for key in ("confidence_logits", "calibrated_confidences")
                    }
                    save_tensors(torch, out / tag / (name + "-confidence-oracle"), oracle)
                    fp64 = confidence.evaluate_fp64(
                        saved[name]["head_hidden"], saved[name]["ids"][:5]
                    )
                    save_tensors(torch, out / tag / (name + "-confidence-fp64"), fp64)
                    for key in ("confidence_logits", "calibrated_confidences"):
                        torch.testing.assert_close(
                            saved[name][key], oracle[key], atol=2e-4, rtol=2e-5
                        )
                state["correctness"][tag]["confidence_on_own_inputs"] = own_input_errors
                if equal_errors["ids"]["different_elements"] or not all(
                    value["finite"] for value in equal_errors.values()
                ):
                    raise ValueError("Equal-work output IDs differ or an output is nonfinite")
            save()
            if any(
                not value["bitwise_equal"] or not value["finite"] for value in differences.values()
            ):
                raise ValueError("Integrated IO changed the retained proposal math")
            if original_errors["ids"]["different_elements"] or not all(
                value["finite"] for value in original_errors.values()
            ):
                raise ValueError("Integrated proposal differs from reference IDs or is nonfinite")
            return saved

        validate("before")
        state["status"] = "timing"
        save()
        start, end = nc.checked(cu.cuEventCreate(0)), nc.checked(cu.cuEventCreate(0))
        comparisons = [
            ("original", "retained"),
            ("retained", "integrated"),
            ("original", "integrated"),
        ]
        if args.equal_work:
            comparisons = [
                ("original", "integrated"),
                ("original", "target_prepared"),
                ("target_prepared", "equal_work"),
                *(("equal_work", "integrated"),) * 3,
            ]
        for reference, candidate in comparisons:
            pairs = []
            for iteration in range(-8, 32):
                order = [reference, candidate] if iteration % 2 == 0 else [candidate, reference]
                row = {"round": iteration, "order": order, "us": {}, "submission_status": {}}
                for name in order:
                    reset(name)
                    launch(name)
                    torch.cuda.synchronize()
                    reset(name)
                    stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
                    torch.cuda._sleep(8_000_000)
                    nc.checked(cu.cuEventRecord(start, stream))
                    launch(name)
                    nc.checked(cu.cuEventRecord(end, stream))
                    query = cu.cuEventQuery(start)
                    if int(query[0]) != int(cu.CUresult.CUDA_ERROR_NOT_READY):
                        raise ValueError("Timed start ran before complete host submission")
                    nc.checked(cu.cuEventSynchronize(end))
                    row["us"][name] = 1000 * float(nc.checked(cu.cuEventElapsedTime(start, end)))
                    row["submission_status"][name] = int(query[0])
                pairs.append(row)
            measured = pairs[8:]
            deltas = [row["us"][candidate] - row["us"][reference] for row in measured]
            record = {
                "reference": reference,
                "candidate": candidate,
                "pairs": pairs,
                "p50_us": {
                    name: statistics.median(row["us"][name] for row in measured)
                    for name in (reference, candidate)
                },
                "paired_delta_us": statistics.median(deltas),
                "pairs_faster": sum(delta < 0 for delta in deltas),
                "diagnostic_only": False,
                "candidate_numerically_qualified": None,
                "timing_scope": (
                    "See timed_work_by_arm; fixed-state greedy proposal outputs"
                    if args.equal_work
                    else "Existing fixed-state numerical screens"
                ),
            }
            state["comparisons"].append(record)
            if disjoint_fixture is not None:
                state["fixture_integrity"][f"post-comparison-{len(state['comparisons'])}"] = (
                    disjoint_fixture.verify()
                )
            bands["integrated"].io_binding.verify_resident()
            emit(
                out / f"io-after-comparison-{len(state['comparisons'])}.json",
                bands["integrated"].io_binding.record(),
            )
            print(
                "FULL_PROPOSAL",
                reference,
                candidate,
                record["p50_us"],
                record["paired_delta_us"],
                flush=True,
            )
            save()
        validate("after")
        for name in executables:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                for repeat in range(3):
                    reset(name)
                    torch.cuda.synchronize()
                    with torch.profiler.record_function(f"{name}/proposal/{repeat}"):
                        launch(name)
                        torch.cuda.synchronize()
            profiler.export_chrome_trace(str(out / f"{name}.perfetto.json"))
        validate("after-profiler")
        state.update(
            complete=True,
            execution_complete=True,
            status="complete",
        )
    except BaseException as failure:
        state.update(status="failed", error=repr(failure), traceback=traceback.format_exc())
        raise
    finally:
        save()


def main():
    """Run the explicit single-GPU experiment using real retained prompt and target fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "artifact-root",
        "output",
        "snapshot",
        "prefill-snapshot",
        "target-taps",
        "build-record",
        "overlay",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dist-port", type=int, default=29617)
    parser.add_argument(
        "--equal-work",
        action="store_true",
        help="Time one target-row refresh and confidence in the reference",
    )
    parser.add_argument(
        "--disjoint-draft-kv-fixture",
        action="store_true",
        help="Use a new reference fixture with five draft slots separate from live prefix KV",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_LAUNCH_BLOCKING"):
        raise ValueError("CUDA_LAUNCH_BLOCKING invalidates this performance experiment")
    run(args)


if __name__ == "__main__":
    main()
