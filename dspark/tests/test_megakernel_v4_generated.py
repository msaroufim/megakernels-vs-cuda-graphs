import re
from collections import Counter
from pathlib import Path

import pytest

from deepspec.megakernel import (
    DeepSeekV4MegaKernelSpec,
    V4LaunchShape,
    build_v4_phase_program,
)
from scripts.megakernel.generate_v4_header import generate


def _generated_array(header: str, name: str) -> list[int]:
    match = re.search(rf"\b{name}\[\d+\] = \{{(.*?)\n\}};", header, re.DOTALL)
    assert match is not None
    return [int(value) for value in re.findall(r"-?\d+", match.group(1))]


def test_committed_v4_cuda_header_matches_python_contract():
    root = Path(__file__).resolve().parents[1]
    header = root / "deepspec/megakernel/csrc/dspark_v4_generated.h"
    assert header.read_text() == generate()


def test_committed_v4_relaxed_cuda_header_matches_python_program():
    root = Path(__file__).resolve().parents[1]
    header = root / "deepspec/megakernel/csrc/dspark_v4_generated_relaxed.h"
    assert header.read_text() == generate(relaxed=True)


def test_committed_v4_greedy_cuda_header_matches_python_program():
    root = Path(__file__).resolve().parents[1]
    header = root / "deepspec/megakernel/csrc/dspark_v4_generated_greedy.h"
    assert header.read_text() == generate(relaxed=True, greedy=True)


def test_committed_v4_fine_overlap_header_fuses_routed_expert_half_bands():
    root = Path(__file__).resolve().parents[1]
    header_path = root / "deepspec/megakernel/csrc/dspark_v4_generated_greedy_overlap.h"
    header = generate(relaxed=True, greedy=True, fine_overlap=True)
    assert header_path.read_text() == header

    spec = DeepSeekV4MegaKernelSpec()
    shape = V4LaunchShape()
    program = build_v4_phase_program(
        spec,
        shape,
        relaxed_tail=True,
        greedy_tail=True,
    )
    by_name = {phase.name: phase.phase_id for phase in program}
    phases = _generated_array(header, "kStaticTaskPhases")
    claims = _generated_array(header, "kStaticTaskClaims")
    work_units = _generated_array(header, "kWorkUnits")
    dependency_counts = _generated_array(header, "kDependencyCounts")
    streaming_consumers = _generated_array(header, "kStreamingConsumerForPhase")

    assert "kStreamEventCount" not in header
    assert "kStaticTaskWaitEvents" not in header
    assert len(phases) == len(claims)

    for layer in range(3):
        producer = by_name[f"layer_{layer}.fp4_routed_w13"]
        consumer = by_name[f"layer_{layer}.routed_swiglu"]
        assert streaming_consumers[producer] == consumer
        assert dependency_counts[consumer] == 0

        producer_slots = [slot for slot, phase in enumerate(phases) if phase == producer]
        consumer_slots = [slot for slot, phase in enumerate(phases) if phase == consumer]
        assert len(producer_slots) == 240
        assert sorted(claims[slot] for slot in producer_slots) == list(range(240))
        # The 120 semantic 512-feature SwiGLU units become 240 physical
        # half-band completions, but no standalone consumer task remains:
        # each producer CTA executes and finishes its paired half-band.
        assert work_units[consumer] == 240
        assert consumer_slots == []


@pytest.mark.parametrize(
    "shared_first,hc_split,woa_split_k,w13_fused_swiglu",
    [
        (False, False, 1, False),
        (True, False, 1, False),
        (True, True, 1, False),
        (True, True, 2, False),
        (True, True, 1, True),
        (True, True, 2, True),
    ],
)
def test_gb300_static_worker_queues_cover_each_claim_in_topological_order(
    shared_first, hc_split, woa_split_k, w13_fused_swiglu
):
    root = Path(__file__).resolve().parents[1]
    spec = DeepSeekV4MegaKernelSpec()
    shape = V4LaunchShape()
    program = build_v4_phase_program(
        spec,
        shape,
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=hc_split,
        woa_split_k=woa_split_k,
        w13_fused_swiglu=w13_fused_swiglu,
    )
    header = generate(
        relaxed=True,
        greedy=True,
        shared_first=shared_first,
        hc_split=hc_split,
        woa_split_k=woa_split_k,
        w13_fused_swiglu=w13_fused_swiglu,
    )
    offsets = _generated_array(header, "kStaticWorkerTaskOffsets")
    phases = _generated_array(header, "kStaticTaskPhases")
    claims = _generated_array(header, "kStaticTaskClaims")
    chunks = _generated_array(header, "kClaimChunks")
    execution_groups = _generated_array(header, "kExecutionGroups")
    phase_layers = _generated_array(header, "kPhaseLayers")

    assert len(offsets) == shape.workers + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(phases) == len(claims)
    assert offsets == sorted(offsets)
    assert len(execution_groups) == len(program)
    assert len(phase_layers) == len(program)
    assert phase_layers == [
        int(phase.name.split(".", 1)[0].removeprefix("layer_"))
        if phase.name.startswith("layer_")
        else 0xFF
        for phase in program
    ]
    assert set(execution_groups) == set(range(12))
    assert execution_groups[:12] == [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2]
    assert all(
        execution_groups[phase.phase_id] == 10
        for phase in program
        if phase.name.startswith("head.")
    )
    assert all(
        execution_groups[phase.phase_id] == 11
        for phase in program
        if phase.name.startswith("tail_") or phase.name == "proposal.finalize_audit"
    )

    expected = Counter()
    by_name = {phase.name: phase for phase in program}
    for phase in program:
        count = (phase.work_units + chunks[phase.phase_id] - 1) // chunks[phase.phase_id]
        expected.update((phase.phase_id, claim) for claim in range(count))
    assert Counter(zip(phases, claims, strict=True)) == expected

    # Queue-order edges plus DAG dependency edges must remain acyclic. This is
    # stronger than requiring monotonically increasing phase depth: the AOT
    # list scheduler may legally prioritize a long independent branch, but it
    # must never create a cross-worker head-of-line cycle.
    edges = {phase.phase_id: set() for phase in program}
    for phase in program:
        for name in phase.dependencies:
            edges[by_name[name].phase_id].add(phase.phase_id)
    for worker in range(shape.workers):
        begin, end = offsets[worker : worker + 2]
        worker_phases = phases[begin:end]
        for previous, following in zip(worker_phases, worker_phases[1:]):
            if previous != following:
                edges[previous].add(following)
    indegree = [0] * len(program)
    for successors in edges.values():
        for successor in successors:
            indegree[successor] += 1
    ready = [phase_id for phase_id, count in enumerate(indegree) if count == 0]
    visited = 0
    while ready:
        phase_id = ready.pop()
        visited += 1
        for successor in edges[phase_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    assert visited == len(program)

    source = (root / "deepspec/megakernel/csrc/dspark_v4_kernel.cu").read_text()
    assert "switch (generated::kExecutionGroups[task.phase])" in source


def test_hc_split_releases_compute_independently_and_waits_before_residuals():
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=True,
    )
    phases = {phase.name: phase for phase in program}
    for layer in range(3):
        for band, consumer, residual in (
            ("attn", "q_a", "attn_hc_post"),
            ("ffn", "router_sqrtsoftplus_top6", "expert_combine_hc_post"),
        ):
            prefix = f"layer_{layer}."
            norm = prefix + band + "_hc_reduce_rmsnorm"
            sinkhorn = prefix + band + "_hc_sinkhorn"
            assert phases[norm].dependencies == (prefix + band + "_hc_projection",)
            assert phases[sinkhorn].dependencies == phases[norm].dependencies
            assert phases[prefix + consumer].dependencies == (norm,)
            assert sinkhorn in phases[prefix + residual].dependencies


@pytest.mark.parametrize("mode", (1, 2))
def test_gap_schedule_covers_claims_without_worker_dependency_deadlocks(mode):
    """Check actual claim ordering; independent phases may legally interleave."""
    import graphlib

    options = dict(relaxed=True, greedy=True, shared_first=True, hc_split=True)
    original = generate(**options)
    header = generate(**options, gap_schedule=mode)
    phases = _generated_array(header, "kStaticTaskPhases")
    claims = _generated_array(header, "kStaticTaskClaims")
    offsets = _generated_array(header, "kStaticWorkerTaskOffsets")
    assert Counter(zip(phases, claims, strict=True)) == Counter(
        zip(
            _generated_array(original, "kStaticTaskPhases"),
            _generated_array(original, "kStaticTaskClaims"),
            strict=True,
        )
    )
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=True,
    )
    by_name = {phase.name: phase.phase_id for phase in program}
    predecessor_offsets = _generated_array(header, "kPredecessorOffsets")
    predecessors = _generated_array(header, "kPredecessors")
    for phase in program:
        begin, end = predecessor_offsets[phase.phase_id : phase.phase_id + 2]
        assert predecessors[begin:end] == [by_name[name] for name in phase.dependencies]
    graph = {}
    for phase in program:
        graph[("start", phase.phase_id)] = {("done", by_name[name]) for name in phase.dependencies}
        graph[("done", phase.phase_id)] = set()
    for begin, end in zip(offsets, offsets[1:]):
        previous = None
        for slot in range(begin, end):
            phase, claim = phases[slot], claims[slot]
            task = (phase, claim)
            graph[task] = {("start", phase)}
            if previous is not None:
                graph[task].add(previous)
            graph[("done", phase)].add(task)
            previous = task
    assert len(tuple(graphlib.TopologicalSorter(graph).static_order())) == len(graph)


def test_woa_split_reduces_before_reusing_partial_storage():
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=True,
        woa_split_k=2,
    )
    phases = {phase.name: phase for phase in program}
    for layer in range(3):
        compute = phases[f"layer_{layer}.inverse_rope_grouped_wo_a"]
        reduce = phases[f"layer_{layer}.wo_a_reduce_quant"]
        wob = phases[f"layer_{layer}.wo_b"]
        assert compute.work_units == 128
        assert reduce.work_units == 64
        assert reduce.dependencies == (compute.name,)
        assert wob.dependencies == (reduce.name,)


@pytest.mark.parametrize("woa_split_k", [1, 2])
def test_fused_w13_publishes_swiglu_directly_to_w2(woa_split_k):
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=True,
        woa_split_k=woa_split_k,
        w13_fused_swiglu=True,
    )
    phases = {phase.name: phase for phase in program}
    for layer in range(3):
        assert f"layer_{layer}.routed_swiglu" not in phases
        assert phases[f"layer_{layer}.fp4_routed_w2"].dependencies == (
            f"layer_{layer}.fp4_routed_w13",
        )


def test_group_ready_keeps_producers_ahead_of_waiting_consumers():
    options = dict(
        relaxed=True,
        greedy=True,
        shared_first=True,
        hc_split=True,
        woa_split_k=2,
        w13_fused_swiglu=True,
        routed_group_ready=True,
    )
    header = generate(**options)
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=True,
        woa_split_k=2,
        w13_fused_swiglu=True,
        routed_group_ready=True,
    )
    by_name = {phase.name: phase for phase in program}
    offsets = _generated_array(header, "kStaticWorkerTaskOffsets")
    phases = _generated_array(header, "kStaticTaskPhases")
    claims = _generated_array(header, "kStaticTaskClaims")
    for layer in range(3):
        producer = by_name[f"layer_{layer}.fp4_routed_w13"].phase_id
        consumer = by_name[f"layer_{layer}.fp4_routed_w2"].phase_id
        assert by_name[f"layer_{layer}.fp4_routed_w2"].dependencies == (
            f"layer_{layer}.router_top6",
        )
        assert sorted(claims[i] for i, phase in enumerate(phases) if phase == producer) == list(
            range(240)
        )
        for worker in range(152):
            queue = phases[offsets[worker] : offsets[worker + 1]]
            # Workers with no producer work can wait without blocking a producer.
            # Every resident worker that owns producers must finish those slots
            # before entering any consumer wait. Dynamic claiming ensures all
            # producer streams are claimed before a producer slot is drained.
            if producer in queue and consumer in queue:
                assert max(i for i, phase in enumerate(queue) if phase == producer) < queue.index(
                    consumer
                )


def test_shipped_headers_match_generator_and_include_retained_full_proposal():
    """Check the selected schedule and every base mode after retiring snapshots."""
    from scripts.megakernel.generate_v4_header import _VARIANTS, header_name

    csrc = Path(__file__).resolve().parents[1] / "deepspec/megakernel/csrc"
    names = {header_name(**variant) for variant in _VARIANTS}
    assert len(names) == 8
    assert "dspark_v4_generated_greedy_shared_continue_w2_128.h" in names
    for variant in _VARIANTS:
        assert (csrc / header_name(**variant)).read_text() == generate(**variant)
