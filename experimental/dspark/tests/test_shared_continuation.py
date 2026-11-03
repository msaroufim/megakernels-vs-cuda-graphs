"""Validate the schedule-only shared continuation against the retained queues."""

from collections import Counter

from deepspec.megakernel import DeepSeekV4MegaKernelSpec, V4LaunchShape, build_v4_phase_program
from scripts.megakernel.generate_v4_header import generate
from tests.test_megakernel_v4_generated import _generated_array

OPTIONS = dict(
    relaxed=True,
    greedy=True,
    shared_first=True,
    hc_split=True,
    woa_split_k=2,
    w13_fused_swiglu=True,
)


def _queues(header):
    offsets = _generated_array(header, "kStaticWorkerTaskOffsets")
    phases = _generated_array(header, "kStaticTaskPhases")
    claims = _generated_array(header, "kStaticTaskClaims")
    return [list(zip(phases[b:e], claims[b:e])) for b, e in zip(offsets, offsets[1:])]


def test_continuation_changes_only_shared_consumer_placement():
    base = generate(**OPTIONS)
    new = generate(**OPTIONS, shared_continue=True)
    oldq, newq = _queues(base), _queues(new)
    assert Counter(t for q in oldq for t in q) == Counter(t for q in newq for t in q)
    program = build_v4_phase_program(
        DeepSeekV4MegaKernelSpec(),
        V4LaunchShape(),
        relaxed_tail=True,
        greedy_tail=True,
        hc_split=True,
        woa_split_k=2,
        w13_fused_swiglu=True,
    )
    by_name = {p.name: p.phase_id for p in program}
    moved = {p.phase_id for p in program if p.name.endswith((".shared_swiglu", ".fp8_shared_w2"))}
    for before, after in zip(oldq, newq):
        assert [t for t in before if t[0] not in moved] == [t for t in after if t[0] not in moved]
    for layer in range(3):
        producer = by_name[f"layer_{layer}.fp8_shared_w13"]
        swiglu = by_name[f"layer_{layer}.shared_swiglu"]
        w2 = by_name[f"layer_{layer}.fp8_shared_w2"]
        routed = by_name[f"layer_{layer}.fp4_routed_w13"]
        owners = [w for w, q in enumerate(oldq) if any(p == producer for p, _ in q)]
        assert len(owners) == 32
        for worker in owners:
            q = newq[worker]
            start = next(i for i, (p, _) in enumerate(q) if p == producer)
            end = next(i for i, (p, _) in enumerate(q) if p == routed)
            inserted = [p for p, _ in q[start + 1 : end]]
            assert inserted in ([swiglu, w2], [w2])
        for worker in set(range(152)) - set(owners):
            before = [p for p, _ in oldq[worker]]
            after = [p for p, _ in newq[worker]]
            # No work is added ahead of the first routed slot on these 120 CTAs.
            # Shared consumers from earlier layers are only removed.
            assert [p for p in before[: before.index(routed) + 1] if p not in moved] == after[
                : after.index(routed) + 1
            ]
    # Existing phase readiness and semantic work counts are untouched.
    for name in ("kWorkUnits", "kClaimChunks", "kDependencyCounts", "kSuccessors", "kPredecessors"):
        assert _generated_array(base, name) == _generated_array(new, name)
    edges = {p.phase_id: {by_name[d] for d in p.dependencies} for p in program}
    for q in newq:
        for (a, _), (b, _) in zip(q, q[1:]):
            if a != b:
                edges[b].add(a)
    completed = set()
    while True:
        ready = {p for p, deps in edges.items() if p not in completed and deps <= completed}
        if not ready:
            break
        completed |= ready
    assert len(completed) == len(program)
