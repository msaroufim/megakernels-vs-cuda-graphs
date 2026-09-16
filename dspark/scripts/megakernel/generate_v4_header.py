#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))


def _array(name: str, ctype: str, values: list[int], *, columns: int = 12) -> str:
    lines = [f"static __device__ __constant__ {ctype} {name}[{len(values)}] = {{"]
    for index in range(0, len(values), columns):
        chunk = ", ".join(str(value) for value in values[index : index + columns])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)


# Fused routed tile alignment. Must equal the largest DSPARK_W13_DG_TILE_PAIRS
# any shipped build compiles with (deepspec/megakernel/csrc/dspark_w13_phase.cuh):
# the routed claim chunk is rounded up to this multiple so every claim range
# contains whole fused-tile groups. Bitwise-neutral -- the chunk only decides
# how many consecutive items one CTA walks.
_ROUTED_CLAIM_ALIGNMENT = 2


def generate(
    relaxed: bool = False,
    greedy: bool = False,
    batch: int = 1,
    fine_overlap: bool = False,
    shared_first: bool = False,
    hc_split: bool = False,
    gap_schedule: int = 0,
    woa_split_k: int = 1,
    w13_fused_swiglu: bool = False,
    routed_group_ready: bool = False,
    routed_interleave: bool = False,
    front_embed_first: bool = False,
    shared_continue: bool = False,
    routed_w2_workers128: bool = False,
) -> str:
    if gap_schedule and not hc_split:
        raise ValueError("gap scheduling requires the HC split schedule")
    from deepspec.megakernel import (
        DeepSeekV4MegaKernelSpec,
        V4LaunchShape,
        V4WorkerRole,
        build_v4_launch_abi,
        build_v4_phase_program,
    )
    from deepspec.megakernel.v4_weights import (
        V4ExpertWeightSlot,
        V4GlobalWeightSlot,
        V4LayerWeightSlot,
        V4WeightTableLayout,
        plan_v4_weight_arena,
    )

    if greedy and not relaxed:
        raise ValueError("greedy generation requires relaxed=True")
    if batch != 1 and not (relaxed and greedy):
        # Batched serving is a relaxed/greedy-build feature; the contract and
        # stage-1 relaxed headers stay batch-1 and byte-identical.
        raise ValueError("batched generation requires --relaxed --greedy")
    if fine_overlap and not (relaxed and greedy and batch == 1):
        raise ValueError("fine-overlap generation requires the batch-1 relaxed greedy program")
    if shared_first and not (relaxed and greedy and batch == 1 and not fine_overlap):
        raise ValueError("shared-first generation requires batch-1 relaxed greedy without overlap")
    if hc_split and not (shared_first and relaxed and greedy and batch == 1 and not fine_overlap):
        raise ValueError("HC split generation requires batch-1 shared-first relaxed greedy")
    if woa_split_k not in (1, 2) or (woa_split_k != 1 and (not hc_split or gap_schedule)):
        raise ValueError("WO_A split-K2 requires HC split without gap scheduling")
    if w13_fused_swiglu and (not hc_split or gap_schedule):
        raise ValueError("W13 SwiGLU fusion requires HC split without gap scheduling")
    if routed_interleave and not (
        hc_split
        and woa_split_k == 2
        and not w13_fused_swiglu
        and not gap_schedule
        and not routed_group_ready
    ):
        raise ValueError("interleaved routing requires the unfused HC split2 program")
    if front_embed_first and not (
        w13_fused_swiglu and woa_split_k == 2 and not routed_group_ready and not routed_interleave
    ):
        raise ValueError("front embedding priority requires the fused W13/split2 program")
    if shared_continue and not (
        w13_fused_swiglu
        and woa_split_k == 2
        and hc_split
        and not routed_interleave
        and not routed_group_ready
        and not front_embed_first
    ):
        raise ValueError("shared continuation requires the fused W13/WO_A split2 schedule")
    if routed_w2_workers128 and not shared_continue:
        raise ValueError("W2 width128 requires the shared-continuation schedule")
    spec = DeepSeekV4MegaKernelSpec()
    shape = V4LaunchShape(batch_size=batch)
    program = build_v4_phase_program(
        spec,
        shape,
        relaxed_tail=relaxed,
        greedy_tail=greedy,
        hc_split=hc_split,
        woa_split_k=woa_split_k,
        w13_fused_swiglu=w13_fused_swiglu,
        routed_group_ready=routed_group_ready,
    )
    abi = build_v4_launch_abi(spec, shape)
    if (
        routed_group_ready
        and len(program) * 8 + 30 * 4 > abi.workspace.region("phase_dependency_arrivals").nbytes
    ):
        raise ValueError("group readiness counters do not fit unused dependency storage")
    by_name = {phase.name: phase.phase_id for phase in program}
    from deepspec.megakernel.v4_schedule import router_chain_geometry

    router_chains, router_rows = router_chain_geometry()

    def tail_phase(name: str) -> int:
        # Relaxed DAG fuses confidence/correct/sum_reduce away; their
        # dispatch constants become 0xFFFF sentinels no task.phase matches.
        return by_name.get(name, 0xFFFF)

    # Fine-overlap headers lower routed-W13 -> routed-SwiGLU as per-claim
    # vertical fusion. Keep the semantic DAG unchanged: this is a physical
    # lowering choice, not a model rewrite.
    streaming_edges = (
        {
            (
                by_name[f"layer_{layer}.fp4_routed_w13"],
                by_name[f"layer_{layer}.routed_swiglu"],
            )
            for layer in range(3)
        }
        if fine_overlap
        else set()
    )
    normal_dependency_counts = [0] * len(program)
    semantic_successors: list[list[int]] = [[] for _ in program]
    successors: list[list[int]] = [[] for _ in program]
    for phase in program:
        for dependency in phase.dependencies:
            predecessor = by_name[dependency]
            semantic_successors[predecessor].append(phase.phase_id)
            if (predecessor, phase.phase_id) in streaming_edges:
                continue
            successors[predecessor].append(phase.phase_id)
            normal_dependency_counts[phase.phase_id] += 1
    successor_offsets = [0]
    successor_values: list[int] = []
    for values in successors:
        successor_values.extend(values)
        successor_offsets.append(len(successor_values))
    predecessor_offsets = [0]
    predecessor_values: list[int] = []
    for phase in program:
        predecessor_values.extend(by_name[name] for name in phase.dependencies)
        predecessor_offsets.append(len(predecessor_values))
    roots = [phase.phase_id for phase in program if not phase.dependencies]

    # The persistent kernel has one execution body for each of these groups.
    # Emit ownership once so the static AOT path can jump directly to the
    # owning dispatcher instead of asking all twelve dispatchers to reject the
    # same task. Fail generation if a new phase has no owner.
    execution_group_names = (
        "main",
        "embedding",
        "main_kv",
        "attn_hc",
        "attn_projection",
        "sparse_attention",
        "attention_output",
        "ffn_hc",
        "router",
        "expert",
        "head",
        "tail",
    )

    def execution_group(phase) -> int:
        name = phase.name
        if name in {
            "main.activation_quant",
            "main.projection",
            "main.split_k_reduce",
            "main.rmsnorm",
        }:
            return 0
        if name == "draft.embedding_hc_expand":
            return 1
        if name == "main.projected_activation_quant" or any(
            name.endswith(suffix)
            for suffix in (".main_kv_projection", ".main_kv_norm_rope_quant_store")
        ):
            return 2
        if any(
            name.endswith(suffix)
            for suffix in (
                ".attn_hc_projection",
                ".attn_hc_sinkhorn",
                ".attn_hc_reduce_rmsnorm",
            )
        ):
            return 3
        if any(
            name.endswith(suffix)
            for suffix in (
                ".q_a",
                ".q_a_rmsnorm",
                ".q_b",
                ".draft_kv_projection",
                ".q_norm_rope",
                ".draft_kv_norm_rope_quant",
            )
        ):
            return 4
        if name.endswith(".sparse_attention"):
            return 5
        if any(
            name.endswith(suffix)
            for suffix in (
                ".inverse_rope_grouped_wo_a",
                ".wo_a_reduce_quant",
                ".wo_b",
                ".attn_hc_post",
            )
        ):
            return 6
        if any(
            name.endswith(suffix)
            for suffix in (
                ".ffn_hc_projection",
                ".ffn_hc_sinkhorn",
                ".ffn_hc_reduce_rmsnorm",
            )
        ):
            return 7
        if any(name.endswith(suffix) for suffix in (".router_sqrtsoftplus_top6", ".router_top6")):
            return 8
        if any(
            name.endswith(suffix)
            for suffix in (
                ".fp4_routed_w13",
                ".routed_swiglu",
                ".fp4_routed_w2",
                ".fp8_shared_w13",
                ".shared_swiglu",
                ".fp8_shared_w2",
                ".expert_combine_hc_post",
            )
        ):
            return 9
        if name.startswith("head."):
            return 10
        if (
            name.startswith("tail_")
            or name.startswith("prefix.")
            or name == "proposal.finalize_audit"
        ):
            return 11
        raise ValueError(f"V4 phase {name!r} has no execution group")

    execution_groups = [execution_group(phase) for phase in program]
    # A masked proposal segment may select one draft layer while retaining the
    # same execution family for all three layers.  Keep the layer ownership in
    # the generated contract so the static scheduler can collapse an omitted
    # phase to one completion task instead of replaying every no-op tile.
    phase_layers = [
        int(phase.name.split(".", 1)[0].removeprefix("layer_"))
        if phase.name.startswith("layer_")
        else 0xFF
        for phase in program
    ]

    def claim_chunk(phase) -> int:
        if relaxed and phase.name.endswith(".expert_combine_hc_post"):
            # 160 one-tile claims spill a second wave onto a 152-worker grid
            # that is 8/152 occupied for it. Two tiles per claim retires the
            # band in one wave; the body's items are independent per element,
            # so the span is order-free.
            return 2
        if phase.name.endswith(".fp4_routed_w13") or phase.name.endswith(".fp4_routed_w2"):
            # MEASURED EXCEPTION to the one-wave policy below (R1 leg 6, run 2).
            # The grouped FP4 bodies are the only bands whose per-item cost is
            # DATA-dependent: route_row(item) = item / 32, so a claim of
            # consecutive items sits inside one route row, and a route row is
            # either a chunk leader (which computes up to 8 members' outputs)
            # or a duplicate that early-outs to nothing. One claim per CTA
            # therefore hands whole CTAs a no-op while others carry a full
            # leader group. Measured at chunk 7 (138 claims): layer_0 w13
            # 86.2 -> 109.4 us, w2 53.1 -> 69.4 us, layer_2 w13 98.2 -> 111.7.
            # Keeping ~2 claims per worker preserves the dynamic balancing
            # these bimodal bands need.
            #
            # R11: the chunk is additionally rounded UP to a multiple of
            # _ROUTED_CLAIM_ALIGNMENT. The routed DG body fuses
            # DSPARK_W13_DG_TILE_PAIRS consecutive items into one wide-tile
            # pass, and a claim range must contain whole fused groups or the
            # pass would have to fall back to a narrow tile at the boundary.
            # begin is always a multiple of the chunk (the claim atomic adds
            # exactly claim_chunk) and every routed work-unit count is a
            # multiple of 32, so an aligned chunk makes every claim aligned.
            # The claim chunk decides only how many consecutive items one CTA
            # walks, so this is bitwise-neutral. Batch 1 and 8 are unchanged
            # (4 and 16); batch 2 moves 7 -> 8 and batch 4 moves 13 -> 14.
            chunk = min(
                16,
                max(
                    1,
                    (phase.work_units + 2 * shape.workers - 1) // (2 * shape.workers),
                ),
            )
            alignment = _ROUTED_CLAIM_ALIGNMENT
            chunk = ((chunk + alignment - 1) // alignment) * alignment
            return min(16 // alignment * alignment, chunk)
        # R1 leg 6 (intra-node retile): one-wave claim granularity.
        #
        # The retired policy aimed at TWO claims per worker, which puts every
        # phase whose item count exceeds the grid into a partially occupied
        # second wave: 1010 LM tiles at chunk 4 became 253 claims over 148
        # CTAs (wave 2 at 71% occupancy, +190 us of first-to-last claim
        # spread), 960 routed-expert tiles at chunk 4 became 240 claims (wave
        # 2 at 62%), and the 253-tile vocabulary bands became 253 one-tile
        # claims (wave 2 at 71%). Rounding the chunk UP to ceil(units /
        # workers) retires every one of those bands in a single wave with at
        # most one claim per CTA, so the span is the cost of ceil(units /
        # workers) tiles instead of two ragged waves.
        #
        # Bitwise: the claim chunk only decides how many consecutive items one
        # CTA walks in its `for (item = begin; item < end; ++item)` loop. Items
        # keep their identity, their decode, and their per-item reduction
        # order, so no accumulation is reassociated.
        return min(16, max(1, (phase.work_units + shape.workers - 1) // shape.workers))

    claim_chunks = [claim_chunk(phase) for phase in program]

    # AOT instruction streams, one per GB300 SM. This follows the fixed-SM
    # queues used by HazyResearch/Megakernels while retaining our existing
    # release/acquire phase events (the same worker-side contract Mirage uses).
    # An offline dependency/cost list scheduler assigns every claim to the SM
    # with the earliest modeled finish. A worker may reach a task early, but
    # it waits only on that phase's epoch; no whole-grid barrier, runtime
    # autotuner, or dynamic ready-queue scan is required.
    def gb300_claim_cost(phase, claims: int) -> int:
        """Fixed relative claim cost from the retained sm_103 device trace.

        These are scheduling weights, not latency claims. They AOT-select the
        per-SM instruction streams; no profiler, tuner, or shape search runs in
        production. Phase spans are divided by their unavoidable claim waves
        so an individual routed claim is not priced as two serial claims.
        """

        name = phase.name
        phase_cost = 8
        costs = (
            ("head.lm_row_", 242),
            (".fp4_routed_w13", 126),
            (".fp8_shared_w13", 78),
            (".fp4_routed_w2", 57),
            (".fp8_shared_w2", 65),
            (".shared_swiglu", 48),
            ("main.projected_activation_quant", 42),
            ("inverse_rope_grouped_wo_a", 40),
            ("main.projection", 36),
            ("attn_hc_sinkhorn", 30),
            ("ffn_hc_sinkhorn", 30),
            ("expert_combine_hc_post", 6),
            ("sparse_attention", 26),
            (".q_b", 26),
            (".wo_b", 23),
            ("head.hc_reduce_rmsnorm", 25),
            (".markov_w2", 18),
            (".q_a", 17),
            ("main_kv_projection", 14),
            ("draft_kv_projection", 14),
            ("embedding_hc_expand", 13),
            ("main_kv_norm_rope_quant_store", 10),
            ("draft_kv_norm_rope_quant", 10),
            ("router_top6", 10),
            ("router_sqrtsoftplus_top6", 10),
            (".markov_gather", 7),
            ("activation_quant", 7),
            ("split_k_reduce", 7),
            ("attn_hc_projection", 7),
            ("ffn_hc_projection", 7),
            ("routed_swiglu", 7),
        )
        for fragment, cost in costs:
            if fragment in name:
                phase_cost = cost
                break
        waves = (claims + shape.workers - 1) // shape.workers
        if gap_schedule == 2:
            # Rounded claim durations from the best7 sm_103 trace, plus the
            # observed ~2 us controller handoff. Whole-phase spans overprice
            # short work delayed by its old worker queue (embedding/SwiGLU).
            # Routed groups have runtime-dependent no-op claims: retain the
            # observed phase span divided by claim waves for those two bands.
            measured = (
                ("head.lm_row_", 160),
                (".fp4_routed_w13", 19),
                (".fp4_routed_w2", 13),
                (".fp8_shared_w13", 36),
                (".fp8_shared_w2", 18),
                (".shared_swiglu", 5),
                ("main.projected_activation_quant", 2),
                ("main.activation_quant", 2),
                ("main.projection", 19),
                ("main.split_k_reduce", 3),
                ("main.rmsnorm", 2),
                ("embedding_hc_expand", 2),
                ("inverse_rope_grouped_wo_a", 27),
                ("sparse_attention", 23),
                ("attn_hc_projection", 5),
                ("ffn_hc_projection", 5),
                ("attn_hc_sinkhorn", 6),
                ("ffn_hc_sinkhorn", 6),
                ("attn_hc_reduce_rmsnorm", 8),
                ("ffn_hc_reduce_rmsnorm", 8),
                ("head.hc_reduce_rmsnorm", 23),
                (".q_a_rmsnorm", 6),
                (".q_a", 14),
                (".q_b", 18),
                (".q_norm_rope", 3),
                (".wo_b", 18),
                ("main_kv_projection", 6),
                ("draft_kv_projection", 16),
                ("main_kv_norm_rope_quant_store", 9),
                ("draft_kv_norm_rope_quant", 8),
                ("router_sqrtsoftplus_top6", 7),
                ("router_top6", 5),
                ("routed_swiglu", 5),
                ("expert_combine_hc_post", 3),
                (".attn_hc_post", 3),
                (".markov_gather", 4),
                (".markov_w2", 11),
            )
            for fragment, cost in measured:
                if fragment in name:
                    return cost + 2
        return max(1, (phase_cost + waves - 1) // waves)

    # The fused routed consumer is split into two 256-feature half-bands for
    # each historical 512-feature work unit. That gives a one-to-one physical
    # consumer completion for every four-item routed-W13 claim while retaining
    # 240 producer claims (and therefore the grouped body's load balance).
    physical_work_units = [phase.work_units for phase in program]
    for _, consumer in streaming_edges:
        physical_work_units[consumer] *= 2
    claim_counts = [
        (physical_work_units[phase.phase_id] + claim_chunks[phase.phase_id] - 1)
        // claim_chunks[phase.phase_id]
        for phase in program
    ]
    claim_costs = [gb300_claim_cost(phase, claim_counts[phase.phase_id]) for phase in program]

    # Routed W13 -> SwiGLU is lowered as per-claim vertical fusion. A four-item
    # W13 claim becomes two discontiguous two-item bands (matching gate/up),
    # followed immediately by one 256-feature SwiGLU half-band in the same
    # CTA. No global event is needed: the CTA barrier before finish publishes
    # both semantic completions. Price that fused epilogue on the producer.
    streaming_consumer_for_phase = [0xFFFF] * len(program)
    if fine_overlap:
        for producer, consumer in sorted(streaming_edges):
            if claim_chunks[producer] != 4 or claim_counts[producer] != 240:
                raise ValueError("routed-W13 fine overlap requires 240 four-item claims")
            if claim_chunks[consumer] != 1 or claim_counts[consumer] != 240:
                raise ValueError("fused routed-SwiGLU requires 240 half-band completions")
            streaming_consumer_for_phase[producer] = consumer
            claim_costs[producer] += claim_costs[consumer]

    # Upward rank gives independent branches with a long path to the terminal
    # phase priority over short side work. Positive costs make this a strict
    # topological order: every predecessor ranks above every successor.
    upward_rank = [0] * len(program)
    for phase in reversed(program):
        span = claim_costs[phase.phase_id] * (
            (claim_counts[phase.phase_id] + shape.workers - 1) // shape.workers
        )
        upward_rank[phase.phase_id] = span + max(
            (upward_rank[successor] for successor in semantic_successors[phase.phase_id]),
            default=0,
        )
    phase_order = sorted(
        range(len(program)), key=lambda phase_id: (-upward_rank[phase_id], phase_id)
    )
    if shared_first:
        # The shared W13 branch is ready at the FFN input, before routing.
        # The original append-only list scheduler reserves every routed slot
        # first, leaving ready shared work behind future routed work. Place
        # shared W13 before the router so its workers start immediately;
        # remaining workers can route and claim expert tiles concurrently.
        for layer in range(3):
            shared = by_name[f"layer_{layer}.fp8_shared_w13"]
            router = by_name[f"layer_{layer}.router_sqrtsoftplus_top6"]
            if phase_order.index(shared) > phase_order.index(router):
                phase_order.remove(shared)
                phase_order.insert(phase_order.index(router), shared)
    if hc_split:
        # Both tasks depend on the mix projection. Reserve their workers
        # together, before reserving any downstream GEMM work.
        for layer in range(3):
            for band in ("attn", "ffn"):
                norm = by_name[f"layer_{layer}.{band}_hc_reduce_rmsnorm"]
                sinkhorn = by_name[f"layer_{layer}.{band}_hc_sinkhorn"]
                phase_order.remove(sinkhorn)
                phase_order.insert(phase_order.index(norm) + 1, sinkhorn)
    if routed_group_ready:
        # Every worker must drain its W13 claim slot before it may wait in W2.
        # Dynamic W13 claiming exhausts all 240 tickets before any worker exits
        # that slot, so all 120 active producer streams are resident by then.
        for layer in range(3):
            producer = by_name[f"layer_{layer}.fp4_routed_w13"]
            consumer = by_name[f"layer_{layer}.fp4_routed_w2"]
            phase_order.remove(producer)
            phase_order.insert(phase_order.index(consumer), producer)
    if front_embed_first:
        # Main-KV can overlap the Q path. Reserve the independent embedding/HC
        # tasks first so their workers do not wait behind the main projection.
        projection = by_name["main.projection"]
        for name in ("draft.embedding_hc_expand", "layer_0.attn_hc_projection"):
            phase = by_name[name]
            phase_order.remove(phase)
            phase_order.insert(phase_order.index(projection), phase)
    order_index = {phase_id: index for index, phase_id in enumerate(phase_order)}
    for phase in program:
        if any(
            order_index[by_name[name]] >= order_index[phase.phase_id] for name in phase.dependencies
        ):
            raise ValueError(
                f"GB300 static queue order is not topological at {phase.name}: {phase.dependencies}"
            )

    worker_tasks: list[list[tuple[int, int]]] = [[] for _ in range(shape.workers)]
    worker_intervals: list[list[tuple[int, int]]] = [[] for _ in range(shape.workers)]
    worker_available = [0] * shape.workers
    phase_completion = [0] * len(program)
    fused_producer_for_consumer = {consumer: producer for producer, consumer in streaming_edges}
    for phase_id in phase_order:
        phase = program[phase_id]
        if phase_id in fused_producer_for_consumer:
            producer = fused_producer_for_consumer[phase_id]
            phase_completion[phase_id] = phase_completion[producer]
            continue
        phase_release = max(
            (phase_completion[by_name[name]] for name in phase.dependencies),
            default=0,
        )
        cost = claim_costs[phase_id]
        completion = phase_release
        eligible_workers = range(shape.workers)
        if routed_interleave and phase.name.endswith(".fp8_shared_w13"):
            # Keep 128 workers available for the routed producer's cyclic streams.
            eligible_workers = sorted(
                range(shape.workers), key=lambda worker: (worker_available[worker], worker)
            )[:24]
        for claim in range(claim_counts[phase_id]):
            if gap_schedule:
                choices = []
                for candidate in range(shape.workers):
                    start = phase_release
                    slot = 0
                    for occupied_start, occupied_end in worker_intervals[candidate]:
                        if start + cost <= occupied_start:
                            break
                        start = max(start, occupied_end)
                        slot += 1
                    choices.append((start + cost, candidate, slot))
                finish, worker, slot = min(choices)
                worker_intervals[worker].insert(slot, (finish - cost, finish))
                worker_tasks[worker].insert(slot, (phase_id, claim))
            else:
                finish, worker = min(
                    (
                        max(worker_available[candidate], phase_release) + cost,
                        candidate,
                    )
                    for candidate in eligible_workers
                )
                worker_tasks[worker].append((phase_id, claim))
                worker_available[worker] = finish
            completion = max(completion, finish)
        phase_completion[phase_id] = completion
    if shared_continue:
        # Preserve all routed/router assignments and their original prefixes.
        # The 32 shared producer CTAs have no routing tasks; the other 120 CTAs
        # remain available for all 120 compact routed streams. Move only shared
        # consumers, using the existing GPU publication/dependency events.
        for layer in range(3):
            producer = by_name[f"layer_{layer}.fp8_shared_w13"]
            swiglu = by_name[f"layer_{layer}.shared_swiglu"]
            w2 = by_name[f"layer_{layer}.fp8_shared_w2"]
            routing = {
                by_name[f"layer_{layer}.router_sqrtsoftplus_top6"],
                by_name[f"layer_{layer}.router_top6"],
            }
            owners = sorted(
                (claim, worker)
                for worker, tasks in enumerate(worker_tasks)
                for phase, claim in tasks
                if phase == producer
            )
            if len(owners) != 32 or len({w for _, w in owners}) != 32:
                raise ValueError("shared continuation requires 32 distinct producer CTAs")
            if any(p in routing for _, w in owners for p, _ in worker_tasks[w]):
                raise ValueError("shared producer CTAs unexpectedly own routing work")
            if claim_counts[swiglu] != 20 or claim_counts[w2] != 32:
                raise ValueError("unexpected shared continuation claim geometry")
            for tasks in worker_tasks:
                tasks[:] = [(p, c) for p, c in tasks if p not in (swiglu, w2)]
            for index, (_, worker) in enumerate(owners):
                tasks = worker_tasks[worker]
                slot = next(i for i, (p, _) in enumerate(tasks) if p == producer) + 1
                continuation = ([(swiglu, index)] if index < 20 else []) + [(w2, index)]
                tasks[slot:slot] = continuation
    if routed_w2_workers128:
        # Post-process only W2 admission/physical credits. The original cost
        # schedule and every non-W2 task assignment remain unchanged.
        for layer in range(3):
            w2 = by_name[f"layer_{layer}.fp4_routed_w2"]
            old_owners = {
                w for w, tasks in enumerate(worker_tasks) if any(p == w2 for p, _ in tasks)
            }
            added = set(range(128)) - old_owners
            if old_owners - set(range(128)) or added != {5, 6, 7, 8, 9, 125, 126, 127}:
                raise ValueError("unexpected W2 baseline owners; refusing to move KV workers")
            local_shared = {
                by_name[f"layer_{layer}.{name}"]
                for name in ("fp8_shared_w13", "shared_swiglu", "fp8_shared_w2")
            }
            for worker, tasks in enumerate(worker_tasks):
                if worker in old_owners:
                    slot = next(i for i, (p, _) in enumerate(tasks) if p == w2)
                    tasks[:] = [(p, c) for p, c in tasks if p != w2]
                    tasks.insert(slot, (w2, worker))
                elif worker in added:
                    slot = next(
                        (
                            i
                            for i, (p, _) in enumerate(tasks)
                            if order_index[p] > order_index[w2] and p not in local_shared
                        ),
                        len(tasks),
                    )
                    tasks.insert(slot, (w2, worker))
            physical_work_units[w2] = 512
            claim_chunks[w2] = 4
            claim_counts[w2] = 128
    if routed_w2_workers128:
        # Layer-0 KV finalization depends only on the KV projection. Its five
        # baseline slots follow the independent Q chain on workers 0..4.
        # Continue on five KV producer workers, as later layers already do,
        # preserving every phase edge, claim index and completion credit.
        producer = by_name["layer_0.draft_kv_projection"]
        finalize = by_name["layer_0.draft_kv_norm_rope_quant"]
        owners = sorted(
            (worker, claim)
            for worker, tasks in enumerate(worker_tasks)
            for phase, claim in tasks
            if phase == finalize
        )
        if batch != 1 or producer != 18 or finalize != 20 or owners != [(i, i) for i in range(5)]:
            raise ValueError("unexpected layer-0 KV-finalize baseline geometry")
        if claim_counts[finalize] != 5 or claim_chunks[finalize] != 1:
            raise ValueError("KV-finalize continuation requires five one-row claims")
        for tasks in worker_tasks:
            tasks[:] = [(phase, claim) for phase, claim in tasks if phase != finalize]
        for claim, worker in enumerate(range(128, 133)):
            tasks = worker_tasks[worker]
            slots = [i for i, (phase, _) in enumerate(tasks) if phase == producer]
            if len(slots) != 1:
                raise ValueError("KV-finalize owner must have one preceding KV claim")
            tasks.insert(slots[0] + 1, (finalize, claim))

    worker_task_offsets = [0]
    worker_task_phases: list[int] = []
    worker_task_claims: list[int] = []
    for tasks in worker_tasks:
        worker_task_phases.extend(phase for phase, _ in tasks)
        worker_task_claims.extend(claim for _, claim in tasks)
        worker_task_offsets.append(len(worker_task_phases))

    role_masks = [
        sum(1 << int(role) for role in phase.roles) | (1 << int(V4WorkerRole.CONTROLLER))
        for phase in program
    ]
    table = V4WeightTableLayout.flash(spec)
    workspace = {region.name: region for region in abi.workspace.regions}

    def offset_constant(name: str, region: str) -> str:
        return f"inline constexpr uint64_t {name} = {workspace[region].offset_bytes}ULL;"

    def int_constant(name: str, value: int) -> str:
        return f"inline constexpr int {name} = {value};"

    lines = [
        "// Generated by scripts/megakernel/generate_v4_header.py. Do not edit.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace deepspec::v4_generated {",
        *(
            [
                "inline constexpr int kRoutedW2Workers = 128;",
                "inline constexpr int kRoutedW2Credits = 512;",
            ]
            if routed_w2_workers128
            else []
        ),
        f"inline constexpr int kPhaseCount = {len(program)};",
        f"inline constexpr int kRootCount = {len(roots)};",
        f"inline constexpr int kStaticTaskCount = {len(worker_task_phases)};",
        *(["inline constexpr uint16_t kNoStreamingConsumer = 0xFFFF;"] if fine_overlap else []),
        f"inline constexpr int kWorkers = {shape.workers};",
        "inline constexpr int kThreads = 256;",
        f"inline constexpr int kLocalQueueDepth = {shape.local_queue_depth};",
        f"inline constexpr int kTraceCapacity = {shape.trace_events_per_worker};",
        "inline constexpr int kTraceColumns = 8;",
        *(
            f"inline constexpr uint8_t kExecution{name.title().replace('_', '')} = {index};"
            for index, name in enumerate(execution_group_names)
        ),
        f"inline constexpr uint64_t kWorkspaceBytes = {abi.workspace.total_bytes}ULL;",
        offset_constant("kDependencyOffset", "phase_dependency_arrivals"),
        *(
            [
                int_constant(
                    "kRoutedGroupReadyOffset",
                    abi.workspace.region("phase_dependency_arrivals").offset_bytes
                    + len(program) * 8,
                )
            ]
            if routed_group_ready
            else []
        ),
        offset_constant("kNextTileOffset", "phase_next_tile"),
        offset_constant("kCompletedTileOffset", "phase_completed_tiles"),
        offset_constant("kPublishEpochOffset", "phase_publish_epoch"),
        offset_constant("kReadyQueueOffset", "local_ready_queues"),
        offset_constant("kQueueStateOffset", "local_queue_state"),
        offset_constant("kWorkerStateOffset", "worker_state"),
        f"inline constexpr int kWeightOffsetSlots = {table.total_slots};",
        f"inline constexpr int kGlobalWeightBase = {table.global_base};",
        f"inline constexpr int kLayerWeightBase = {table.layer_base};",
        f"inline constexpr int kRoutedExpertWeightBase = {table.routed_expert_base};",
        f"inline constexpr int kSharedExpertWeightBase = {table.shared_expert_base};",
        f"inline constexpr int kGlobalWeightSlots = {len(V4GlobalWeightSlot)};",
        f"inline constexpr int kLayerWeightSlots = {len(V4LayerWeightSlot)};",
        f"inline constexpr int kExpertWeightSlots = {len(V4ExpertWeightSlot)};",
        # R12: the chain geometry the router work-unit count was derived from.
        # Relaxed-only -- the contract header stays byte-identical, and
        # dspark_v4_kernel.cu static_asserts these against the compile-time
        # DSPARK_ROUTER_CHAINS / DSPARK_ROUTER_ROWS so a stale header cannot
        # silently drop chains.
        *(
            [
                f"inline constexpr int kRouterChainsPerItem = {router_chains};",
                f"inline constexpr int kRouterRowsPerChain = {router_rows};",
            ]
            if relaxed
            else []
        ),
        f"inline constexpr unsigned long long kWeightArenaBytes = "
        f"{plan_v4_weight_arena(DeepSeekV4MegaKernelSpec()).arena_bytes}ULL;",
        f"inline constexpr int kMainQuantPhase = {by_name['main.activation_quant']};",
        f"inline constexpr int kMainProjectionPhase = {by_name['main.projection']};",
        f"inline constexpr int kMainReducePhase = {by_name['main.split_k_reduce']};",
        f"inline constexpr int kMainNormPhase = {by_name['main.rmsnorm']};",
        f"inline constexpr int kEmbeddingPhase = {by_name['draft.embedding_hc_expand']};",
        int_constant(
            "kMainProjectedQuantPhase",
            by_name["main.projected_activation_quant"],
        ),
        int_constant(
            "kLayer0AttnHcProjectionPhase",
            by_name["layer_0.attn_hc_projection"],
        ),
        int_constant(
            "kLayer0AttnHcSinkhornPhase",
            by_name["layer_0.attn_hc_sinkhorn"],
        ),
        int_constant(
            "kLayer0AttnHcReduceNormPhase",
            tail_phase("layer_0.attn_hc_reduce_rmsnorm"),
        ),
        int_constant("kLayer0QaPhase", by_name["layer_0.q_a"]),
        int_constant("kLayer0QaNormPhase", by_name["layer_0.q_a_rmsnorm"]),
        int_constant("kLayer0QbPhase", by_name["layer_0.q_b"]),
        int_constant(
            "kLayer0DraftKvProjectionPhase",
            by_name["layer_0.draft_kv_projection"],
        ),
        int_constant("kLayer0QueryNormRopePhase", by_name["layer_0.q_norm_rope"]),
        int_constant(
            "kLayer0DraftKvFinalizePhase",
            by_name["layer_0.draft_kv_norm_rope_quant"],
        ),
        int_constant(
            "kLayer0SparseAttentionPhase",
            by_name["layer_0.sparse_attention"],
        ),
        int_constant(
            "kLayer0OutputAProjectionPhase",
            by_name["layer_0.inverse_rope_grouped_wo_a"],
        ),
        int_constant("kLayer0OutputBProjectionPhase", by_name["layer_0.wo_b"]),
        int_constant("kLayer0AttnPostPhase", by_name["layer_0.attn_hc_post"]),
        int_constant(
            "kLayer0FfnHcProjectionPhase",
            by_name["layer_0.ffn_hc_projection"],
        ),
        int_constant(
            "kLayer0FfnHcSinkhornPhase",
            by_name["layer_0.ffn_hc_sinkhorn"],
        ),
        int_constant(
            "kLayer0FfnHcReduceNormPhase",
            tail_phase("layer_0.ffn_hc_reduce_rmsnorm"),
        ),
        int_constant(
            "kLayer0RouterScorePhase",
            by_name["layer_0.router_sqrtsoftplus_top6"],
        ),
        int_constant("kLayer0RouterTop6Phase", by_name["layer_0.router_top6"]),
        int_constant("kLayer0RoutedW13Phase", by_name["layer_0.fp4_routed_w13"]),
        int_constant("kLayer0RoutedSwiGluPhase", by_name.get("layer_0.routed_swiglu", 0xFFFF)),
        int_constant("kLayer0RoutedW2Phase", by_name["layer_0.fp4_routed_w2"]),
        int_constant("kLayer0SharedW13Phase", by_name["layer_0.fp8_shared_w13"]),
        int_constant("kLayer0SharedSwiGluPhase", by_name["layer_0.shared_swiglu"]),
        int_constant("kLayer0SharedW2Phase", by_name["layer_0.fp8_shared_w2"]),
        int_constant(
            "kLayer0ExpertCombinePhase",
            by_name["layer_0.expert_combine_hc_post"],
        ),
        int_constant("kHeadReducePhase", by_name["head.hc_reduce_rmsnorm"]),
        int_constant(
            "kPrefixSchedulerPhase",
            tail_phase("prefix.causal_algorithm_1"),
        ),
        int_constant("kExpertW1Slot", V4ExpertWeightSlot.W1),
        int_constant("kExpertW1ScaleSlot", V4ExpertWeightSlot.W1_SCALE),
        int_constant("kExpertW2Slot", V4ExpertWeightSlot.W2),
        int_constant("kExpertW2ScaleSlot", V4ExpertWeightSlot.W2_SCALE),
        int_constant("kExpertW3Slot", V4ExpertWeightSlot.W3),
        int_constant("kExpertW3ScaleSlot", V4ExpertWeightSlot.W3_SCALE),
        int_constant(
            "kMainProjWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.MAIN_PROJ),
        ),
        int_constant(
            "kEmbeddingWeightSlot",
            table.global_slot(V4GlobalWeightSlot.EMBEDDING),
        ),
        int_constant(
            "kMainProjScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.MAIN_PROJ_SCALE),
        ),
        int_constant(
            "kMainNormWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.MAIN_NORM),
        ),
        int_constant(
            "kLayer0AttnHcFnSlot",
            table.layer_slot(0, V4LayerWeightSlot.HC_ATTN_FN),
        ),
        int_constant(
            "kLayer0AttnHcBaseSlot",
            table.layer_slot(0, V4LayerWeightSlot.HC_ATTN_BASE),
        ),
        int_constant(
            "kLayer0AttnHcScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.HC_ATTN_SCALE),
        ),
        int_constant(
            "kLayer0AttnNormWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.ATTN_NORM),
        ),
        int_constant(
            "kLayer0WqaWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.WQ_A),
        ),
        int_constant(
            "kLayer0WqaScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.WQ_A_SCALE),
        ),
        int_constant(
            "kLayer0QaNormWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.Q_NORM),
        ),
        int_constant(
            "kLayer0WqbWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.WQ_B),
        ),
        int_constant(
            "kLayer0WqbScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.WQ_B_SCALE),
        ),
        int_constant(
            "kLayer0WkvWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.WKV),
        ),
        int_constant(
            "kLayer0WkvScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.WKV_SCALE),
        ),
        int_constant(
            "kLayer0KvNormWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.KV_NORM),
        ),
        int_constant(
            "kLayer0AttnSinkWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.ATTN_SINK),
        ),
        int_constant(
            "kLayer0WoaWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.WO_A),
        ),
        int_constant(
            "kLayer0WobWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.WO_B),
        ),
        int_constant(
            "kLayer0WobScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.WO_B_SCALE),
        ),
        int_constant(
            "kLayer0FfnHcFnSlot",
            table.layer_slot(0, V4LayerWeightSlot.HC_FFN_FN),
        ),
        int_constant(
            "kLayer0FfnHcBaseSlot",
            table.layer_slot(0, V4LayerWeightSlot.HC_FFN_BASE),
        ),
        int_constant(
            "kLayer0FfnHcScaleSlot",
            table.layer_slot(0, V4LayerWeightSlot.HC_FFN_SCALE),
        ),
        int_constant(
            "kLayer0FfnNormWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.FFN_NORM),
        ),
        int_constant(
            "kLayer0RouterWeightSlot",
            table.layer_slot(0, V4LayerWeightSlot.GATE),
        ),
        int_constant(
            "kLayer0RouterBiasSlot",
            table.layer_slot(0, V4LayerWeightSlot.GATE_BIAS),
        ),
        int_constant(
            "kFinalNormWeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.FINAL_NORM),
        ),
        int_constant(
            "kHcHeadFnWeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.HC_HEAD_FN),
        ),
        int_constant(
            "kHcHeadBaseWeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.HC_HEAD_BASE),
        ),
        int_constant(
            "kHcHeadScaleWeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.HC_HEAD_SCALE),
        ),
        int_constant(
            "kLmHeadWeightSlot",
            table.global_slot(V4GlobalWeightSlot.LM_HEAD),
        ),
        int_constant(
            "kMarkovW1WeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.MARKOV_W1),
        ),
        int_constant(
            "kMarkovW2WeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.MARKOV_W2),
        ),
        int_constant(
            "kConfidenceWeightSlot",
            table.layer_slot(2, V4LayerWeightSlot.CONFIDENCE),
        ),
        offset_constant("kMainQuantizedOffset", "main_hidden_quantized"),
        offset_constant("kMainScaleOffset", "main_hidden_scales"),
        offset_constant("kMainPartialOffset", "main_projection_partials"),
        offset_constant("kMainFp32Offset", "main_projection_fp32"),
        offset_constant("kMainRmsPartialOffset", "main_rms_partial"),
        offset_constant("kMainProjectedOffset", "main_projected"),
        offset_constant("kMainProjectedQuantizedOffset", "main_projected_quantized"),
        offset_constant("kMainProjectedScaleOffset", "main_projected_scales"),
        offset_constant("kMainKvPartialOffset", "main_kv_partials"),
        offset_constant("kHiddenStreamsOffset", "hidden_streams"),
        offset_constant("kHcMixOffset", "hc_mixes"),
        offset_constant("kHcPreOffset", "hc_pre"),
        offset_constant("kHcPostOffset", "hc_post"),
        offset_constant("kHcCombOffset", "hc_comb"),
        offset_constant("kNormalizedHiddenOffset", "normalized_hidden"),
        offset_constant("kAttnInputQuantizedOffset", "attn_input_quantized"),
        offset_constant("kAttnInputScaleOffset", "attn_input_scales"),
        offset_constant("kQaPartialOffset", "q_a_partials"),
        offset_constant("kQloraOffset", "q_lora"),
        offset_constant("kQloraQuantizedOffset", "q_lora_quantized"),
        offset_constant("kQloraScaleOffset", "q_lora_scales"),
        offset_constant("kQueryProjectionOffset", "query_projection"),
        offset_constant("kQueryInverseRmsOffset", "query_inverse_rms"),
        offset_constant("kQueriesOffset", "queries"),
        offset_constant("kDraftKvPartialOffset", "draft_kv_partials"),
        offset_constant("kDraftKvOffset", "draft_kv"),
        offset_constant("kAttentionAccumulatorOffset", "attention_accumulator"),
        offset_constant("kAttentionRawOffset", "attention_raw"),
        offset_constant("kAttentionValuesOffset", "attention_values"),
        offset_constant("kOutputLoraOffset", "output_lora"),
        offset_constant("kOutputLoraQuantizedOffset", "output_lora_quantized"),
        offset_constant("kOutputLoraScaleOffset", "output_lora_scales"),
        offset_constant("kAttentionOutputPartialOffset", "attention_output_partials"),
        offset_constant("kAttentionOutputOffset", "attention_output"),
        offset_constant("kAttentionHiddenStreamsOffset", "attention_hidden_streams"),
        offset_constant("kFfnHcMixOffset", "ffn_hc_mixes"),
        offset_constant("kFfnHcPreOffset", "ffn_hc_pre"),
        offset_constant("kFfnHcPostOffset", "ffn_hc_post"),
        offset_constant("kFfnHcCombOffset", "ffn_hc_comb"),
        offset_constant("kFfnNormalizedHiddenOffset", "ffn_normalized_hidden"),
        offset_constant("kFfnInputQuantizedOffset", "ffn_input_quantized"),
        offset_constant("kFfnInputScaleOffset", "ffn_input_scales"),
        offset_constant("kRouterScoreOffset", "router_scores"),
        offset_constant("kRouterIndexOffset", "router_indices"),
        offset_constant("kRouterWeightOffset", "router_weights"),
        offset_constant("kRoutedW13Offset", "routed_w13"),
        offset_constant("kSharedW13Offset", "shared_w13"),
        offset_constant("kRoutedSwiGluOffset", "routed_swiglu"),
        offset_constant("kRoutedSwiGluQuantizedOffset", "routed_swiglu_quantized"),
        offset_constant("kRoutedSwiGluScaleOffset", "routed_swiglu_scales"),
        offset_constant("kSharedSwiGluOffset", "shared_swiglu"),
        offset_constant("kSharedSwiGluQuantizedOffset", "shared_swiglu_quantized"),
        offset_constant("kSharedSwiGluScaleOffset", "shared_swiglu_scales"),
        offset_constant("kRoutedOutputPartialOffset", "routed_output_partials"),
        offset_constant("kRoutedOutputOffset", "routed_output"),
        offset_constant("kSharedOutputOffset", "shared_output"),
        offset_constant("kHeadHiddenOffset", "head_hidden"),
        offset_constant("kHeadNormalizedOffset", "head_normalized"),
        offset_constant("kBaseLogitsOffset", "base_logits"),
        offset_constant("kMarkovEmbeddingOffset", "markov_embeddings"),
        offset_constant("kMarkovLogitsOffset", "markov_logits_row"),
        offset_constant("kSoftmaxPartialMaxOffset", "softmax_partial_max"),
        offset_constant("kSoftmaxPartialSumOffset", "softmax_partial_sum"),
        offset_constant("kSampleScanOffset", "sample_scan"),
        "",
        _array(
            "kMainKvProjectionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.main_kv_projection"] for layer in range(3)],
        ),
        "",
        _array(
            "kMainKvNormPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.main_kv_norm_rope_quant_store"] for layer in range(3)],
        ),
        "",
        _array(
            "kMainKvWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WKV) for layer in range(3)],
        ),
        "",
        _array(
            "kMainKvScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WKV_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kMainKvNormWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.KV_NORM) for layer in range(3)],
        ),
        "",
        _array(
            "kAttnHcProjectionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.attn_hc_projection"] for layer in range(3)],
        ),
        "",
        _array(
            "kAttnHcSinkhornPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.attn_hc_sinkhorn"] for layer in range(3)],
        ),
        "",
        _array(
            "kAttnHcReduceNormPhases",
            "uint16_t",
            [tail_phase(f"layer_{layer}.attn_hc_reduce_rmsnorm") for layer in range(3)],
        ),
        "",
        _array(
            "kQaPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.q_a"] for layer in range(3)],
        ),
        "",
        _array(
            "kQaNormPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.q_a_rmsnorm"] for layer in range(3)],
        ),
        "",
        _array(
            "kQbPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.q_b"] for layer in range(3)],
        ),
        "",
        _array(
            "kDraftKvProjectionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.draft_kv_projection"] for layer in range(3)],
        ),
        "",
        _array(
            "kQueryNormRopePhases",
            "uint16_t",
            [by_name[f"layer_{layer}.q_norm_rope"] for layer in range(3)],
        ),
        "",
        _array(
            "kDraftKvFinalizePhases",
            "uint16_t",
            [by_name[f"layer_{layer}.draft_kv_norm_rope_quant"] for layer in range(3)],
        ),
        "",
        _array(
            "kSparseAttentionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.sparse_attention"] for layer in range(3)],
        ),
        "",
        _array(
            "kOutputAProjectionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.inverse_rope_grouped_wo_a"] for layer in range(3)],
        ),
        "",
        *(
            [
                _array(
                    "kOutputAReductionPhases",
                    "uint16_t",
                    [by_name[f"layer_{layer}.wo_a_reduce_quant"] for layer in range(3)],
                ),
                "",
            ]
            if woa_split_k != 1
            else []
        ),
        _array(
            "kOutputBProjectionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.wo_b"] for layer in range(3)],
        ),
        "",
        _array(
            "kAttnPostPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.attn_hc_post"] for layer in range(3)],
        ),
        "",
        _array(
            "kFfnHcProjectionPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.ffn_hc_projection"] for layer in range(3)],
        ),
        "",
        _array(
            "kFfnHcSinkhornPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.ffn_hc_sinkhorn"] for layer in range(3)],
        ),
        "",
        _array(
            "kFfnHcReduceNormPhases",
            "uint16_t",
            [tail_phase(f"layer_{layer}.ffn_hc_reduce_rmsnorm") for layer in range(3)],
        ),
        "",
        _array(
            "kRouterScorePhases",
            "uint16_t",
            [by_name[f"layer_{layer}.router_sqrtsoftplus_top6"] for layer in range(3)],
        ),
        "",
        _array(
            "kRouterTop6Phases",
            "uint16_t",
            [by_name[f"layer_{layer}.router_top6"] for layer in range(3)],
        ),
        "",
        _array(
            "kRoutedW13Phases",
            "uint16_t",
            [by_name[f"layer_{layer}.fp4_routed_w13"] for layer in range(3)],
        ),
        "",
        _array(
            "kRoutedSwiGluPhases",
            "uint16_t",
            [by_name.get(f"layer_{layer}.routed_swiglu", 0xFFFF) for layer in range(3)],
        ),
        "",
        _array(
            "kRoutedW2Phases",
            "uint16_t",
            [by_name[f"layer_{layer}.fp4_routed_w2"] for layer in range(3)],
        ),
        "",
        _array(
            "kSharedW13Phases",
            "uint16_t",
            [by_name[f"layer_{layer}.fp8_shared_w13"] for layer in range(3)],
        ),
        "",
        _array(
            "kSharedSwiGluPhases",
            "uint16_t",
            [by_name[f"layer_{layer}.shared_swiglu"] for layer in range(3)],
        ),
        "",
        _array(
            "kSharedW2Phases",
            "uint16_t",
            [by_name[f"layer_{layer}.fp8_shared_w2"] for layer in range(3)],
        ),
        "",
        _array(
            "kExpertCombinePhases",
            "uint16_t",
            [by_name[f"layer_{layer}.expert_combine_hc_post"] for layer in range(3)],
        ),
        "",
        _array(
            "kLmRowPhases",
            "uint16_t",
            [
                by_name.get(f"head.lm_row_{step}", by_name["head.lm_row_0"])
                for step in range(spec.block_size)
            ],
        ),
        "",
        _array(
            "kMarkovGatherPhases",
            "uint16_t",
            [by_name[f"tail_{step}.markov_gather"] for step in range(spec.block_size)],
        ),
        "",
        _array(
            "kConfidencePhases",
            "uint16_t",
            [tail_phase(f"tail_{step}.confidence_sts") for step in range(spec.block_size)],
        ),
        "",
        _array(
            "kMarkovW2Phases",
            "uint16_t",
            [by_name[f"tail_{step}.markov_w2"] for step in range(spec.block_size)],
        ),
        "",
        _array(
            "kCorrectLogitPhases",
            "uint16_t",
            [
                tail_phase(f"tail_{step}.correct_logits_partial_max")
                for step in range(spec.block_size)
            ],
        ),
        "",
        _array(
            "kSoftmaxExpSumPhases",
            "uint16_t",
            [tail_phase(f"tail_{step}.softmax_exp_sum") for step in range(spec.block_size)],
        ),
        "",
        _array(
            "kSumReducePhases",
            "uint16_t",
            [tail_phase(f"tail_{step}.sum_reduce") for step in range(spec.block_size)],
        ),
        "",
        _array(
            "kNormalizeSamplePhases",
            "uint16_t",
            [tail_phase(f"tail_{step}.normalize_scan_sample") for step in range(spec.block_size)],
        ),
        "",
        _array(
            "kAttnHcFnSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.HC_ATTN_FN) for layer in range(3)],
        ),
        "",
        _array(
            "kAttnHcBaseSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.HC_ATTN_BASE) for layer in range(3)],
        ),
        "",
        _array(
            "kAttnHcScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.HC_ATTN_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kAttnNormWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.ATTN_NORM) for layer in range(3)],
        ),
        "",
        _array(
            "kWqaWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WQ_A) for layer in range(3)],
        ),
        "",
        _array(
            "kWqaScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WQ_A_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kQaNormWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.Q_NORM) for layer in range(3)],
        ),
        "",
        _array(
            "kWqbWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WQ_B) for layer in range(3)],
        ),
        "",
        _array(
            "kWqbScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WQ_B_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kDraftWkvWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WKV) for layer in range(3)],
        ),
        "",
        _array(
            "kDraftWkvScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WKV_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kDraftKvNormWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.KV_NORM) for layer in range(3)],
        ),
        "",
        _array(
            "kAttnSinkWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.ATTN_SINK) for layer in range(3)],
        ),
        "",
        _array(
            "kWoaWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WO_A) for layer in range(3)],
        ),
        "",
        _array(
            "kWobWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WO_B) for layer in range(3)],
        ),
        "",
        _array(
            "kWobScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.WO_B_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kFfnHcFnSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.HC_FFN_FN) for layer in range(3)],
        ),
        "",
        _array(
            "kFfnHcBaseSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.HC_FFN_BASE) for layer in range(3)],
        ),
        "",
        _array(
            "kFfnHcScaleSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.HC_FFN_SCALE) for layer in range(3)],
        ),
        "",
        _array(
            "kFfnNormWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.FFN_NORM) for layer in range(3)],
        ),
        "",
        _array(
            "kRouterWeightSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.GATE) for layer in range(3)],
        ),
        "",
        _array(
            "kRouterBiasSlots",
            "uint16_t",
            [table.layer_slot(layer, V4LayerWeightSlot.GATE_BIAS) for layer in range(3)],
        ),
        "",
        _array(
            "kRoutedExpertWeightBases",
            "uint16_t",
            [
                table.routed_expert_slot(
                    layer,
                    0,
                    V4ExpertWeightSlot.W1,
                    experts=spec.num_routed_experts,
                )
                for layer in range(3)
            ],
        ),
        "",
        _array(
            "kSharedExpertWeightBases",
            "uint16_t",
            [table.shared_expert_slot(layer, V4ExpertWeightSlot.W1) for layer in range(3)],
        ),
        "",
        _array("kRoots", "uint16_t", roots),
        "",
        _array("kWorkUnits", "uint32_t", physical_work_units),
        "",
        _array("kClaimChunks", "uint8_t", claim_chunks),
        "",
        _array("kExecutionGroups", "uint8_t", execution_groups),
        "",
        _array("kPhaseLayers", "uint8_t", phase_layers),
        "",
        _array("kStaticWorkerTaskOffsets", "uint16_t", worker_task_offsets),
        "",
        _array("kStaticTaskPhases", "uint16_t", worker_task_phases),
        "",
        _array("kStaticTaskClaims", "uint16_t", worker_task_claims),
        "",
        *(
            [
                _array(
                    "kStreamingConsumerForPhase",
                    "uint16_t",
                    streaming_consumer_for_phase,
                ),
                "",
            ]
            if fine_overlap
            else []
        ),
        _array(
            "kDependencyCounts",
            "uint8_t",
            normal_dependency_counts,
        ),
        "",
        _array("kRoleMasks", "uint8_t", role_masks),
        "",
        _array("kSuccessorOffsets", "uint16_t", successor_offsets),
        "",
        _array("kSuccessors", "uint16_t", successor_values),
        "",
        *(
            [
                _array("kPredecessorOffsets", "uint16_t", predecessor_offsets),
                "",
                _array("kPredecessors", "uint16_t", predecessor_values),
                "",
            ]
            if hc_split
            else []
        ),
        "}  // namespace deepspec::v4_generated",
        "",
    ]
    return "\n".join(lines)


def header_name(
    *,
    relaxed: bool,
    greedy: bool,
    batch: int = 1,
    fine_overlap: bool = False,
    shared_first: bool = False,
    hc_split: bool = False,
    gap_schedule: int = 0,
    woa_split_k: int = 1,
    w13_fused_swiglu: bool = False,
    routed_group_ready: bool = False,
    routed_interleave: bool = False,
    front_embed_first: bool = False,
    shared_continue: bool = False,
    routed_w2_workers128: bool = False,
) -> str:
    if routed_w2_workers128:
        return "dspark_v4_generated_greedy_shared_continue_w2_128.h"
    if shared_continue:
        return "dspark_v4_generated_greedy_shared_continue.h"
    if front_embed_first:
        return "dspark_v4_generated_greedy_front_embed.h"
    if routed_interleave:
        return "dspark_v4_generated_greedy_interleave128.h"
    if routed_group_ready:
        return "dspark_v4_generated_greedy_group_ready.h"
    if w13_fused_swiglu:
        suffix = "_woa_split2" if woa_split_k == 2 else ""
        return f"dspark_v4_generated_greedy_w13_swiglu{suffix}.h"
    if woa_split_k != 1:
        return "dspark_v4_generated_greedy_woa_split2.h"
    if gap_schedule == 2:
        return "dspark_v4_generated_greedy_gaps_measured.h"
    if gap_schedule:
        return "dspark_v4_generated_greedy_gaps.h"
    if hc_split:
        return "dspark_v4_generated_greedy_hc_split.h"
    if shared_first:
        return "dspark_v4_generated_greedy_shared_first.h"
    if fine_overlap:
        return "dspark_v4_generated_greedy_overlap.h"
    stem = (
        "dspark_v4_generated_greedy"
        if greedy
        else "dspark_v4_generated_relaxed"
        if relaxed
        else "dspark_v4_generated"
    )
    return f"{stem}_b{batch}.h" if batch != 1 else f"{stem}.h"


# Ship the existing base modes plus the retained full-proposal schedule.
# Intermediate schedules remain constructible for CPU DAG checks, without
# keeping a separate thousand-line generated snapshot for each experiment.
_BATCHED_SIZES = (2, 4, 8)
_VARIANTS = [
    dict(
        relaxed=True,
        greedy=True,
        shared_first=True,
        hc_split=True,
        woa_split_k=2,
        w13_fused_swiglu=True,
        shared_continue=True,
        routed_w2_workers128=True,
    ),
    dict(relaxed=False, greedy=False),
    dict(relaxed=True, greedy=False),
    dict(relaxed=True, greedy=True),
    dict(relaxed=True, greedy=True, fine_overlap=True),
    *[dict(relaxed=True, greedy=True, batch=batch) for batch in _BATCHED_SIZES],
]


def _emit(variant: dict, *, check: bool) -> None:
    output = REPOSITORY / "deepspec/megakernel/csrc" / header_name(**variant)
    generated = generate(**variant)
    if check:
        if not output.is_file() or output.read_text() != generated:
            raise SystemExit(
                f"stale generated V4 header {output.name}: run {Path(__file__).name} --all"
            )
    else:
        output.write_text(generated)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--all", action="store_true", help="every generated variant")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fine-overlap", action="store_true")
    parser.add_argument("--shared-first", action="store_true")
    parser.add_argument("--hc-split", action="store_true")
    parser.add_argument("--w13-fused-swiglu", action="store_true")
    parser.add_argument("--shared-continue", action="store_true")
    parser.add_argument("--routed-w2-workers128", action="store_true")
    parser.add_argument("--routed-group-ready", action="store_true")
    parser.add_argument("--woa-split-k", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gap-schedule", nargs="?", const=1, type=int, choices=(1, 2), default=0)
    args = parser.parse_args()
    if args.check or args.all:
        # --check has always meant "verify the whole generated set"; it now
        # also covers the batched greedy headers.
        for variant in _VARIANTS:
            _emit(variant, check=args.check)
        return 0
    if args.greedy and not args.relaxed:
        parser.error("--greedy requires --relaxed")
    _emit(
        {
            "relaxed": args.relaxed,
            "greedy": args.greedy,
            "batch": args.batch,
            "fine_overlap": args.fine_overlap,
            "shared_first": args.shared_first,
            "hc_split": args.hc_split,
            "woa_split_k": args.woa_split_k,
            "w13_fused_swiglu": args.w13_fused_swiglu,
            "shared_continue": args.shared_continue,
            "routed_w2_workers128": args.routed_w2_workers128,
            "routed_group_ready": args.routed_group_ready,
            "gap_schedule": args.gap_schedule,
        },
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
