from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path

import torch


class TracePhase(IntEnum):
    MARKOV_GEMV = 1
    GRID_WAIT_AFTER_GEMV = 2
    ARGMAX_CONFIDENCE = 3
    GRID_WAIT_AFTER_REDUCTION = 4
    GRID_MAX_REDUCTION = 5
    SOFTMAX_EXP_SUM = 6
    GRID_SUM_REDUCTION = 7
    SOFTMAX_NORMALIZE = 8
    SERIAL_SAMPLE_CONFIDENCE = 9


@dataclass(frozen=True)
class TraceRange:
    worker: int
    step: int
    phase: TracePhase
    start_ns: int
    end_ns: int
    useful: bool

    @property
    def duration_ns(self) -> int:
        return max(self.end_ns - self.start_ns, 0)


@dataclass(frozen=True)
class BubbleSummary:
    workers: int
    proposal_steps: int
    kernel_span_ns: int
    worker_capacity_ns: int
    useful_ns: int
    wait_ns: int
    unaccounted_ns: int
    useful_fraction: float
    wait_fraction: float
    top_level_kernel_launches: int | None = None
    inter_launch_gap_ns: int | None = None


@dataclass(frozen=True)
class DeviceTrace:
    """Decoded in-kernel trace emitted without a second CUDA kernel.

    Raw layout is `[workers, proposal_steps, 8]`, containing begin/end pairs
    for useful GEMV work, the first grid wait, block-0 argmax/confidence work,
    and the second grid wait. Timestamps use `%globaltimer` nanoseconds.
    """

    ranges: tuple[TraceRange, ...]
    workers: int
    proposal_steps: int

    @classmethod
    def from_tensor(cls, raw: torch.Tensor) -> "DeviceTrace":
        if raw.ndim != 3 or raw.shape[-1] not in (8, 12):
            raise ValueError("trace tensor must have shape [workers, steps, 8 or 12]")
        host = raw.detach().to(device="cpu", dtype=torch.int64)
        workers, steps, _ = host.shape
        ranges = []
        if host.shape[-1] == 8:
            phase_pairs = (
                (TracePhase.MARKOV_GEMV, 0, 1),
                (TracePhase.GRID_WAIT_AFTER_GEMV, 2, 3),
                (TracePhase.ARGMAX_CONFIDENCE, 4, 5),
                (TracePhase.GRID_WAIT_AFTER_REDUCTION, 6, 7),
            )
        else:
            phase_pairs = (
                (TracePhase.MARKOV_GEMV, 0, 1),
                (TracePhase.GRID_MAX_REDUCTION, 2, 3),
                (TracePhase.SOFTMAX_EXP_SUM, 4, 5),
                (TracePhase.GRID_SUM_REDUCTION, 6, 7),
                (TracePhase.SOFTMAX_NORMALIZE, 8, 9),
                (TracePhase.SERIAL_SAMPLE_CONFIDENCE, 10, 11),
            )
        for worker in range(workers):
            for step in range(steps):
                for phase, start_col, end_col in phase_pairs:
                    start_ns = int(host[worker, step, start_col])
                    end_ns = int(host[worker, step, end_col])
                    if end_ns < start_ns:
                        raise ValueError(
                            f"non-monotonic trace for worker={worker}, step={step}, "
                            f"phase={phase.name}"
                        )
                    useful_for_every_worker = phase in {
                        TracePhase.MARKOV_GEMV,
                        TracePhase.SOFTMAX_EXP_SUM,
                        TracePhase.SOFTMAX_NORMALIZE,
                    }
                    useful_for_controller = phase in {
                        TracePhase.ARGMAX_CONFIDENCE,
                        TracePhase.GRID_MAX_REDUCTION,
                        TracePhase.GRID_SUM_REDUCTION,
                        TracePhase.SERIAL_SAMPLE_CONFIDENCE,
                    }
                    useful = useful_for_every_worker or (useful_for_controller and worker == 0)
                    ranges.append(
                        TraceRange(
                            worker=worker,
                            step=step,
                            phase=phase,
                            start_ns=start_ns,
                            end_ns=end_ns,
                            useful=useful,
                        )
                    )
        by_worker = {}
        for item in ranges:
            key = item.worker
            previous_end = by_worker.get(key)
            if previous_end is not None and item.start_ns < previous_end:
                raise ValueError(
                    f"overlapping or out-of-order phase for worker={item.worker}, "
                    f"step={item.step}, phase={item.phase.name}"
                )
            by_worker[key] = item.end_ns
        return cls(ranges=tuple(ranges), workers=workers, proposal_steps=steps)

    def summarize(self, *, top_level_kernel_launches: int | None = None) -> BubbleSummary:
        if not self.ranges:
            return BubbleSummary(
                workers=self.workers,
                proposal_steps=self.proposal_steps,
                kernel_span_ns=0,
                worker_capacity_ns=0,
                useful_ns=0,
                wait_ns=0,
                unaccounted_ns=0,
                useful_fraction=0.0,
                wait_fraction=0.0,
                top_level_kernel_launches=top_level_kernel_launches,
                inter_launch_gap_ns=0 if top_level_kernel_launches == 1 else None,
            )
        start_ns = min(item.start_ns for item in self.ranges)
        end_ns = max(item.end_ns for item in self.ranges)
        kernel_span_ns = max(end_ns - start_ns, 0)
        capacity = kernel_span_ns * self.workers
        useful = sum(item.duration_ns for item in self.ranges if item.useful)
        wait = sum(item.duration_ns for item in self.ranges if not item.useful)
        unaccounted = max(capacity - useful - wait, 0)
        return BubbleSummary(
            workers=self.workers,
            proposal_steps=self.proposal_steps,
            kernel_span_ns=kernel_span_ns,
            worker_capacity_ns=capacity,
            useful_ns=useful,
            wait_ns=wait,
            unaccounted_ns=unaccounted,
            useful_fraction=useful / capacity if capacity else 0.0,
            wait_fraction=wait / capacity if capacity else 0.0,
            top_level_kernel_launches=top_level_kernel_launches,
            inter_launch_gap_ns=0 if top_level_kernel_launches == 1 else None,
        )

    def write_json(self, path: str | Path, *, top_level_kernel_launches: int | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "deepspec.dspark.device_trace.v1",
            "timestamp_unit": "ns",
            "summary": asdict(self.summarize(top_level_kernel_launches=top_level_kernel_launches)),
            "ranges": [{**asdict(item), "phase": item.phase.name} for item in self.ranges],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def write_perfetto(self, path: str | Path) -> Path:
        """Write a Chrome-trace JSON file directly consumable by Perfetto."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        events = []
        if self.ranges:
            origin_ns = min(item.start_ns for item in self.ranges)
        else:
            origin_ns = 0
        for item in self.ranges:
            events.append(
                {
                    "name": item.phase.name,
                    "cat": "useful" if item.useful else "wait",
                    "ph": "X",
                    "pid": 1,
                    "tid": item.worker,
                    "ts": (item.start_ns - origin_ns) / 1000.0,
                    "dur": item.duration_ns / 1000.0,
                    "args": {"step": item.step, "useful": item.useful},
                }
            )
        path.write_text(json.dumps({"traceEvents": events}) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class FullPhaseSpec:
    name: str
    active_workers: int


@dataclass(frozen=True)
class FullTraceRange:
    worker: int
    phase_index: int
    phase: str
    segment: str
    start_ns: int
    end_ns: int
    useful: bool

    @property
    def duration_ns(self) -> int:
        return max(self.end_ns - self.start_ns, 0)


@dataclass(frozen=True)
class PhaseCapacitySummary:
    phase: str
    span_ns: int
    worker_capacity_ns: int
    useful_ns: int
    inactive_ns: int
    grid_wait_ns: int
    unaccounted_ns: int
    useful_fraction: float
    wait_fraction: float

    @property
    def wait_ns(self) -> int:
        return self.inactive_ns + self.grid_wait_ns


def full_trace_phase_specs(
    *,
    workers: int,
    block_size: int,
    context_length: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    feature_width: int,
    past_length: int = 0,
) -> tuple[FullPhaseSpec, ...]:
    """Describe the deterministic full-kernel phase order and active CTAs."""

    def elements(name: str, count: int) -> FullPhaseSpec:
        return FullPhaseSpec(name, min(workers, (count + 255) // 256))

    def vectors(name: str, count: int) -> FullPhaseSpec:
        return FullPhaseSpec(name, min(workers, count))

    def linear(name: str, rows: int, input_width: int, output_width: int) -> FullPhaseSpec:
        if input_width % 16 == 0 and output_width % 16 == 0:
            tiles = ((rows + 15) // 16) * (output_width // 16)
            return FullPhaseSpec(name, min(workers, (tiles + 7) // 8))
        return elements(name, rows * output_width)

    def linear_group(
        name: str,
        tasks: tuple[tuple[int, int, int], ...],
    ) -> FullPhaseSpec:
        if all(
            input_width % 16 == 0 and output_width % 16 == 0
            for _, input_width, output_width in tasks
        ):
            tiles = sum(
                ((rows + 15) // 16) * (output_width // 16) for rows, _, output_width in tasks
            )
            return FullPhaseSpec(name, min(workers, (tiles + 7) // 8))
        count = sum(rows * output_width for rows, _, output_width in tasks)
        return elements(name, count)

    def linear_split_k(
        name: str,
        rows: int,
        input_width: int,
        output_width: int,
    ) -> tuple[FullPhaseSpec, FullPhaseSpec]:
        padded_rows = ((rows + 15) // 16) * 16
        output_tiles = (padded_rows // 16) * (output_width // 16)
        splits = min(8, input_width // 16)
        partial_workers = min(workers, (output_tiles * splits + 7) // 8)
        reduction_workers = min(workers, (padded_rows * output_width + 255) // 256)
        return (
            FullPhaseSpec(f"{name}.split_k_partials", partial_workers),
            FullPhaseSpec(f"{name}.split_k_reduce", reduction_workers),
        )

    def linear_group_split_k(
        name: str,
        tasks: tuple[tuple[int, int, int], ...],
        splits: int,
        *,
        fuse_swiglu: bool = False,
    ) -> tuple[FullPhaseSpec, FullPhaseSpec]:
        output_tiles = sum(
            ((rows + 15) // 16) * (output_width // 16) for rows, _, output_width in tasks
        )
        if fuse_swiglu:
            rows, _, output_width = tasks[0]
            reduction_elements = rows * output_width
            reduction_name = f"{name}.split_k_reduce_swiglu"
        else:
            reduction_elements = sum(
                ((rows + 15) // 16) * 16 * output_width for rows, _, output_width in tasks
            )
            reduction_name = f"{name}.split_k_reduce"
        return (
            FullPhaseSpec(
                f"{name}.split_k_partials",
                min(workers, (output_tiles * splits + 7) // 8),
            ),
            FullPhaseSpec(
                reduction_name,
                min(workers, (reduction_elements + 255) // 256),
            ),
        )

    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    padded_context_rows = ((context_length + 15) // 16) * 16
    fast_attention = head_dim == 128 and past_length + context_length + block_size <= 32
    specs: list[FullPhaseSpec] = [
        elements("target_row_padding", padded_context_rows * feature_width),
    ]
    specs.extend(linear_split_k("context_projection", context_length, feature_width, hidden_size))
    specs.append(
        FullPhaseSpec(
            "context_norm_embedding_gather",
            min(workers, 1 + (block_size * hidden_size + 255) // 256),
        )
    )
    for layer in range(num_layers):
        prefix = f"layer_{layer}"
        if layer == 0:
            specs.append(vectors(f"{prefix}.input_norm", block_size))
        specs.extend(
            [
                *linear_group_split_k(
                    f"{prefix}.qkv_projection_group",
                    (
                        (block_size, hidden_size, q_width),
                        (context_length, hidden_size, kv_width),
                        (context_length, hidden_size, kv_width),
                        (block_size, hidden_size, kv_width),
                        (block_size, hidden_size, kv_width),
                    ),
                    2,
                ),
                FullPhaseSpec(
                    f"{prefix}.qkv_head_norm_rope",
                    min(
                        workers,
                        (
                            (
                                block_size * num_attention_heads
                                + context_length * num_key_value_heads
                                + block_size * num_key_value_heads
                            )
                            + 1
                        )
                        // 2
                        if head_dim <= 128
                        else (
                            block_size * num_attention_heads
                            + context_length * num_key_value_heads
                            + block_size * num_key_value_heads
                        ),
                    ),
                ),
                FullPhaseSpec(
                    f"{prefix}.attention_cache_append",
                    min(
                        workers,
                        (block_size * num_attention_heads + 1) // 2
                        + (context_length * num_key_value_heads * head_dim + 255) // 256
                        if fast_attention
                        else block_size * num_attention_heads,
                    ),
                ),
                *linear_split_k(
                    f"{prefix}.o_projection",
                    block_size,
                    q_width,
                    hidden_size,
                ),
                vectors(f"{prefix}.attention_residual_post_norm", block_size),
                *linear_group_split_k(
                    f"{prefix}.gate_up_projection_group",
                    (
                        (block_size, hidden_size, intermediate_size),
                        (block_size, hidden_size, intermediate_size),
                    ),
                    8,
                    fuse_swiglu=True,
                ),
                *linear_split_k(
                    f"{prefix}.down_projection",
                    block_size,
                    intermediate_size,
                    hidden_size,
                ),
                vectors(
                    f"{prefix}.mlp_residual_"
                    + ("next_input_norm" if layer + 1 < num_layers else "final_norm"),
                    block_size,
                ),
            ]
        )
    specs.append(linear("lm_head", block_size, hidden_size, vocab_size))
    for step in range(block_size):
        prefix = f"tail_{step}"
        specs.extend(
            [
                elements(f"{prefix}.markov_gemv", vocab_size),
                FullPhaseSpec(f"{prefix}.softmax_exp_sum", min(workers, vocab_size)),
                FullPhaseSpec(f"{prefix}.sum_reduction", 1),
                FullPhaseSpec(
                    f"{prefix}.softmax_normalize_sample_confidence",
                    min(workers, vocab_size),
                ),
            ]
        )
    if len(specs) > 160:
        raise ValueError(f"full trace needs {len(specs)} phases but capacity is 160")
    return tuple(specs)


@dataclass(frozen=True)
class FullDeviceTrace:
    ranges: tuple[FullTraceRange, ...]
    workers: int
    phases: int

    @classmethod
    def from_tensor(
        cls,
        raw: torch.Tensor,
        specs: tuple[FullPhaseSpec, ...],
    ) -> "FullDeviceTrace":
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError("full trace tensor must have shape [workers, phases, 3]")
        if raw.shape[1] < len(specs):
            raise ValueError("full trace tensor has insufficient phase capacity")
        host = raw[:, : len(specs)].detach().to(device="cpu", dtype=torch.int64)
        ranges: list[FullTraceRange] = []
        for worker in range(host.shape[0]):
            for phase_index, spec in enumerate(specs):
                start_ns, work_end_ns, phase_end_ns = (
                    int(value) for value in host[worker, phase_index]
                )
                if not start_ns <= work_end_ns <= phase_end_ns:
                    raise ValueError(f"non-monotonic full trace worker={worker} phase={spec.name}")
                ranges.append(
                    FullTraceRange(
                        worker=worker,
                        phase_index=phase_index,
                        phase=spec.name,
                        segment="work" if worker < spec.active_workers else "inactive",
                        start_ns=start_ns,
                        end_ns=work_end_ns,
                        useful=worker < spec.active_workers,
                    )
                )
                ranges.append(
                    FullTraceRange(
                        worker=worker,
                        phase_index=phase_index,
                        phase=spec.name,
                        segment="grid_wait",
                        start_ns=work_end_ns,
                        end_ns=phase_end_ns,
                        useful=False,
                    )
                )
        return cls(tuple(ranges), int(host.shape[0]), len(specs))

    def summarize(self, *, top_level_kernel_launches: int | None = None) -> BubbleSummary:
        if not self.ranges:
            return BubbleSummary(
                self.workers,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0.0,
                top_level_kernel_launches,
                0 if top_level_kernel_launches == 1 else None,
            )
        start_ns = min(item.start_ns for item in self.ranges)
        end_ns = max(item.end_ns for item in self.ranges)
        capacity = (end_ns - start_ns) * self.workers
        useful = sum(item.duration_ns for item in self.ranges if item.useful)
        wait = sum(item.duration_ns for item in self.ranges if not item.useful)
        return BubbleSummary(
            workers=self.workers,
            proposal_steps=0,
            kernel_span_ns=end_ns - start_ns,
            worker_capacity_ns=capacity,
            useful_ns=useful,
            wait_ns=wait,
            unaccounted_ns=max(capacity - useful - wait, 0),
            useful_fraction=useful / capacity if capacity else 0.0,
            wait_fraction=wait / capacity if capacity else 0.0,
            top_level_kernel_launches=top_level_kernel_launches,
            inter_launch_gap_ns=0 if top_level_kernel_launches == 1 else None,
        )

    def phase_durations_ns(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for item in self.ranges:
            totals[item.phase] = totals.get(item.phase, 0) + item.duration_ns
        return totals

    def phase_spans_ns(self) -> dict[str, int]:
        bounds: dict[str, tuple[int, int]] = {}
        for item in self.ranges:
            start, end = bounds.get(item.phase, (item.start_ns, item.end_ns))
            bounds[item.phase] = (min(start, item.start_ns), max(end, item.end_ns))
        return {name: end - start for name, (start, end) in bounds.items()}

    def phase_capacity_summaries(self) -> tuple[PhaseCapacitySummary, ...]:
        grouped: dict[int, list[FullTraceRange]] = {}
        for item in self.ranges:
            grouped.setdefault(item.phase_index, []).append(item)
        summaries = []
        for phase_index in sorted(grouped):
            items = grouped[phase_index]
            start_ns = min(item.start_ns for item in items)
            end_ns = max(item.end_ns for item in items)
            span_ns = end_ns - start_ns
            capacity = span_ns * self.workers
            useful = sum(item.duration_ns for item in items if item.useful)
            inactive = sum(item.duration_ns for item in items if item.segment == "inactive")
            grid_wait = sum(item.duration_ns for item in items if item.segment == "grid_wait")
            unaccounted = max(capacity - useful - inactive - grid_wait, 0)
            summaries.append(
                PhaseCapacitySummary(
                    phase=items[0].phase,
                    span_ns=span_ns,
                    worker_capacity_ns=capacity,
                    useful_ns=useful,
                    inactive_ns=inactive,
                    grid_wait_ns=grid_wait,
                    unaccounted_ns=unaccounted,
                    useful_fraction=useful / capacity if capacity else 0.0,
                    wait_fraction=(inactive + grid_wait) / capacity if capacity else 0.0,
                )
            )
        return tuple(summaries)

    def write_json(self, path: str | Path, *, top_level_kernel_launches: int = 1) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "deepspec.dspark.full_device_trace.v1",
            "timestamp_unit": "ns",
            "summary": asdict(self.summarize(top_level_kernel_launches=top_level_kernel_launches)),
            "phase_capacity": [
                {**asdict(item), "wait_ns": item.wait_ns}
                for item in self.phase_capacity_summaries()
            ],
            "ranges": [asdict(item) for item in self.ranges],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def write_perfetto(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        origin_ns = min((item.start_ns for item in self.ranges), default=0)
        events = [
            {
                "name": item.phase,
                "cat": "useful" if item.useful else "wait",
                "ph": "X",
                "pid": 2,
                "tid": item.worker,
                "ts": (item.start_ns - origin_ns) / 1000.0,
                "dur": item.duration_ns / 1000.0,
                "args": {"segment": item.segment, "phase_index": item.phase_index},
            }
            for item in self.ranges
        ]
        path.write_text(json.dumps({"traceEvents": events}) + "\n", encoding="utf-8")
        return path
