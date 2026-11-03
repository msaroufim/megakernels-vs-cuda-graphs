from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum, IntFlag
from pathlib import Path

import torch

from deepspec.megakernel.v4_schedule import V4Phase, V4WorkerRole


class V4TraceSegment(IntEnum):
    USEFUL = 0
    CONTROLLER = 1
    DEPENDENCY_WAIT = 2
    QUEUE_EMPTY_WAIT = 3
    ROLE_WAIT = 4
    # Idle-window L2 weight prefetch: deliberately not USEFUL so occupancy
    # and productivity metrics never paint scheduler bubbles as work.
    PREFETCH = 5

    @property
    def useful(self) -> bool:
        return self in {V4TraceSegment.USEFUL, V4TraceSegment.CONTROLLER}


class V4TraceFlag(IntFlag):
    NONE = 0
    STOLEN = 1 << 0
    LAST_TILE = 1 << 1
    REMOTE_PUBLISH = 1 << 2
    QUEUE_OVERFLOW = 1 << 3
    CAUSAL_CONFIDENCE_READ = 1 << 4


@dataclass(frozen=True)
class V4TraceEvent:
    worker: int
    event_index: int
    phase_id: int
    phase: str
    role: V4WorkerRole
    segment: V4TraceSegment
    start_ns: int
    end_ns: int
    work_item: int
    ticket: int
    flags: V4TraceFlag

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def warp_duration_ns(self) -> int:
        return self.duration_ns * self.role.warps


@dataclass(frozen=True)
class V4TraceSummary:
    workers: int
    events: int
    kernel_span_ns: int
    warp_capacity_ns: int
    useful_warp_ns: int
    controller_warp_ns: int
    dependency_wait_warp_ns: int
    queue_empty_wait_warp_ns: int
    role_wait_warp_ns: int
    unaccounted_warp_ns: int
    productive_fraction: float
    wait_fraction: float
    top_level_kernel_launches: int | None
    inter_launch_gap_ns: int | None
    prefetch_warp_ns: int = 0
    prefetch_fraction: float = 0.0


@dataclass(frozen=True)
class V4PhaseTraceSummary:
    phase_id: int
    phase: str
    useful_warp_ns: int
    controller_warp_ns: int
    dependency_wait_warp_ns: int
    queue_empty_wait_warp_ns: int
    role_wait_warp_ns: int
    work_items: int
    stolen_work_items: int
    useful_warps_per_cta: int

    @property
    def wait_warp_ns(self) -> int:
        return self.dependency_wait_warp_ns + self.queue_empty_wait_warp_ns + self.role_wait_warp_ns


@dataclass(frozen=True)
class V4DeviceTrace:
    """Decoded event ring written directly by the persistent proposal kernel.

    Raw columns are phase, role, segment, start, end, work item, scheduler
    ticket, and flags. `trace_counts` is written by the same kernel and prevents
    host decoding from treating untouched ring entries as events.
    """

    events: tuple[V4TraceEvent, ...]
    workers: int
    event_capacity: int
    phase_program: tuple[V4Phase, ...]

    @classmethod
    def from_tensors(
        cls,
        raw: torch.Tensor,
        counts: torch.Tensor,
        phase_program: tuple[V4Phase, ...],
    ) -> V4DeviceTrace:
        if raw.ndim != 3 or raw.shape[-1] != 8:
            raise ValueError("V4 trace records must have shape [workers, capacity, 8]")
        if counts.ndim != 1 or counts.shape[0] != raw.shape[0]:
            raise ValueError("V4 trace counts must have shape [workers]")
        host = raw.detach().to(device="cpu", dtype=torch.int64)
        host_counts = counts.detach().to(device="cpu", dtype=torch.int64)
        workers, capacity, _ = host.shape
        phase_names = {phase.phase_id: phase.name for phase in phase_program}
        events: list[V4TraceEvent] = []
        last_end: dict[tuple[int, V4WorkerRole], int] = {}
        for worker in range(workers):
            count = int(host_counts[worker])
            if not 0 <= count <= capacity:
                raise ValueError(
                    f"trace count {count} for worker {worker} exceeds capacity {capacity}"
                )
            for event_index in range(count):
                (
                    phase_id,
                    raw_role,
                    raw_segment,
                    start_ns,
                    end_ns,
                    work_item,
                    ticket,
                    raw_flags,
                ) = (int(value) for value in host[worker, event_index])
                if phase_id not in phase_names:
                    raise ValueError(f"unknown V4 phase id {phase_id}")
                try:
                    role = V4WorkerRole(raw_role)
                except ValueError as error:
                    raise ValueError(f"unknown V4 worker role {raw_role}") from error
                try:
                    segment = V4TraceSegment(raw_segment)
                except ValueError as error:
                    raise ValueError(f"unknown V4 trace segment {raw_segment}") from error
                if end_ns < start_ns:
                    raise ValueError(f"non-monotonic event worker={worker} index={event_index}")
                key = (worker, role)
                if start_ns < last_end.get(key, start_ns):
                    raise ValueError(f"overlapping role timeline worker={worker} role={role.name}")
                last_end[key] = end_ns
                events.append(
                    V4TraceEvent(
                        worker=worker,
                        event_index=event_index,
                        phase_id=phase_id,
                        phase=phase_names[phase_id],
                        role=role,
                        segment=segment,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        work_item=work_item,
                        ticket=ticket,
                        flags=V4TraceFlag(raw_flags),
                    )
                )
        return cls(tuple(events), workers, capacity, phase_program)

    def summarize(
        self,
        *,
        top_level_kernel_launches: int | None = None,
    ) -> V4TraceSummary:
        if not self.events:
            return V4TraceSummary(
                workers=self.workers,
                events=0,
                kernel_span_ns=0,
                warp_capacity_ns=0,
                useful_warp_ns=0,
                controller_warp_ns=0,
                dependency_wait_warp_ns=0,
                queue_empty_wait_warp_ns=0,
                role_wait_warp_ns=0,
                unaccounted_warp_ns=0,
                productive_fraction=0.0,
                wait_fraction=0.0,
                top_level_kernel_launches=top_level_kernel_launches,
                inter_launch_gap_ns=0 if top_level_kernel_launches == 1 else None,
            )
        origin = min(event.start_ns for event in self.events)
        finish = max(event.end_ns for event in self.events)
        span = finish - origin
        warps_per_worker = sum(role.warps for role in V4WorkerRole)
        capacity = span * self.workers * warps_per_worker

        def total(segment: V4TraceSegment) -> int:
            return sum(event.warp_duration_ns for event in self.events if event.segment == segment)

        useful = total(V4TraceSegment.USEFUL)
        controller = total(V4TraceSegment.CONTROLLER)
        dependency_wait = total(V4TraceSegment.DEPENDENCY_WAIT)
        queue_empty_wait = total(V4TraceSegment.QUEUE_EMPTY_WAIT)
        role_wait = total(V4TraceSegment.ROLE_WAIT)
        prefetch = total(V4TraceSegment.PREFETCH)
        accounted = useful + controller + dependency_wait + queue_empty_wait + role_wait + prefetch
        waits = dependency_wait + queue_empty_wait + role_wait
        return V4TraceSummary(
            workers=self.workers,
            events=len(self.events),
            kernel_span_ns=span,
            warp_capacity_ns=capacity,
            useful_warp_ns=useful,
            controller_warp_ns=controller,
            dependency_wait_warp_ns=dependency_wait,
            queue_empty_wait_warp_ns=queue_empty_wait,
            role_wait_warp_ns=role_wait,
            unaccounted_warp_ns=max(capacity - accounted, 0),
            productive_fraction=(useful + controller) / capacity if capacity else 0.0,
            wait_fraction=waits / capacity if capacity else 0.0,
            top_level_kernel_launches=top_level_kernel_launches,
            inter_launch_gap_ns=0 if top_level_kernel_launches == 1 else None,
            prefetch_warp_ns=prefetch,
            prefetch_fraction=prefetch / capacity if capacity else 0.0,
        )

    def phase_summaries(self) -> tuple[V4PhaseTraceSummary, ...]:
        grouped: dict[int, list[V4TraceEvent]] = {}
        for event in self.events:
            grouped.setdefault(event.phase_id, []).append(event)

        def total(items: list[V4TraceEvent], segment: V4TraceSegment) -> int:
            return sum(event.warp_duration_ns for event in items if event.segment == segment)

        summaries = []
        for phase_id in sorted(grouped):
            items = grouped[phase_id]
            useful_items = [item for item in items if item.segment == V4TraceSegment.USEFUL]
            summaries.append(
                V4PhaseTraceSummary(
                    phase_id=phase_id,
                    phase=items[0].phase,
                    useful_warp_ns=total(items, V4TraceSegment.USEFUL),
                    controller_warp_ns=total(items, V4TraceSegment.CONTROLLER),
                    dependency_wait_warp_ns=total(items, V4TraceSegment.DEPENDENCY_WAIT),
                    queue_empty_wait_warp_ns=total(items, V4TraceSegment.QUEUE_EMPTY_WAIT),
                    role_wait_warp_ns=total(items, V4TraceSegment.ROLE_WAIT),
                    work_items=len({(item.ticket, item.work_item) for item in useful_items}),
                    stolen_work_items=sum(
                        bool(item.flags & V4TraceFlag.STOLEN) for item in useful_items
                    ),
                    # useful_warp_ns weights each event by its role's warp count,
                    # and a phase only reports the roles it actually uses (a
                    # VECTOR phase is EPILOGUE-only = 2 warps, a GEMM phase is
                    # TMA+MMA+EPILOGUE = 7). Dividing that total by a flat 8
                    # warps/CTA therefore reports a perfectly parallel vector
                    # phase as 4x "inflated". Carry the real divisor.
                    useful_warps_per_cta=sum(
                        role.warps for role in {item.role for item in useful_items}
                    ),
                )
            )
        return tuple(summaries)

    def phase_spans_ns(self) -> dict[str, int]:
        bounds: dict[str, tuple[int, int]] = {}
        for event in self.events:
            start, end = bounds.get(event.phase, (event.start_ns, event.end_ns))
            bounds[event.phase] = (
                min(start, event.start_ns),
                max(end, event.end_ns),
            )
        return {phase: end - start for phase, (start, end) in bounds.items()}

    def critical_path(self, phase_program) -> list[dict]:
        """Walk the binding phase-barrier chain that sets the kernel span.

        The scheduler publishes a phase only when every predecessor phase has
        fully completed, so phase readiness is the max predecessor completion.
        Each chain node reports the scheduler gap (ready -> first useful task
        start) and the execution stretch (first start -> last useful end),
        plus which predecessor was binding. The gap+exec sum along the chain
        approximates the kernel span; large exec entries are true critical
        work and large gaps are queue/claim latency.
        """

        first_start: dict[str, int] = {}
        completion: dict[str, int] = {}
        for event in self.events:
            if event.segment != V4TraceSegment.USEFUL:
                continue
            first_start[event.phase] = min(
                first_start.get(event.phase, event.start_ns), event.start_ns
            )
            completion[event.phase] = max(completion.get(event.phase, event.end_ns), event.end_ns)
        by_name = {phase.name: phase for phase in phase_program}
        origin = min(first_start.values(), default=0)

        def ready(name: str) -> tuple[int, str | None]:
            phase = by_name[name]
            best_time = origin
            best_dep: str | None = None
            for dep in phase.dependencies:
                dep_completion = completion.get(dep, origin)
                if dep_completion >= best_time:
                    best_time = dep_completion
                    best_dep = dep
            return best_time, best_dep

        if not completion:
            return []
        current: str | None = max(completion, key=lambda name: completion[name])
        chain: list[dict] = []
        while current is not None:
            ready_time, binding = ready(current)
            start = first_start.get(current, ready_time)
            chain.append(
                {
                    "phase": current,
                    "ready_ns": ready_time - origin,
                    "gap_ns": max(0, start - ready_time),
                    "exec_ns": max(0, completion[current] - max(start, ready_time)),
                    "binding": binding,
                }
            )
            current = binding
        chain.reverse()
        return chain

    def write_json(
        self,
        path: str | Path,
        *,
        top_level_kernel_launches: int = 1,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "deepspec.deepseek_v4.device_trace.v1",
            "timestamp_unit": "ns",
            "summary": asdict(self.summarize(top_level_kernel_launches=top_level_kernel_launches)),
            "phase_summary": [
                {**asdict(summary), "wait_warp_ns": summary.wait_warp_ns}
                for summary in self.phase_summaries()
            ],
            "events": [
                {
                    **asdict(event),
                    "role": event.role.name,
                    "segment": event.segment.name,
                    "flags": int(event.flags),
                }
                for event in self.events
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def write_perfetto(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        origin = min((event.start_ns for event in self.events), default=0)
        events = [
            {
                "name": event.phase,
                "cat": event.segment.name.lower(),
                "ph": "X",
                "pid": 4,
                "tid": event.worker * len(V4WorkerRole) + int(event.role),
                "ts": (event.start_ns - origin) / 1000.0,
                "dur": event.duration_ns / 1000.0,
                "args": {
                    "worker": event.worker,
                    "role": event.role.name,
                    "work_item": event.work_item,
                    "ticket": event.ticket,
                    "flags": int(event.flags),
                },
            }
            for event in self.events
        ]
        path.write_text(json.dumps({"traceEvents": events}) + "\n", encoding="utf-8")
        return path
