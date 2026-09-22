#!/usr/bin/env python3
"""Render recorded DSpark task/controller spans, not inferred GPU utilization.

Input: the nanosecond JSON export from V4DeviceTrace.write_json(). No GPU needed.
See docs/bubbles.md for the README figure's provenance and reproduction command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCHEMA = "deepspec.deepseek_v4.device_trace.v1"
FAMILIES = {
    "Preparation / target KV": "#5cc8b2",
    "Attention": "#6ba9f4",
    "FFN prep / routing": "#e09ed8",
    "Routed W13": "#ffb366",
    "Routed W2": "#f57778",
    "Shared expert": "#efd575",
    "Expert combine": "#81d7cd",
    "Head / vocabulary": "#ac9af1",
    "Markov / finalize": "#91cc86",
    "Other task": "#d5d9e3",
}
BACKGROUND = "#101720"
PANEL = "#17212d"
FOREGROUND = "#eef3fa"
MUTED = "#aebdce"
CONTROLLER = "#394658"


@dataclass(frozen=True)
class Span:
    worker: int
    phase: str
    segment: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class Trace:
    spans: tuple[Span, ...]
    workers: int
    origin_ns: int
    end_ns: int
    sha256: str

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.origin_ns) / 1000


def read_trace(path: Path) -> Trace:
    data = path.read_bytes()
    payload = json.loads(data)
    if payload.get("schema") != SCHEMA or payload.get("timestamp_unit") != "ns":
        raise ValueError("expected a V4 device trace with nanosecond timestamps")
    events = payload["events"]
    summary = payload["summary"]
    if not events or summary["events"] != len(events):
        raise ValueError("empty trace or event count does not match summary")
    workers = summary["workers"]
    spans = set()
    for event in events:
        start, end, worker = (event[k] for k in ("start_ns", "end_ns", "worker"))
        # Absolute globaltimer values exceed float's exact integer range.
        if any(type(v) is not int for v in (start, end, worker)):
            raise ValueError("timestamps and worker IDs must be integers")
        if start < 0 or end < start or not 0 <= worker < workers:
            raise ValueError("invalid interval or worker outside declared grid")
        spans.add(Span(worker, event["phase"], event["segment"], start, end))
    if not any(s.segment == "USEFUL" and s.end_ns > s.start_ns for s in spans):
        raise ValueError("no nonempty task spans; entry-only probes cannot produce this diagram")
    return Trace(
        tuple(sorted(spans, key=lambda s: (s.start_ns, s.worker, s.segment, s.phase))),
        workers,
        min(s.start_ns for s in spans),
        max(s.end_ns for s in spans),
        hashlib.sha256(data).hexdigest(),
    )


def family(phase: str) -> str:
    if phase.startswith(("head.", "tail_", "proposal.")):
        return "Head / vocabulary" if phase.startswith("head.") else "Markov / finalize"
    if phase.startswith(("main.", "draft.")) or ".main_kv" in phase:
        return "Preparation / target KV"
    if "routed" in phase:
        return "Routed W2" if "w2" in phase else "Routed W13"
    if "shared" in phase:
        return "Shared expert"
    if "expert_combine" in phase:
        return "Expert combine"
    if "ffn_hc" in phase or "router" in phase:
        return "FFN prep / routing"
    if phase.startswith("layer_"):
        return "Attention"
    return "Other task"


def task_counts(trace: Trace) -> tuple[list[float], list[int]]:
    """Sweep the union per CTA; overlapping roles/tasks never count a CTA twice."""
    by_worker = defaultdict(list)
    for s in trace.spans:
        if s.segment == "USEFUL" and s.end_ns > s.start_ns:
            by_worker[s.worker].append((s.start_ns, s.end_ns))
    deltas = defaultdict(int)
    for intervals in by_worker.values():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            deltas[start] += 1
            deltas[end] -= 1
    deltas[trace.origin_ns] += 0
    deltas[trace.end_ns] += 0
    times, counts, active = [], [], 0
    for stamp, delta in sorted(deltas.items()):
        active += delta
        times.append((stamp - trace.origin_ns) / 1000)
        counts.append(active)
    return times, counts


def render(trace: Trace, output: Path, title: str, context: str, zoom: tuple[float, float]):
    if not 0 <= zoom[0] < zoom[1] <= trace.duration_us:
        raise ValueError("zoom must be inside the recorded span, in microseconds")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(16, 10), facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        3,
        1,
        left=0.075,
        right=0.975,
        top=0.785,
        bottom=0.10,
        height_ratios=(3.8, 0.9, 2.5),
        hspace=0.43,
    )
    overview, activity, detail = (fig.add_subplot(g) for g in grid)
    for ax in (overview, activity, detail):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=10, length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#6e8197", alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)

    segments = defaultdict(list)
    for s in trace.spans:
        if s.segment not in {"USEFUL", "CONTROLLER"} or s.end_ns == s.start_ns:
            continue
        # Subtract integer timestamps BEFORE converting to fractional microseconds.
        x0, x1 = ((t - trace.origin_ns) / 1000 for t in (s.start_ns, s.end_ns))
        label = "Controller" if s.segment == "CONTROLLER" else family(s.phase)
        segments[label].append(((x0, s.worker), (x1, s.worker)))
    colors = {"Controller": CONTROLLER, **FAMILIES}
    for ax, limits in ((overview, (0, trace.duration_us)), (detail, zoom)):
        for label, color in colors.items():
            if label in segments:
                # Butt caps and exact endpoints: no minimum-width invented work.
                ax.add_collection(
                    LineCollection(
                        segments[label],
                        colors=color,
                        linewidths=0.82,
                        capstyle="butt",
                        zorder=2 if label == "Controller" else 3,
                    )
                )
        ax.set_xlim(*limits)
        ax.set_ylim(trace.workers - 0.5, -0.5)
        ax.set_yticks(sorted({0, trace.workers // 2, trace.workers - 1}))
        ax.set_ylabel("Worker CTA", color=MUTED, labelpad=12)

    overview.set_title(
        "FULL CAPTURE  /  ALL WORKERS",
        color=FOREGROUND,
        loc="left",
        pad=12,
        fontsize=11,
        weight="bold",
    )
    overview.axvspan(*zoom, facecolor="none", edgecolor=FOREGROUND, lw=1.0, zorder=4)
    times, counts = task_counts(trace)
    activity.fill_between(times, counts, step="post", color="#6ba9f4", alpha=0.30)
    activity.step(times, counts, where="post", color="#8bbcf5", linewidth=0.85)
    activity.set_xlim(0, trace.duration_us)
    activity.set_ylim(0, trace.workers)
    activity.set_yticks([0, trace.workers])
    activity.set_ylabel("CTAs in\ntask spans", color=MUTED, fontsize=10)
    activity.set_xlabel("Time since first recorded event (µs)", color=MUTED, fontsize=10)
    detail.set_title(
        f"ZOOM  /  {zoom[0]:g}–{zoom[1]:g} µs  /  SAME WORKERS",
        color=FOREGROUND,
        loc="left",
        pad=12,
        fontsize=11,
        weight="bold",
    )
    detail.set_xlabel("Time since first recorded event (µs)", color=MUTED, fontsize=10)

    fig.text(0.075, 0.958, title, color=FOREGROUND, fontsize=25, weight="bold")
    fig.text(0.075, 0.923, context, color=MUTED, fontsize=12)
    tasks = sum(s.segment == "USEFUL" and s.end_ns > s.start_ns for s in trace.spans)
    fig.text(
        0.075,
        0.890,
        f"{trace.workers} persistent CTAs     /     {tasks:,} task spans"
        f"     /     {trace.duration_us:,.3f} µs recorded span",
        color=FOREGROUND,
        fontsize=12,
    )
    fig.legend(
        handles=[
            Patch(color=color, label=label) for label, color in colors.items() if label in segments
        ],
        loc="upper left",
        bbox_to_anchor=(0.07, 0.875),
        ncol=5,
        frameon=False,
        labelcolor=FOREGROUND,
        fontsize=10,
        handlelength=1.1,
        columnspacing=1.6,
    )
    fig.text(
        0.075,
        0.039,
        "Color = recorded task body   ·   Gray = controller (may include waiting)"
        "   ·   Blank = unrecorded",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.075,
        0.015,
        f"Task-span coverage is not hardware utilization.  Source SHA256: {trace.sha256[:16]}…",
        color=MUTED,
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="V4DeviceTrace.write_json() export")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--title", default="Inside a DSpark megakernel")
    parser.add_argument("--context", required=True, help="GPU, date, and capture scope")
    parser.add_argument("--zoom-us", type=float, nargs=2, required=True, metavar=("START", "END"))
    args = parser.parse_args()
    trace = read_trace(args.trace)
    render(trace, args.output, args.title, args.context, tuple(args.zoom_us))
    print(f"{args.output}: {trace.duration_us:.3f} µs, {trace.workers} workers")
    print(f"input SHA256: {trace.sha256}")


if __name__ == "__main__":
    main()
