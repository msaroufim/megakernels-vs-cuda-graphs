# Reading and reproducing the worker diagram

The [README figure](../../docs/dspark-bubbles.png) uses device `%globaltimer`
start/end records from one historical DSpark proposal on a **152-SM GB300**.
The layout is inspired by HazyResearch's
[Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

Each row is a persistent worker CTA. Colors identify recorded task bodies;
gray shows controller spans, which can include dependency waiting. Blank
regions have no displayed task/controller record. They are **not measured idle
time**. Task bodies include instrumentation and dispatch bookkeeping, so their
widths are not isolated matmul timings either.

The middle panel counts CTAs inside at least one recorded task body. It takes
the union per worker, removing duplicated TMA/MMA/epilogue role records and
overlapping spans. This is task-span coverage, **not hardware utilization**.
The bottom panel magnifies 140–245 µs: first-layer routing, shared/routed expert
work and combination. W13 is the gate/up projection; W2 is the down projection.
Gray regions and sparse task coverage suggest where to investigate; they do
not identify stall causes without further instrumentation.

## Which measurement is this?

**September 4, 2026, instrumented control**, using SF4 dense kernels and shared
bulk staging, before the later native shared-expert/continuation work. It is
not the September 14 equal-work implementation or its **828.544 µs p50**.

Of seven control captures, recorded spans were 951.008, 960.928, 954.752,
959.424, **956.768**, 957.760 and 955.904 µs. Capture 4 is the median (p50),
containing 19,576 role records, 100 phases and 5,408 distinct task-body spans.
The span is first-to-last device record, not a CUDA-event latency measurement.

External artifact packet: `profile-review-20260904/04-latest-paired-device-traces/committed-kv-device-paired`.
Input: `control/device/4/phases.json`.
SHA256: `a11e42042af4ce03f379bc207ced4f4ee734284933d748e5e0afb5c61abfaa1c`.
The packet's `paired-receipt.json` records build flags and the instrumented
control module identifier ending in `c843a2bc463c94b5`.
Its recorded source hashes are:

| Source | SHA256 |
|---|---|
| `deepspec/megakernel/extension.py` | `d4cd0a11306aa3129f16faff8eb04a0ad07bf2165a587592fc6a30ae600f17b2` |
| `deepspec/megakernel/full_loop_proposal.py` | `c2acd7f6434450326419388770a2be03f0debe60d85fab3d1bbda9a99450493c` |

## Render locally

The raw packet is external, like the other DSpark replay artifacts. The PNG
is checked in; regenerating it requires the input above. No GPU is needed.
From `dspark/`, after restoring the packet:

```bash
uv sync --frozen --group dev
export TRACE_PACKET=/path/to/committed-kv-device-paired
shasum -a 256 "$TRACE_PACKET/control/device/4/phases.json"
# Check against the SHA256 above before rendering.
uv run python scripts/megakernel/render_trace.py \
  "$TRACE_PACKET/control/device/4/phases.json" \
  --output ../docs/dspark-bubbles.png \
  --context 'GB300 · September 4, 2026 · Median of 7 instrumented captures · Historical implementation' \
  --zoom-us 140 245
```

For another capture, pass a JSON export from
[`V4DeviceTrace.write_json()`](../deepspec/megakernel/v4_trace.py), its GPU/date/scope
in `--context`, and a zoom range in microseconds. The renderer validates the
schema, units, event count and intervals; it subtracts integer timestamps
before converting units and preserves recorded durations without widening
short tasks. Entry-only probes cannot supply this diagram. Wait/prefetch
segments are not drawn or counted; the selected capture contains neither.
