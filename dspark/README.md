# DSpark: a persistent GPU scheduler versus CUDA Graphs

| September 14, 2026 matched-work result | Proposal p50 |
|---|---:|
| SGLang-based CUDA Graph with PDL | **803.968 µs** |
| One persistent GPU-scheduled CUDA kernel, without PDL | **828.544 µs** |

The megakernel takes **3.06% longer** and wins **0/96 pairs** (median paired
difference: +22.592 µs). One GB300, one session, one fixed input; no new GPU run
was made during packaging. Each of three blocks uses eight warmup pairs,
alternating arm order, same-arm primers and CUDA-event timing without a profiler.

## Workload and limits

One request, five draft tokens, three draft layers, 128 cached keys plus five
draft slots, TP1. The megakernel uses 152 resident CTAs and 100 phases containing
5,793 scheduled task chunks. The SGLang 0.5.16 graph uses 176 kernel launches.

Both include target-row selection, projection/normalization, three target KV
writes, proposal preparation, draft layers, head/Markov processing and
confidence/STS publication. Target verification, allocation, packing/loading
and target-tap mean/concatenation are excluded. The graph has custom DSpark
wrappers. This is not a scheduler-only ablation or an end-to-end serving test.

Greedy IDs match, but head/base-logit relative L2 differs by **4.1622% / 3.4438%**;
confidence differs by up to **0.03929**. Evolving-cache execution and normalized
speculative sampling remain unqualified. Eleven byte-exact outputs match the
native parent, not the graph.

## Source map

- [Persistent CUDA scheduler](deepspec/megakernel/csrc/dspark_v4_kernel.cu) and [proposal I/O](deepspec/megakernel/csrc/dspark_proposal_io.cuh).
- [Matched-work benchmark](scripts/megakernel/hazy_hybrid/full_proposal_lab.py), [build integration](scripts/megakernel/hazy_hybrid/full_proposal_build.py), and [pinned inputs](scripts/megakernel/hazy_hybrid/pins.json).

Shared runtime modules with `full_loop` names support the proposal harness.

## Historical replay requires external artifacts

**Replay requires the original artifact packet:** weights, prefill/target-tap
tensors, native binaries, source-capture helpers and build records. These are
not included. A public checkpoint alone is insufficient; fresh builds need
new numerical and runtime validation.

Checkpoint: [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark),
revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`. The measured fixture tensors
are separate and must match the replay hashes.

The measured runtime was Torch **2.11.0+cu130**, NVCC **13.0.88**, driver **580.126.20**, using:

```text
docker.io/lmsysorg/sglang@sha256:ba76b5c16979039b29c0e7dff896ca71760841431af9041cc1b6a2c362b64c06
```

The bundled `pyproject.toml` / `uv.lock` describe source development and CPU testing; they do **not** recreate this measured container. The benchmark rejects devices other than one visible 152-SM GB300. Use this subtree's commands, not the root Llama runner.

With the original packet restored, provide your own paths below. Run from this directory, use a fresh output directory, and preserve packet contents and SHA checks. `ARTIFACT_ROOT` is the historical tools root, including `hazy-original-post-ffn-capture-draft-0911/capture.py`, its admitted source snapshot and the dependencies named in `pins.json`. Its parent must contain `extensions-resumed-completion-red-0908/build-record.json` and the source/binary paths that record references. Recorded absolute paths must resolve to the original files; changing paths must
preserve their hashes and provenance.

```bash
# Set these to supplied artifacts; all are required.
export ARTIFACT_ROOT=/path/to/original/tools
export MODEL_SNAPSHOT=/path/to/snapshots/62af8fffb2f7030cac4de2f0169f5b8d1101b646
export PREFILL_SNAPSHOT=/path/to/pinned-prefill.pt
export TARGET_TAPS=/path/to/pinned-target-taps.pt
export INTEGRATED_BUILD_RECORD=/path/to/integrated/build-record.json
export NEW_OUTPUT=/path/to/new-replay-output
export GPU_UUID=GPU-your-GB300-uuid

CUDA_VISIBLE_DEVICES="$GPU_UUID" python scripts/megakernel/hazy_hybrid/full_proposal_lab.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$NEW_OUTPUT" \
  --snapshot "$MODEL_SNAPSHOT" \
  --prefill-snapshot "$PREFILL_SNAPSHOT" \
  --target-taps "$TARGET_TAPS" \
  --build-record "$INTEGRATED_BUILD_RECORD" \
  --overlay deepspec/megakernel \
  --dist-port 29617 \
  --disjoint-draft-kv-fixture --equal-work
```

The native binary is supplied through the build record's `module` path and `binary_sha256`, not a separate benchmark flag. The measured integrated image SHA256 is **`edc71b2e4f67057d6b37f57090134c5e2e73a6d07c339b4e67e1aba1ba89a757`**. The retained parent record/binary hashes and the prefill/target-tap hashes are checked by source; do not edit those pins to make unrelated inputs appear equivalent.

To rebuild against the original parent source and compiler inputs:

```bash
export NEW_BUILD=/path/to/new-build-output
CUDA_VISIBLE_DEVICES= python scripts/megakernel/hazy_hybrid/full_proposal_build.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$NEW_BUILD" \
  --overlay deepspec/megakernel/csrc
```

This produces a new `build-record.json` requiring numerical/runtime validation.
The original packet's `pipeline.json` benchmark stage and `main_bootstrap.py`
record the measured command and cache/environment setup.

## CPU checks

Run these from this directory using Python 3.11+ and the subtree's isolated environment:

```bash
uv sync --frozen --group dev
uv run ruff format --check deepspec tests scripts
uv run ruff check deepspec tests scripts
uv run ty check deepspec/megakernel
uv run python -m pytest -q
```

GPU tests require external fixtures and skip on CPU. CPU checks validate source
integration, not historical GPU timing or numerics.

## Provenance

Derived from [DeepSpec](https://github.com/deepseek-ai/DeepSpec), commit
`005e03b81cec38b7da6399833d609ee89a2587f2`. [LICENSE](LICENSE) and [NOTICE](NOTICE)
retain upstream and third-party terms. Model weights are separate.
