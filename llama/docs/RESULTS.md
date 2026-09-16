# Llama results and method

Same kernels and workload on H100 and H200; no H100 tuning. Each GPU has fresh
references and a separate source freeze. Measurements: H200 September 14, 2026;
H100 September 16, 2026 UTC.

| GPU | Graphs + PDL, µs/step | Hazy, µs/step | Hazy throughput advantage, 95% CI |
|---|---:|---:|---|
| H100 | 979.51 | 972.78 | 0.69% [0.67%, 0.72%] |
| H200 | 794.74 | 759.83 | 4.59% [4.50%, 4.71%] |

Latencies are arithmetic means of process means; throughput ratios are geometric
means with hierarchical prompt/process bootstrap intervals. Native-loop
diagnostics, excluded from these estimates, measured 1010.01/971.75 µs on H100
and 817.83/766.19 µs on H200 (conventional/Hazy).

## Workload

Batch 1, 32 input tokens, 128 outputs, KV capacity 16,384. The first output comes
from a fixture; 127 dependent decode steps are timed. Both arms share HF BF16
prefix KV, initial token, selection and embedding. Prefill, packing, compilation
and prefix resets are excluded; required barriers and position updates are timed.

Each GPU ran eight prompts × three fresh paired graph processes plus three
native diagnostics on one prompt. Each process uses five warmups and ten timed
sequences per arm, with balanced AB/BA order. Arms run serially on the same GPU;
checking never overlaps timing. Within-process samples are not independent
replicates. All 27 runs and independent checkers completed on each GPU,
covering 864 saved-output observations apiece.

## Numerical fidelity

HF FP32 eager execution of the BF16 checkpoint promoted to FP32 is the reference;
native HF BF16 calibrates acceptance. Canonical checks share token histories;
actual-output checks replay each arm's own saved history and compare final logits.

| Error minus native HF BF16 error | H100 conventional | H100 Hazy | H200 conventional | H200 Hazy |
|---|---:|---:|---:|---:|
| Canonical graph KL | −0.00024440 | +0.00010668 | −0.00024241 | +0.00833578 |
| Canonical graph absolute NLL | −0.00628648 | +0.00016063 | −0.00540327 | +0.03479354 |
| Actual timed final KL | −0.00018257 | +0.00010127 | −0.00008910 | +0.01203546 |
| Actual timed final absolute NLL | −0.00573267 | +0.00058635 | −0.00274440 | +0.04466018 |

Lower is better. Acceptance requires the upper paired-prompt bootstrap 95% bound
of each error delta to be ≤0. Conventional passes both criteria for canonical
eager/graph and actual initial/warmup/timed outputs in the primary graph study;
Hazy misses them. Both produce finite outputs and valid token boundaries.
Several Hazy intervals cross zero: a missed gate does not prove every output is
worse. Hazy's smaller H100 errors have no established explanation.

Both arms must pass for an accuracy-matched speed win; neither study meets that
rule. These metrics measure numerical fidelity, not downstream task accuracy.
Conventional scratch/logits are FP32; Hazy logits are BF16, so this is not an
identical-arithmetic scheduling ablation. Per-feature speedups need ablations.
The known prompt suite is a replication workload, not an unseen holdout. These
runs do not reconstruct the historical vLLM/SGLang blog comparison.

## Runtime and audit

Both use 132-SM Hopper GPUs, NGC PyTorch 25.03, Torch
`2.7.0a0+7c8ec84dab.nv25.03`, CUDA 12.8, nvcc 12.8.93, Triton 3.4.0 and
Transformers 4.48.3. Drivers differ: H100 580.159.04, H200 570.133.20.
This does not isolate hardware from driver effects.

Original Hazy commit: `7309cec801537b61fea3b50d7dfe454a6cde578e`;
ThunderKittens: `664c108d16f12707a73d3072ab525f26fb2b4f62`.
Build: unmodified `GPU=H100`, `sm_90a`. The conventional stager selects the
portable FP32-FMA path. [Source hashes](../tools/source_inventory.py).

Saved outputs, reference hashes, receipt joins and statistics were audited.
Distinct histories counted within processes: H100 181, H200 449. Generated
artifacts stay outside Git. Final summary SHA256s:

- H100: `520c3045386fa2f463d25dd5c54021f7e05472632632f4ba517d7088635a1191`
- H200: `bc06e093c85de50970d3430dbef25cd4da4f718d0f29299bebdddb66c651978e`

[Run instructions](../README.md). Frozen protocols: [original](PROTOCOL.md),
[H200](../PROTOCOL.md), [H100](H100_PROTOCOL.md).
[Separate DSpark study](../../dspark/README.md).
