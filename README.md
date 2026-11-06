# Megakernels vs CUDA Graphs

**Llama 1B decode is close between CUDA Graphs + programmatic dependent launch
(PDL) and the original HazyResearch megakernel:** measured gaps are 0.7–4.6%.
An experimental DSpark study is included separately below.

Megakernel research needs more implementations and baselines that others can
run, inspect and extend. We’re collecting both approaches to support that work.

| Case | GPU | CUDA Graphs + PDL | Megakernel | Difference |
|---|---|---:|---:|---|
| [Llama 1B](llama/docs/RESULTS.md) | H100 | 979.51 µs | 972.78 µs | Megakernel throughput +0.69% |
| [Llama 1B](llama/docs/RESULTS.md) | H200 | 794.74 µs | 759.83 µs | Megakernel throughput +4.59% |

Llama: mean per decode step, original unmodified HazyResearch kernel. The
conventional baseline passes the numerical criteria; Hazy misses them. These
timings do not establish a general advantage or end-to-end serving result.

## Layout

| Directory | Contents |
|---|---|
| [`llama/`](llama/README.md) | H100/H200 kernels, benchmark, references and one-GPU runner |
| [`experimental/dspark/`](experimental/dspark/README.md) | Exploratory GB300 proposal kernel, graph baseline and replay tools; external artifacts required |
| [`experimental/evo2/`](experimental/evo2/README.md) | Evo 2 7B decode on B300: CUDA Graph baseline versus the Hyperkernel megakernel |

Each study has its own environment and tests. Follow its README and run
commands from that study’s folder.

Improve either Llama baseline by editing [`llama/kernels/`](llama/kernels/) or
[`llama/megakernel/`](llama/megakernel/), then submit a PR. See the
[competitor guide](llama/AGENTS.md).

The Llama baseline uses fused projections, packed weights, split FP32 reductions
and Triton attention: 100 kernels per decode step. Graphs reduce host submissions;
PDL overlaps independent weight loading with preceding work. Start with
[llama/kernels/candidate.py](llama/kernels/candidate.py) and [projection.cu](llama/kernels/projection.cu).

## Experimental: DSpark

The [DSpark study](experimental/dspark/README.md) is an exploratory scheduler
prototype, not a validated performance result. On one fixed GB300 workload, the
megakernel took **828.544 µs p50** against **803.968 µs** for CUDA Graphs + PDL:
**3.06% longer**, winning 0/96 pairs. It also has unresolved logit/confidence
differences and requires unavailable external replay artifacts. Treat its code
and measurements as a starting point for further investigation.

## Sources

The Llama megakernel vendors [HazyResearch Megakernels](https://github.com/HazyResearch/Megakernels)
CUDA sources; see [origin and license](llama/megakernel/ORIGIN.md).
The runner fetches the pinned Hazy Python runtime and
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens).
[Measured source hashes](llama/tools/source_inventory.py).
DSpark derives from [DeepSpec](https://github.com/deepseek-ai/DeepSpec) and retains
its [license](experimental/dspark/LICENSE) and [notices](experimental/dspark/NOTICE),
which apply to that subtree. Model weights are obtained separately.
