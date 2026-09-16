# Megakernels vs CUDA Graphs

**Two case studies where CUDA Graphs + programmatic dependent launch (PDL)
perform roughly on par with megakernels:** Llama decode and DSpark proposals.
The measured gaps are 0.7–4.6%, with different numerical behavior.

Megakernel research needs more implementations and baselines that others can
run, inspect and extend. We’re collecting both approaches to support that work.

| Case | GPU | CUDA Graphs + PDL | Megakernel | Difference |
|---|---|---:|---:|---|
| [Llama 1B](llama/docs/RESULTS.md) | H100 | 979.51 µs | 972.78 µs | Megakernel throughput +0.69% |
| [Llama 1B](llama/docs/RESULTS.md) | H200 | 794.74 µs | 759.83 µs | Megakernel throughput +4.59% |
| [DSpark](dspark/README.md) | GB300 | 803.968 µs | 828.544 µs | Megakernel latency +3.06% |

Llama: mean per decode step, original unmodified HazyResearch kernel.
DSpark: p50 per proposal, one fixed input/session. These are timing comparisons:
Llama’s conventional baseline passes the numerical criteria; Hazy misses them.
DSpark has unresolved logit/confidence differences. Neither establishes a general
advantage or end-to-end serving result.

## Layout

| Directory | Contents |
|---|---|
| [`llama/`](llama/README.md) | H100/H200 kernels, benchmark, references and one-GPU runner |
| [`dspark/`](dspark/README.md) | GB300 proposal kernel, graph baseline and replay tools; external artifacts required |

Each directory has its own environment and tests. Follow its README and run
commands from that study’s folder.

The Llama baseline uses fused projections, packed weights, split FP32 reductions
and Triton attention: 100 kernels per decode step. Graphs reduce host submissions;
PDL overlaps independent weight loading with preceding work. Start with
[llama/kernels/candidate.py](llama/kernels/candidate.py) and [projection.cu](llama/kernels/projection.cu).

## Sources

The Llama control fetches pinned [HazyResearch Megakernels](https://github.com/HazyResearch/Megakernels)
and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens); their licenses
apply. [Measured source hashes](llama/tools/source_inventory.py).
DSpark derives from [DeepSpec](https://github.com/deepseek-ai/DeepSpec) and retains
its [license](dspark/LICENSE) and [notices](dspark/NOTICE),
which apply to that subtree. Model weights are obtained separately.
