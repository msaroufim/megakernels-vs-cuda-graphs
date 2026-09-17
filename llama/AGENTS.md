# Improving the Llama 1B baselines

Optimize either implementation and submit a PR with the change, measured latency
and numerical impact. Maintainers review the code and merge manually.

## Where to edit

| Baseline | Entry points |
|---|---|
| CUDA Graphs + PDL | `kernels/candidate.py` builds the decode step; `kernels/projection.cu` implements fused projections/reductions; `kernels/attention.py` implements Triton attention |
| Hazy megakernel | `megakernel/llama.cu`, `megakernel/llama.cuh`, sibling operators and `megakernel/include/` runtime headers |

Paths are relative to `llama/`. Edit either directory directly in your fork.
Keep changes to each baseline separate when possible so their effects are clear.

## CUDA Graphs + PDL interface

Preserve `prepare(g, *, hf_config=..., options=...)`, `build(g, position, prepared)`
and `output_logits(g, prepared)` in `candidate.py`. `prepare` compiles, allocates
scratch and packs immutable weights. `build` returns a zero-argument callable
that reads the current hidden state, updates the KV cache and computes logits
for one full decode step. The harness owns embedding, token selection and graph
capture. Keep the callable capture-safe and free of CPU/GPU synchronization.

Use the Hopper runner: staging disables the Blackwell-only `native_bf16_fma`
option. Preserve the existing option declarations. CI copies `.py`, `.cu`,
`.cuh`, `.h` and `.hpp` files from `kernels/`; other new dependencies need a
separate harness change.

## Megakernel interface

Preserve the `mk_llama` Python binding and Globals/schedule interface used by the
harness. The CUDA sources, Makefile and shared runtime headers are vendored in
`megakernel/`; retain their [attribution and license](megakernel/ORIGIN.md).

Both local `reproduce.py` and CI fetch the pinned Hazy Python runtime and
ThunderKittens, overlay this checkout's megakernel sources and compile with
`make GPU=H100`. Follow the [run recipe](README.md#run-both-baselines) to test
locally. Rebuild in a fresh work directory after source changes; an old `--build`
does not include your edits.

## Validate and submit

Run the [CPU checks](README.md#cpu-checks), then open a PR. Once the CI workflow
lands on `main`, same-repository PRs get a one-H100 base/head comparison; forks
need a maintainer to dispatch it. Contributors do not need the shared Modal
credentials. See [PR performance checks](README.md#pr-performance-checks).

Keep the model, prompts, harness, references and numerical rules fixed. Setup may
precompute weight layouts, but not prompt-dependent activations or outputs. Do
not specialize to benchmark answers, bypass work during timing or relax checks.
Run GPU timing without other workloads. Report numerical fidelity separately
from speed; the one-prompt CI check does not replace the full evaluation.

Keep weights, binaries and generated results out of Git. Include the mechanism,
benchmark comment and any numerical tradeoff in the PR; leave merging to a human.
