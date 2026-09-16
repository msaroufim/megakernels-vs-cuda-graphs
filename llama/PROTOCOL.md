# H200 hardware replication, revision 1

The hypothesis and all numerical, timing, realized-output, freeze and estimator
requirements in [the original protocol](docs/PROTOCOL.md) apply.
This document declares the changes before H200 reserved outcomes. It is hashed
into the new runtime freeze together with the unchanged original protocol.

## Architecture and implementation

Require exactly one visible NVIDIA H200, Hopper capability 9.0 and 132 SMs per
worker. Build the unchanged publication-era Hopper source at Megakernels
`7309cec801537b61fea3b50d7dfe454a6cde578e` with its pinned ThunderKittens
`664c108d16f12707a73d3072ab525f26fb2b4f62`, using the authors' `GPU=H100` target.
Record complete source and binary hashes, image, package versions, driver and
compiler. Never reuse the GB300 control binary or its compatibility changes.
Use the CUDA 12.8 NGC PyTorch 25.03 image with Transformers 4.48.3 and Triton
3.4.0 in an isolated environment. The explicit Triton pin preserves PDL launcher
support; do not replace the image's PyTorch while resolving dependencies.

Use the original conventional candidate with its existing portable BF16-to-FP32
conversion and FP32 FMA path (`native_bf16_fma=False`). PDL, attention tiling,
fast math, graph scope and shared selector/embedding stay at original settings.
No PTX/SASS optimization variants are part of this comparison. Extensions compile for the visible device; an architecture environment override
is rejected. Runtime changes and generated source hashes are explicitly frozen.

The newer public `mk-v2-llama` control at `ebf5bc0` is Blackwell-specific and is
not an H200 arm. This is an architecture eligibility decision made before any
H200 outcomes, not performance-based omission. Porting it would require a
separate study and cannot establish unchanged-upstream H200 performance.

## Inputs, coverage and freezing

Reuse the exact checkpoint/tokenizer hashes and original four development/eight
validation prompt texts as a hardware replication. The original validation
suite was observed on GB300, so it is not a wholly new unseen workload suite.
Its H200 outcomes remain unavailable for candidate selection before the freeze.
Generate new H200 FP32/BF16 reference tensors. Do not copy the GB300 tensor
fixtures. Development runs may fix portability or harness defects; record those
changes and freeze again before any reserved H200 measurements.

Primary evaluation is one legacy-control graph comparison, eight prompts and
three fresh processes per prompt: 24 paired processes. Each process uses five
warmup sequences and ten measured sequences per arm, with the existing balanced
AB/BA seed rule `314159 + 100 * prompt_index + replicate`. The mandatory native
diagnostic uses the first validation prompt and three fresh processes. Thus the
complete campaign contains 27 paired processes and 864 inspected actual
executions, plus untimed canonical checks. Do not pool native and graph results.

Every paired process runs both arms serially on the same physical GPU. Run all
27 paired processes on GPU0: the unchanged statistical summarizer groups by
physical GPU UUID, so one device must cover the complete prompt suite. Assign
independent reference/checking work to GPU1, GPU2 and GPU3. Use a central UUID/ownership ledger, isolated
compiler caches, and record processes and GPU telemetry at run boundaries. No
reference or other work may overlap a measured GPU. Any capacity-driven change
to lane assignment must be declared before reserved execution, preserving
same-GPU pairing.
After GPU0 completes all its paired measurements, its owner may use that GPU
for independent realized-history validation. Record that transition in the
ledger before starting checker work; never overlap checking with measurement.

Freeze the new candidate, staged harness, protocol, campaign, upstream
source/binary manifests and reference generator before reserved generation and
measurement. Preserve raw timings, every token stream and actual final logits,
and independently HF-replay every distinct realized history. Freeze violations,
nonfinite outputs, numerical failures and interrupted runs remain visible.

## Interpretation

Use the original hierarchical prompt/process bootstrap of paired log latency
ratios. A claimed speed win requires the 95% interval to exclude parity and
**both arms** to pass every canonical and actual-execution numerical criterion.
Do not loosen the native-BF16-calibrated KL or absolute-NLL requirements after
observing results. If either arm fails, report measured latency descriptively
and numerical acceptance separately; do not certify a paired win.

Keep H200 and GB300 tables separate. Their architecture-specific implementations
and toolchains differ. A result on this workload cannot prove that megakernels
are always better or unnecessary in general.
