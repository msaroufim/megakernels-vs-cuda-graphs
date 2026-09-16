# H100 hardware replication, revision 1

Declared before any H100 timing or reference outputs in this replication.
This supplements the retained original protocol and the H200-derived hardware
supplement, overriding hardware, lane assignment and prior-observation language.

Use one full **H100 SXM with 132 SMs**, compute capability 9.0. H100 PCIe with
114 SMs and partitioned devices are ineligible: the original upstream Hopper
scheduler is compiled for 132 SMs. Build pristine Hazy Megakernels and
ThunderKittens at the same commits using `GPU=H100`; make no upstream patches.

Keep the conventional kernel source, portable BF16 conversion, FP32 accumulation,
PDL, graph boundaries, model hashes, prompt texts, seed rule, timing procedure,
reference implementation, numerical thresholds and bootstrap estimator identical
to the H200 replication. No H100-specific performance tuning is allowed in this
hardware comparison. Newly generated wrappers bind their strict hardware guards,
manifest names, campaign labels and freeze keys to H100. Preserve original
source hashes and record rendered wrapper hashes separately.

The eight validation prompts have already been examined on H200 and GB300;
they are a fixed cross-hardware workload, not an unseen test set. Generate all
FP32/BF16 reference tensors freshly on H100 after freezing the candidate. Do not
reuse H200 build metadata, numerical fixtures or performance measurements.
A prior H100 development smoke may only diagnose portability/harness defects;
record any repair and freeze before the full campaign.

Run the 24 paired graph processes and three native diagnostics on the same
physical H100 UUID. Then run all 27 independent actual-output checkers serially
on that same device. The three logical checking lanes do not allocate more GPUs.
No timing overlaps checking or another GPU workload. Record model, software,
GPU variant, memory size, driver, clocks, power and thermal telemetry. Keep
artifacts outside Git on durable storage; budget at least 150 GiB free space.

The primary table reports the eight-prompt graph comparison with the existing
hierarchical prompt/process interval; native diagnostics remain separate. Report
numerical fidelity and timing independently if either arm misses the unchanged
symmetric criteria. Publish H100 results separately from H200 and DSpark/GB300.

This reproduces the original Hazy Hopper implementation against our conventional
baseline on H100. It does not recreate the historical blog's exact environment,
vLLM/SGLang baseline or headline speedup unless those are separately reproduced.
