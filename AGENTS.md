# Repository guide

Keep the Llama comparison runnable on one H100/H200 with public dependencies
and user-supplied weights. DSpark replay requires external artifacts; document
that limitation. Keep weights, generated data, binaries and caches out of Git.
For Llama submissions, follow [the competitor guide](llama/AGENTS.md).

Keep historical result claims tied to their recorded source hashes. Kernel changes
are allowed and measured by PR CI; changes to the benchmark, reference or numerical
rules require a separate study. Edit `llama/kernels/` or `llama/megakernel/`
directly; local runs and CI compile those sources. Preserve the vendored
[attribution and license](llama/megakernel/ORIGIN.md). Historical Llama results
used the original, unmodified Hazy code.

Never overlap GPU timing with checking or other workloads. Report numerical
fidelity separately from latency; feature-level speedups require ablations.

Each study has its own `uv` environment. Run formatting, lint, type checks and tests from
[the reproduction guide](llama/README.md#cpu-checks). DSpark has its own
[environment](dspark/README.md#cpu-checks).

Contributors work on branches and submit PRs; do not push to `main`.
Maintainers keep `main` as one parentless commit: amend it for approved updates
and push with an explicit `--force-with-lease`; never publish backup refs or old history.
Keep the repository private unless the user asks to change visibility.
