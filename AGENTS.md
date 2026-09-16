# Repository guide

Keep the Llama comparison runnable on one H100/H200 with public dependencies
and user-supplied weights. DSpark replay requires external artifacts; document
that limitation. Keep weights, generated data, binaries and caches out of Git.

Preserve measured `llama/kernels/`, `llama/benchmarks/`, `llama/reference/` and
`llama/docs/PROTOCOL.md` bytes. New algorithms or numerical rules need a separate
study. Fetch upstream Hazy code at its pinned commits without patches.

Never overlap GPU timing with checking or other workloads. Report numerical
fidelity separately from latency; feature-level speedups require ablations.

Each study has its own `uv` environment. Run formatting, lint, type checks and tests from
[the reproduction guide](llama/README.md#cpu-checks). DSpark has its own
[environment](dspark/README.md#cpu-checks).

Keep `main` as one parentless commit. Amend that commit for updates and push with
an explicit `--force-with-lease`; never publish backup refs or old history.
Keep the repository private unless the user asks to change visibility.
