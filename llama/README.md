# Run the Llama comparison

To improve either baseline, start with the [competitor guide](AGENTS.md):
edit `kernels/` or `megakernel/`, validate and submit a PR.

## Requirements

- One full **H100 SXM or H200 with 132 SMs**, Linux and an NVIDIA container
  runtime. The original Hazy scheduler assumes 132 SMs; H100 PCIe, partitioned
  GPUs and Blackwell are rejected.
- `nvcr.io/nvidia/pytorch:25.03-py3` (CUDA 12.8). The build keeps the image's
  PyTorch and installs Triton 3.4.0 and Transformers 4.48.3 privately.
- `meta-llama/Llama-3.2-1B-Instruct`, obtained under its model terms.
  [Exact file hashes](upstream/prepare.py) are checked before running.
- At least 150 GiB of storage outside the checkout. Allow several hours for
  the full evaluation and reserve the GPU exclusively during measurement.

## Run both baselines

Clone on the GPU host and download the model using your Hugging Face access:

```bash
git clone https://github.com/msaroufim/megakernels-vs-cuda-graphs.git
cd megakernels-vs-cuda-graphs
uvx --from huggingface_hub==0.36.2 hf download \
  meta-llama/Llama-3.2-1B-Instruct \
  config.json model.safetensors tokenizer.json tokenizer_config.json \
  --local-dir /absolute/path/to/model
nvidia-smi --query-gpu=uuid,name --format=csv
```

The download resolves the current model revision; the runner rejects files
whose hashes differ from the measured checkpoint.

```bash
mkdir -p /absolute/path/to/experiments
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace/baselines:ro \
  -v /absolute/path/to/model:/model:ro \
  -v /absolute/path/to/experiments:/experiments \
  -w /workspace/baselines/llama \
  nvcr.io/nvidia/pytorch:25.03-py3 bash
```

Inside the container, set the physical GPU UUID and choose `H100` or `H200`:

```bash
export GPU_TYPE=H100
export GPU_UUID=GPU-REPLACE-WITH-YOUR-UUID
python -m pip install uv==0.8.22
git config --global --add safe.directory /workspace/baselines
env -u TORCH_CUDA_ARCH_LIST uv run --no-project python reproduce.py \
  --model /model --workdir /experiments/smoke-001 \
  --gpu-type "$GPU_TYPE" --gpu-uuid "$GPU_UUID" --suite smoke
```

The smoke builds both implementations from this checkout, generates development
references and checks one paired run. Run the full comparison in a fresh directory:

```bash
env -u TORCH_CUDA_ARCH_LIST uv run --no-project python reproduce.py \
  --model /model --workdir /experiments/full-001 \
  --gpu-type "$GPU_TYPE" --gpu-uuid "$GPU_UUID" --suite full
```

The full run freezes sources, generates fresh references, runs 27 paired
processes and checks all outputs serially on the same GPU. Builds from another
GPU family are rejected. Always rebuild in a new work directory after changing
either kernel; do not reuse an old `--build`. Logs, tensors, hashes and results
stay outside Git. Execution errors stop the runner; numerical rejection is
recorded in the results.

```bash
uv run --no-project python tools/report.py --workdir /experiments/full-001
```

[Results and method](docs/RESULTS.md).

## PR performance checks

Changes to Llama kernels run both base and PR versions on one Modal H100.
The check alternates three process pairs, each with five warmups and ten timed
127-step sequences on `dev_weather`. It posts latency, paired t intervals
and independent numerical checks in one updated PR comment. These are development
checks, not a replacement for the full evaluation. Review and merge are manual.

The workflow uses the base branch's harness. Proposed code runs without credentials
or network access; model weights mount read-only. Both CI and the local runner
compile `kernels/` and the vendored `megakernel/` sources, using the pinned Hazy
Python runtime and ThunderKittens. See [source attribution](megakernel/ORIGIN.md).

Maintainers configure `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the
`llama-benchmarks` GitHub environment using the GPU MODE account. Cache weights once
with Hugging Face access (`HF_TOKEN`) and Modal credentials:

```bash
uv run python ci/run_modal.py seed-weights
```

This creates the `megakernels-llama-weights` Modal volume and verifies all four model
file hashes. Same-repository PRs run automatically. For forks, a maintainer runs
**Llama performance → Run workflow** on `main` with the PR number after reviewing
the code that will access the weights. Raw logs and reports are retained as
artifacts for 14 days. New pushes cancel older runs. Each run has a 90-minute
sandbox limit.
The workflow becomes active after it lands on `main`.

## CPU checks

Run from `llama/`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check reproduce.py tools upstream/prepare.py upstream/vendor.py campaign.py ci
uv run pytest -q
```

Measured kernel, benchmark and reference sources are excluded from formatting.
DSpark uses its own environment and [checks](../dspark/README.md#cpu-checks).
Weights, generated data and binaries stay outside Git.
[Source attribution](../README.md#sources).
