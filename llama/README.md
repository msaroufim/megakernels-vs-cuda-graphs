# Run the Llama comparison

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

The smoke builds both implementations, generates development references and
checks one paired run. For the full comparison, reuse that build with a fresh
work directory:

```bash
env -u TORCH_CUDA_ARCH_LIST uv run --no-project python reproduce.py \
  --model /model --workdir /experiments/full-001 \
  --gpu-type "$GPU_TYPE" --gpu-uuid "$GPU_UUID" --suite full \
  --build /experiments/smoke-001/build
```

The full run freezes sources, generates fresh references, runs 27 paired
processes and checks all outputs serially on the same GPU. Builds from another
GPU family are rejected. Use a new work directory for each run; logs, tensors,
hashes and results stay there. Setup/execution errors stop the runner;
numerical rejection is recorded in the results.

```bash
uv run --no-project python tools/report.py --workdir /experiments/full-001
```

[Results and method](docs/RESULTS.md).

## CPU checks

Run from `llama/`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check reproduce.py tools upstream/prepare.py campaign.py
uv run pytest -q
```

Measured kernel, benchmark and reference sources are excluded from formatting.
DSpark uses its own environment and [checks](../dspark/README.md#cpu-checks).
Weights, generated data and binaries stay outside Git.
[Source attribution](../README.md#sources).
