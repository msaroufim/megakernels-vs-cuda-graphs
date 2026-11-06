# Experimental: Evo 2 7B decode, Hyperkernel megakernel vs CUDA Graphs

**Status: one-session example, not a validated result.** CUDA Graphs without
PDL are within 0.5% of the megakernel at 1k context and faster at longer contexts.

| Context | Eager vortex | CUDA Graph | Hyperkernel megakernel |
|---:|---:|---:|---:|
| 1,024 | 25.31 ms | 6.37 ms | **6.34 ms** |
| 8,192 | 28.81 ms | **6.50 ms** | 7.53 ms |
| 65,536 | 29.16 ms | **7.38 ms** | 16.48 ms |

Milliseconds per greedy token, batch 1, one NVIDIA B300 SXM6 (driver 580.173.02),
1,000 decode steps after the same seeded random ACGT prompt, run sequentially on
the same GPU. Both arms read each token back to the host every step; eager uses
200 steps. Single runs on November 6, 2026 (UTC); no paired
alternation or repeat statistics. Prefill is not compared.

## Arms

- **CUDA Graph:** [`graph_decode.py`](graph_decode.py) captures Arc's
  [vortex](https://github.com/Zymrael/vortex) decode step (`evo2==0.6.0`)
  as one graph, including argmax and the position update. Two changes make it
  capture-safe: Hyena FIR/IIR state is copied back into fixed buffers, and the
  FlashAttention KV-cache position is a device tensor. One replay runs about
  1,540 kernels. There is no PDL: these are cuBLAS, PyTorch and FlashAttention
  kernels without `griddepcontrol`.
- **Megakernel:** [Hyperkernel](https://github.com/Hyperstitioncc/Hyperkernel)
  commit `14d1e14`, prebuilt `kernels/b300/evo2` bundles. Nsight Systems shows
  one `mega_main` launch per decode step. Its generator source is not public.

## Fidelity

Graph greedy tokens match eager vortex for 200 of 200 tokens at each context.
Against vortex on the first 8,192 bases of E. coli K-12 (NC_000913.3), the
megakernel's prompt logits agree on top-1 for 99.07% of positions, with mean
KL 5.1e-4 and next-base NLL 0.5163 versus 0.5161 nats.

## Run

Requires one B300, `pip install evo2==0.6.0` and `evo2_7b.pt` from
[`arcinstitute/evo2_7b`](https://huggingface.co/arcinstitute/evo2_7b).
`--write-prompt` saves the prompt for the megakernel run.

```sh
python graph_decode.py --weights evo2_7b.pt --context 8192 --write-prompt p8192.txt

# Hyperkernel: convert weights, then time decode with the same prompt.
python scripts/evo2_safetensors.py evo2_7b.pt evo2_7b
target/release/hyperserve dna --model evo2-7b --weights evo2_7b --sequence p8192.txt --steps 1000
```

Hyperkernel prints `step_ms` per token. Its prefill chunk is 1,024 bases;
leftover bases run through the decode kernel. At 65,536 bases, vortex's
parallel FFT prefill runs out of memory on one B300, so the graph arm prefills
8,192 bases in parallel and feeds the rest through the graph. Decode timing is
unaffected.
