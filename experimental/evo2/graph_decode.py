"""Evo 2 7B batch-1 greedy decode: Arc's vortex step captured as one CUDA graph."""

import argparse
import json
import random
import time

import torch
from evo2 import Evo2
from vortex.model.model import HyenaCascade

STATE_DICTS = ("fir_state_dict", "fir_inner_state_dict", "state_dict")
_sequential_forward = HyenaCascade.sequential_forward


def sequential_forward_in_place(self, u, inference_params):
    """Keep Hyena state at fixed addresses so graph replays see each update."""
    owned = [d for d in STATE_DICTS if self.layer_idx in getattr(inference_params, d, {})]
    buffers = {d: getattr(inference_params, d)[self.layer_idx] for d in owned}
    y, inference_params = _sequential_forward(self, u, inference_params)
    for d, buf in buffers.items():
        buf.copy_(getattr(inference_params, d)[self.layer_idx])
        getattr(inference_params, d)[self.layer_idx] = buf
    return y, inference_params


def prompt(n):
    random.seed(n)
    return "".join(random.choice("ACGT") for _ in range(n))


def state_tensors(params):
    mha = params["mha"]
    out = list(mha.key_value_memory_dict.values()) + [mha.lengths_per_sample]
    for name in ("hcl", "hcm", "hcs"):
        for d in STATE_DICTS:
            out += list(getattr(params[name], d, {}).values())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="evo2_7b.pt from arcinstitute/evo2_7b")
    parser.add_argument("--context", type=int, default=1024, help="Prompt bases (seeded random ACGT)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--parallel-prefill", type=int, default=8192, help="Bases prefilled in one forward")
    parser.add_argument("--check-tokens", type=int, default=200)
    parser.add_argument("--write-prompt", help="Also write the prompt here, for hyperserve --sequence")
    args = parser.parse_args()

    seq = prompt(args.context)
    if args.write_prompt:
        with open(args.write_prompt, "w") as f:
            f.write(seq + "\n")

    HyenaCascade.sequential_forward = sequential_forward_in_place
    torch.set_grad_enabled(False)
    evo2 = Evo2("evo2_7b", local_path=args.weights)
    model, dev = evo2.model, torch.device("cuda:0")
    ids = torch.tensor(evo2.tokenizer.tokenize(seq), dtype=torch.long, device=dev)

    head = min(args.context, args.parallel_prefill)
    params = model.initialize_inference_params(max_seqlen=args.context + args.steps + args.check_tokens + 8)
    logits, params = model(ids[None, :head], inference_params_dict=params)
    for p in params.values():
        p.seqlen_offset = head
    params["mha"].lengths_per_sample = torch.full((1,), head, dtype=torch.int32, device=dev)
    x = logits[:, -1].argmax(-1).view(1, 1)

    def step():
        out, _ = model(x, inference_params_dict=params)
        params["mha"].lengths_per_sample += 1
        x.copy_(out[:, -1].argmax(-1).view(1, 1))

    tensors = state_tensors(params)
    saved = [t.clone() for t in tensors] + [x.clone()]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for t, s in zip(tensors + [x], saved):
        t.copy_(s)

    for i in range(head, args.context):
        x.copy_(ids[i].view(1, 1))
        graph.replay()
    assert int(params["mha"].lengths_per_sample) == args.context

    tensors = state_tensors(params)
    saved = [t.clone() for t in tensors] + [x.clone()]
    graph_tokens = []
    for _ in range(args.check_tokens):
        graph.replay()
        graph_tokens.append(int(x))
    for t, s in zip(tensors + [x], saved):
        t.copy_(s)
    eager_tokens = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.check_tokens):
        step()
        eager_tokens.append(int(x))
    eager_ms = (time.perf_counter() - start) * 1e3 / args.check_tokens
    for t, s in zip(tensors + [x], saved):
        t.copy_(s)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        graph.replay()
        int(x)
    graph_ms = (time.perf_counter() - start) * 1e3 / args.steps

    result = {
        "context": args.context,
        "steps": args.steps,
        "graph_ms_per_token": round(graph_ms, 3),
        "eager_ms_per_token": round(eager_ms, 3),
        "greedy_tokens_matching_eager": f"{sum(a == b for a, b in zip(graph_tokens, eager_tokens))}/{args.check_tokens}",
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
