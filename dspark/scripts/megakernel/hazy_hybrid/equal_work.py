"""Capture the real SGLang confidence/STS work omitted by the retained greedy graft.

The caller supplies the pre-RMS BF16 HC output from the same reference head
invocation. This helper does not rerun HC, LM, Markov W2 or greedy sampling.
The original sampler already forms corrected logits; its selected IDs and
logits must be checked by the caller independently of this appended graph.
"""

import ast
import hashlib
import importlib
import inspect
import json
import textwrap
from pathlib import Path

_METHODS = {
    "forward": "e0dafa20b1a0a542f2fa11103a8cc27da5114cfcafdbdecd5e1609249e5f8a28",
    "apply_sts": "1867dacc8ce91dccf02b1304e493e511a7036803a59f8e21ee481190ee5a6cac",
    "get_prev_embeddings": "e874a19715d5b09b4cbc4eab959b72dc7bbb17f1aa97cb1e46584bde1659e2d6",
}


def _canonical(node):
    """Normalize AST without Python-version-dependent empty optional fields."""
    if isinstance(node, ast.AST):
        return {
            "node": type(node).__name__,
            **{
                key: _canonical(value)
                for key, value in ast.iter_fields(node)
                if value is not None and value != []
            },
        }
    if isinstance(node, list):
        return [_canonical(value) for value in node]
    return node


def _source(method):
    """Qualify the exact retained numerical methods and retain their live source."""
    source = textwrap.dedent(inspect.getsource(method))
    node = ast.parse(source).body[0]
    digest = hashlib.sha256(
        json.dumps(_canonical(node), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != _METHODS[method.__name__]:
        raise ValueError(f"Reference confidence source changed: {method.__name__}")
    path = Path(inspect.getsourcefile(method))
    return {
        "file": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ast_sha256": digest,
    }


def _identity(value):
    """Record owners and exact views without reading any CUDA values."""
    storage = value.untyped_storage()
    return (
        id(value),
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
        value.storage_offset(),
        storage.data_ptr(),
        storage.nbytes(),
    )


def _raw(torch, value):
    """Read small admission values outside capture and timing."""
    return value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


class ConfidenceBinding:
    """Own one appended graph, its real model parameters and confidence outputs."""

    def __init__(
        self, torch, graft, candidates, *, head_hidden, sts_temperatures=None, snapshot=None
    ):
        """Capture the live head or an independent trained head omitted by STATIC mode."""
        self.torch, self.graft = torch, graft
        self.model = graft.bundle.draft_model
        self.model_head_at_admission = self.model.confidence_head
        self.head, self.markov = self.model_head_at_admission, self.model.markov_head
        materialization = {"kind": "existing_model_head", "model_head_omitted": False}
        if self.head is None:
            self.head, materialization = load_static_reference_head(
                torch, snapshot, head_hidden.device
            )
        self.head_at_admission = self.head
        if not self.head.with_markov:
            raise ValueError("Equal work requires the real Markov-conditioned confidence head")
        if self.head.proj.bias is not None or tuple(self.head.proj.weight.shape) != (1, 4352):
            raise ValueError("Expected the unbiased 4096+256 confidence projection")
        if (
            self.head.proj.weight.dtype != torch.float32
            or type(self.head.proj) is not torch.nn.Linear
        ):
            raise ValueError("Expected the actual FP32 linear confidence projection")
        if (
            tuple(head_hidden.shape) != (5, 4096)
            or head_hidden.dtype != torch.bfloat16
            or tuple(candidates.shape) != (6,)
            or candidates.dtype != torch.int64
            or not head_hidden.is_contiguous()
            or not candidates.is_contiguous()
        ):
            raise ValueError(
                "Expected BF16 pre-RMS hidden [5,4096] and int64 IDs [anchor,5 drafts]"
            )
        device = head_hidden.device
        if not head_hidden.is_cuda or candidates.device != device:
            raise ValueError("Confidence capture requires one CUDA device")
        if (
            int(self.markov.markov_rank) != 256
            or self.markov.markov_w1.weight.dtype != torch.bfloat16
        ):
            raise ValueError("Expected the retained BF16 rank-256 Markov embedding")
        if int(self.markov.markov_w1.tp_size) != 1:
            raise ValueError("Equal-work confidence capture requires TP1 embeddings")
        self.head_hidden, self.candidates = head_hidden, candidates
        self.native_temperatures = (
            torch.ones(5, dtype=torch.float32, device=device)
            if sts_temperatures is None
            else sts_temperatures
        )
        actual = self.head.sts_temperatures
        if (
            self.native_temperatures.dtype != torch.float32
            or tuple(self.native_temperatures.shape) != (5,)
            or actual.dtype != torch.float32
            or actual.numel() not in (1, 5)
            or actual.device != device
            or not torch.equal(actual.reshape(-1).expand(5), self.native_temperatures)
            or not torch.equal(self.native_temperatures, torch.ones_like(self.native_temperatures))
        ):
            raise ValueError("Actual SGLang STS must match the native five unit temperatures")
        self.owners = (
            head_hidden,
            candidates,
            self.head.proj.weight,
            actual,
            self.markov.markov_w1.weight,
            self.native_temperatures,
        )
        if any(value.device != device for value in self.owners):
            raise ValueError("Confidence owners span devices")
        self.identities = tuple(_identity(value) for value in self.owners)
        self.methods = (
            self.head.forward.__func__,
            self.head.apply_sts.__func__,
            self.markov.get_prev_embeddings.__func__,
        )
        self.receipt = {
            "scope": "Actual SGLang confidence/STS; pre-RMS HC supplied by original head",
            "head_materialization": materialization,
            "source": {method.__name__: _source(method) for method in self.methods},
            "owners": self.identities,
            "confidence_weight_sha256": hashlib.sha256(
                _raw(torch, self.head.proj.weight)
            ).hexdigest(),
            "sts_temperatures": self.native_temperatures.cpu().tolist(),
            "previous_tokens": "candidates[0:5]: anchor followed by the first four sampled tokens",
            "raw_shape": [1, 5],
            "calibrated_shape": [1, 5],
            "extra_hc_or_markov_w2": False,
            "native_reduction_bitwise_equivalence_claimed": False,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        with torch.inference_mode():
            for _ in range(2):
                self.evaluate(head_hidden, candidates[:5])
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.graph):
                self.outputs = self.evaluate(head_hidden, candidates[:5])
        self.verify_host()

    def verify_host(self):
        """Reject changed model owners or views without synchronizing the device."""
        current = (
            self.head_hidden,
            self.candidates,
            self.head.proj.weight,
            self.head.sts_temperatures,
            self.markov.markov_w1.weight,
            self.native_temperatures,
        )
        methods = (
            self.head.forward.__func__,
            self.head.apply_sts.__func__,
            self.markov.get_prev_embeddings.__func__,
        )
        if (
            self.model is not self.graft.bundle.draft_model
            or self.model.confidence_head is not self.model_head_at_admission
            or self.head is not self.head_at_admission
            or self.markov is not self.model.markov_head
            or tuple(_identity(value) for value in current) != self.identities
            or methods != self.methods
        ):
            raise ValueError("Confidence owners or source bindings changed")

    def evaluate(self, head_hidden, previous_tokens):
        """Run the same reference on supplied inputs outside timing for native-input validation."""
        if tuple(head_hidden.shape) != (5, 4096) or previous_tokens.numel() != 5:
            raise ValueError("Confidence validation requires five hidden rows and previous tokens")
        with self.torch.inference_mode():
            embedding = self.markov.get_prev_embeddings(previous_tokens.reshape(1, 5))
            raw = self.head(head_hidden.reshape(1, 5, 4096), embedding)
            calibrated = self.head.apply_sts(raw)
        return {
            "confidence_logits": raw,
            "calibrated_confidences": calibrated,
            "previous_embeddings": embedding,
        }

    def verify_native_parameters(self, band):
        """Compare actual native arena bytes and STS with the reference outside timing."""
        self.verify_host()
        torch, packed = self.torch, band.keepalive[0]
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Native parameter comparison must precede capture/timing")
        result = {}
        for suffix, expected, dtype in (
            ("confidence_head.proj.weight", self.head.proj.weight, "float32"),
            ("markov_head.markov_w1.weight", self.markov.markov_w1.weight, "bfloat16"),
        ):
            entries = [e for e in packed.plan.entries if e.name.endswith("mtp.2." + suffix)]
            if len(entries) != 1:
                raise ValueError(f"Native final-layer parameter is ambiguous: {suffix}")
            entry = entries[0]
            offset = int(packed.offsets[entry.offset_index].item())
            if (
                entry.dtype != dtype
                or tuple(expected.shape) != tuple(entry.shape)
                or offset != entry.arena_offset
                or offset < 0
                or offset + entry.nbytes > packed.arena.numel()
            ):
                raise ValueError(f"Native confidence parameter layout changed: {suffix}")
            native = packed.arena[offset : offset + entry.nbytes]
            actual = expected.detach().contiguous().reshape(-1).view(torch.uint8)
            if not torch.equal(native, actual):
                raise ValueError(f"Native/reference parameter bytes differ: {suffix}")
            result[suffix] = {
                "shape": list(entry.shape),
                "bytes": entry.nbytes,
                "offset_index": entry.offset_index,
                "arena_offset": offset,
                "bitwise_equal": True,
            }
        temperatures = band.keepalive[14]
        if not torch.equal(temperatures, self.native_temperatures):
            raise ValueError("Native/reference STS temperatures differ")
        result["sts_temperatures_equal"] = True
        return result

    def evaluate_fp64(self, head_hidden, previous_tokens):
        """Provide a CPU double dot oracle and a conservative native FP32 reduction bound."""
        torch = self.torch
        with torch.inference_mode():
            embedding = self.markov.get_prev_embeddings(previous_tokens.reshape(1, 5))
            features = torch.cat(
                (
                    head_hidden.reshape(5, 4096).double().cpu(),
                    embedding.reshape(5, 256).double().cpu(),
                ),
                dim=-1,
            )
            products = features * self.head.proj.weight.detach().double().cpu()
            raw = products.sum(-1).reshape(1, 5)
            # Native: 17 ordered FMAs per lane, then eight reduction levels.
            # 32*u is conservative; products of BF16 and FP32 fit exactly in FP64.
            unit = 2.0**-24
            bound = (32 * unit / (1 - 32 * unit)) * products.abs().sum(-1).reshape(1, 5)
            calibrated = torch.sigmoid(raw / self.native_temperatures.double().cpu())
        return {
            "confidence_logits": raw,
            "calibrated_confidences": calibrated,
            "native_fp32_absolute_error_bound": bound,
        }

    def record(self):
        """Return admission/source metadata without a resident read."""
        self.verify_host()
        return self.receipt

    def evidence(self):
        """Clone actual captured outputs before another reference evaluation reuses model state."""
        self.verify_host()
        return {name: value.clone() for name, value in self.outputs.items()}


def load_static_reference_head(torch, snapshot, device):
    """Load the trained reference head without changing the STATIC model or global mode."""
    if snapshot is None:
        raise ValueError("STATIC confidence materialization requires the checkpoint snapshot")
    v4 = importlib.import_module("sglang.srt.models.deepseek_v4_dspark")
    dspark = importlib.import_module("sglang.srt.models.dspark")
    loader = importlib.import_module("deepspec.megakernel.full_loop_graft")._load_global
    mode = v4.read_ragged_verify_mode()
    if mode is not v4.RaggedVerifyMode.STATIC:
        raise ValueError("An omitted confidence head is supported only in actual STATIC mode")
    snapshot = Path(snapshot)
    key = "mtp.2.confidence_head.proj.weight"
    index_path = snapshot / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    shard = json.loads(index_bytes)["weight_map"][key]
    weight = loader(snapshot, key)
    if tuple(weight.shape) != (1, 4352) or weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Trained confidence checkpoint must be BF16 or FP32 [1,4352]")
    if weight.device != device or not bool(torch.isfinite(weight).all()):
        raise ValueError("Trained confidence checkpoint has the wrong device or nonfinite values")
    head = dspark.DSparkConfidenceHead(
        hidden_size=4096, markov_rank=256, with_markov=True, bias=False, dtype=torch.float32
    ).to(device=device)
    with torch.no_grad():
        head.proj.weight.copy_(weight)
    head.eval()
    provenance = {}
    for name, function in (
        ("static_factory", v4.build_dspark_v4_confidence_head),
        ("constructor", dspark.DSparkConfidenceHead.__init__),
        ("checkpoint_loader", loader),
    ):
        source = textwrap.dedent(inspect.getsource(function))
        path = Path(inspect.getsourcefile(function))
        provenance[name] = {
            "file": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    return head, {
        "kind": "independent_trained_reference_head",
        "model_head_omitted": True,
        "original_mode": str(mode),
        "reason": "The SGLang V4 factory returns None in STATIC mode; global mode is unchanged",
        "snapshot": str(snapshot),
        "checkpoint_key": key,
        "checkpoint_shard": shard,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "checkpoint_weight_dtype": str(weight.dtype),
        "checkpoint_weight_sha256": hashlib.sha256(_raw(torch, weight)).hexdigest(),
        "loaded_fp32_weight_sha256": hashlib.sha256(_raw(torch, head.proj.weight)).hexdigest(),
        "source": provenance,
    }


def capture_confidence(
    torch, graft, candidates, *, head_hidden, sts_temperatures=None, snapshot=None
):
    """Build one confidence-only graph for composition after the unmodified greedy sampler."""
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("Capture confidence separately before composing the measured graph")
    return ConfidenceBinding(
        torch,
        graft,
        candidates,
        head_hidden=head_hidden,
        sts_temperatures=sts_temperatures,
        snapshot=snapshot,
    )
