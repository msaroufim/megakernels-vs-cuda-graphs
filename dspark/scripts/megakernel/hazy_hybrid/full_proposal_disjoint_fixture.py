"""Private fixed-fixture repair: preserve 128 prefix slots and reserve five draft slots.

This changes benchmark cache placement, hence establishes a new reference baseline.
It changes no captured operation, math, target injection, or timed Python path.
The caller must retain this binding and verify it after synchronized untimed replays.
"""

import hashlib
import importlib
import inspect
import json
from pathlib import Path

BACKEND_SHA = "57e6a41f005e4f28a12394addb028cb4777b0f05026a8cbcefc0319527f74ea7"
FIXTURE_SHA = "2346a1caac736584a8364dd6a541ed926f3636cc5394a82a945dccda67f874c8"
POOL_SHA = "cda2e39ff538f5bed654fdf038c2249ddb95c13e5c4784be9ea4dec43dd4aa31"
METHOD_SHA = {
    "get_swa_out_cache_loc": "dcc576c75c485fddceee6e885aa0b136ee2de7963955a9c2e5b31c6c3b116eca",
    "get_dspark_swa_page_indices": (
        "3d561efce45cc385e75190c97f4f1e1544f702ba492037d94faf21ba6c1cfe45"
    ),
}
PAGE_BYTES, PAGE_TOKENS, DATA_BYTES, SCALE_BYTES = 149760, 256, 576, 8
PREFIX_SLOTS, DRAFT_SLOTS = tuple(range(128)), tuple(range(128, 133))


def digest(data):
    """Hash immutable bytes for compact outward-facing evidence."""
    return hashlib.sha256(data).hexdigest()


def source_contract():
    """Admit the exact captured reference whose graph refreshes all live mappings."""
    fixture = importlib.import_module("fixture")
    backend = importlib.import_module("sglang.srt.layers.attention.deepseek_v4_backend")
    files = {"fixture": (fixture, FIXTURE_SHA), "backend": (backend, BACKEND_SHA)}
    receipt = {}
    for name, (module, expected) in files.items():
        path = Path(inspect.getsourcefile(module))
        actual = digest(path.read_bytes())
        if actual != expected:
            raise ValueError(f"Unsupported disjoint fixture {name} source: {actual}")
        receipt[name] = {"path": str(path), "sha256": actual}
    cls = backend.DeepseekV4AttnBackend
    receipt["methods"] = {}
    for name, expected in METHOD_SHA.items():
        source = inspect.getsource(getattr(cls, name))
        if digest(source.encode()) != expected:
            raise ValueError(f"Unsupported live mapping method {name}")
        receipt["methods"][name] = {"sha256": expected, "source": source}
    source = inspect.getsource(cls.init_forward_metadata_in_graph)
    for required in (
        "translate_loc_from_full_to_swa(out_cache_loc)",
        "self.get_dspark_swa_page_indices(",
        "out_loc=out_cache_loc",
        "metadata.core_attn_metadata.swa_page_indices = swa_page_indices",
    ):
        if required not in source:
            raise ValueError("Reference no longer refreshes captured draft metadata")
    receipt["methods"]["init_forward_metadata_in_graph"] = {
        "sha256": digest(source.encode()),
        "source": source,
    }
    return receipt


def identity(value):
    """Describe the retained allocation and view without reading device values."""
    return {
        "pointer": value.data_ptr(),
        "storage_pointer": value.untyped_storage().data_ptr(),
        "storage_bytes": value.untyped_storage().nbytes(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "offset": value.storage_offset(),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def require_tensor(torch, value, dtype, shape, device):
    """Reject unsupported owners before changing any live fixture bytes."""
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != dtype
        or tuple(value.shape) != tuple(shape)
        or value.device != device
        or value.device.type != "cuda"
        or not value.is_contiguous()
    ):
        raise ValueError("Unexpected fixed-fixture tensor layout/device")


def prefix_bytes(value):
    """Read only packed data/scales for physical prefix slots0..127, on the CPU."""
    page = value[0].detach().cpu().contiguous().numpy().tobytes()
    scale_start = PAGE_TOKENS * DATA_BYTES
    return page[: 128 * DATA_BYTES] + page[scale_start : scale_start + 128 * SCALE_BYTES]


def getter_source(pool):
    """Pin the actual pool getter before trusting its view of packed storage."""
    path = Path(inspect.getsourcefile(type(pool)))
    if digest(path.read_bytes()) != POOL_SHA:
        raise ValueError("Unsupported original packed pool implementation")
    method = pool.get_swa_key_buffer_radix
    source = inspect.getsource(method)
    delegate = pool.swa_kv_pool.get_key_buffer
    delegate_source = inspect.getsource(delegate)
    expected = "1023110136f870e955f9212fca62e2dd6b6217f1f7c26ca180babf7bf64477f9"
    if (
        digest(source.encode())
        != "fd69f691409c052a73abb7b476f53f3324a758ef34950f3c213a2ebce27dd07e"
        or digest(delegate_source.encode()) != expected
        or Path(inspect.getsourcefile(delegate)).resolve() != path.resolve()
    ):
        raise ValueError("Unsupported packed pool getter/delegate implementation")
    return {
        "path": str(path),
        "file_sha256": POOL_SHA,
        "method": method.__qualname__,
        "source_sha256": digest(source.encode()),
        "source": source,
        "delegate": {
            "method": delegate.__qualname__,
            "source_sha256": expected,
            "source": delegate_source,
        },
    }


class Binding:
    """Retain immutable admission evidence for one nonoverlapping fixed-state run."""

    def __init__(self, torch, graft, prepared):
        """Validate source, shape, values and free slots before the three mapping edits."""
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Disjoint fixture preparation must precede capture")
        self.torch, self.graft, self.prepared = torch, graft, prepared
        self.source = source_contract()
        if (
            prepared.metadata.get("prefix_len") != 130
            or prepared.metadata.get("proposal_seq_len") != 131
        ):
            raise ValueError("Disjoint fixture supports only the audited prefix130 fixture")
        if prepared.metadata.get("target_injection_timed") is not False:
            raise ValueError("Target injection must already have completed outside timing")
        self.runner = graft.bundle.draft_model_runner
        self.pool = self.runner.token_to_kv_pool
        self.req_pool = self.runner.req_to_token_pool
        self.buffers = graft.runner.buffers
        self.backend = graft.runner.attn_backend
        if (
            graft.runner.enable_pdmux
            or type(self.backend).__name__ != "DeepseekV4AttnBackend"
            or type(self.backend).__module__ != "sglang.srt.layers.attention.deepseek_v4_backend"
            or self.backend.token_to_kv_pool is not self.pool
            or self.backend.req_to_token is not self.req_pool.req_to_token
        ):
            raise ValueError("Captured runner does not use the admitted live mapping backend")
        if self.pool._unified_kv or self.pool.swa_kv_pool.page_size != PAGE_TOKENS:
            raise ValueError("Expected separate packed SWA storage with256-token pages")
        self.owners = self.live_owners()
        device = self.owners["out_cache_loc"].device
        req, full = self.owners["req_to_token"], self.owners["full_to_swa"]
        if req.ndim != 2 or req.shape[0] != 2 or req.shape[1] < 136:
            raise ValueError("Unexpected request mapping capacity")
        if full.ndim != 1 or full.numel() < 133:
            raise ValueError("Insufficient full-to-SWA mapping capacity")
        for name, dtype, shape in (
            ("req_to_token", torch.int32, req.shape),
            ("full_to_swa", torch.int64, full.shape),
            ("out_cache_loc", torch.int64, self.buffers.out_cache_loc.shape),
            ("positions", torch.int64, self.buffers.positions.shape),
            ("seq_lens", torch.int64, self.buffers.seq_lens.shape),
            ("req_pool_indices", torch.int64, self.buffers.req_pool_indices.shape),
        ):
            require_tensor(torch, self.owners[name], dtype, shape, device)
        pools = tuple(self.pool.swa_kv_pool.kv_buffer)
        if len(pools) != 3:
            raise ValueError("Expected exactly three distinct layer pools")
        for value in pools:
            require_tensor(torch, value, torch.uint8, (5, PAGE_BYTES), device)
        self.source["pool_getter"] = getter_source(self.pool)
        self.getter = self.pool.get_swa_key_buffer_radix
        self.reader_layout = self.storage_layout()
        spans = sorted(
            (v.data_ptr(), v.data_ptr() + v.numel() * v.element_size())
            for v in self.owners.values()
        )
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("Fixture controls or pools alias")
        self.owner_identity = {name: identity(v) for name, v in self.owners.items()}
        before = self.values()
        if (
            before["positions"] != list(range(131, 136))
            or before["seq_lens"] != [131]
            or before["req_pool_indices"] != [0]
            or before["out_cache_loc"] != [3, 4, 5, 6, 7]
            or before["req_to_token_used"] != [p % 128 for p in range(3, 136)]
            or before["full_to_swa_used"][:128] != list(PREFIX_SLOTS)
        ):
            raise ValueError("Fixture no longer has the measured compact-ring mapping")
        inject, _, prefix, commit, _, _ = prepared.keepalive
        if prefix.cpu().tolist() != [130] or commit.cpu().tolist() != [1]:
            raise ValueError("Expected exactly one already-injected target token")
        if inject.swa_loc.cpu().tolist() != [2, -1, -1, -1, -1, -1]:
            raise ValueError("Target injection does not match physical prefix slot2")
        self.prefix = tuple(prefix_bytes(v) for v in pools)
        self.before = before
        # All128 live prefix slots are retained. New slots are in the same
        # allocated page, outside the prefix; no allocator or pool grows.
        replacement = torch.tensor(DRAFT_SLOTS, dtype=torch.int64, device=device)
        old_full = full[128:133].clone()
        old_req = req[0, 131:136].clone()
        old_out = self.buffers.out_cache_loc[:5].clone()
        try:
            full[128:133].copy_(replacement)
            req[0, 131:136].copy_(replacement.to(torch.int32))
            self.buffers.out_cache_loc[:5].copy_(replacement)
            self.expected = dict(before)
            self.expected["full_to_swa_used"] = list(range(133))
            self.expected["req_to_token_used"] = [p % 128 for p in range(3, 131)] + list(
                DRAFT_SLOTS
            )
            self.expected["out_cache_loc"] = list(DRAFT_SLOTS)
            self.verify(require_metadata=False)
        except BaseException:
            full[128:133].copy_(old_full)
            req[0, 131:136].copy_(old_req)
            self.buffers.out_cache_loc[:5].copy_(old_out)
            torch.cuda.synchronize()
            raise

    def live_owners(self):
        """Resolve allocation owners again so a rebound buffer cannot pass validation."""
        runner = self.graft.bundle.draft_model_runner
        buffers = self.graft.runner.buffers
        pool = runner.token_to_kv_pool
        return {
            "req_to_token": runner.req_to_token_pool.req_to_token,
            "full_to_swa": pool.full_to_swa_index_mapping,
            **{
                name: getattr(buffers, name)
                for name in ("out_cache_loc", "positions", "seq_lens", "req_pool_indices")
            },
            **{f"swa{n}": v for n, v in enumerate(pool.swa_kv_pool.kv_buffer)},
        }

    def values(self):
        """Read the complete used mapping slices and live proposal scalars, untimed."""
        owners = self.live_owners()
        return {
            "req_to_token_used": owners["req_to_token"][0, 3:136].cpu().tolist(),
            "full_to_swa_used": owners["full_to_swa"][:133].cpu().tolist(),
            "out_cache_loc": owners["out_cache_loc"][:5].cpu().tolist(),
            "positions": owners["positions"][:5].cpu().tolist(),
            "seq_lens": owners["seq_lens"][:1].cpu().tolist(),
            "req_pool_indices": owners["req_pool_indices"][:1].cpu().tolist(),
        }

    def storage_layout(self):
        """Prove actual FlashMLA reader views and writer use the same256-token pages."""
        configuration = {
            "reader_page_tokens": self.pool.swa_window_size,
            "writer_page_tokens": self.pool.swa_kv_pool.page_size,
            "packed_dim_bytes": self.pool.swa_kv_pool.kv_cache_total_dim,
            "store_dtype": str(self.pool.swa_kv_pool.store_dtype),
            "reader_dtype": str(self.pool.swa_kv_pool.dtype),
        }
        if (
            self.pool.swa_window_size != PAGE_TOKENS
            or self.pool.swa_kv_pool.page_size != PAGE_TOKENS
            or self.pool.swa_kv_pool.kv_cache_total_dim != DATA_BYTES + SCALE_BYTES
            or self.pool.swa_kv_pool.store_dtype != self.torch.uint8
            or self.pool.swa_kv_pool.dtype != self.torch.float8_e4m3fn
            or self.pool.get_swa_key_buffer_radix != self.getter
        ):
            raise ValueError(
                "Packed KV reader/writer page geometry or getter changed: "
                + json.dumps(configuration, sort_keys=True)
            )
        stages = tuple(self.graft.bundle.draft_model.stages)
        if len(stages) != 3:
            raise ValueError("Expected three actual draft attention owners")
        result = []
        for n, stage in enumerate(stages):
            layer_id = stage.self_attn.layer_id
            if stage.self_attn.attn.layer_id != layer_id:
                raise ValueError("Attention and radix layer IDs disagree")
            raw = self.owners[f"swa{n}"]
            view = self.getter(layer_id)
            evidence = {
                "layer_id": layer_id,
                "raw": identity(raw),
                "getter": identity(view),
                **configuration,
            }
            width = PAGE_TOKENS * (DATA_BYTES + SCALE_BYTES)
            if (
                raw.dtype != self.torch.uint8
                or raw.element_size() != 1
                or view.dtype != self.torch.float8_e4m3fn
                or view.element_size() != 1
                or view.device != raw.device
                or view.ndim != 2
                or view.shape[0] != 5
                or not width <= view.shape[1] <= PAGE_BYTES
                or view.data_ptr() != raw.data_ptr()
                or tuple(view.stride()) != (PAGE_BYTES, 1)
                or view.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()
            ):
                raise ValueError(
                    "Original pool getter does not preserve packed page addresses: "
                    + json.dumps(evidence, sort_keys=True)
                )
            # This is the exact view in the pinned backend.forward. Page padding
            # makes the complete view noncontiguous; only its token rows are dense.
            forward = view[:, :width].view(5, PAGE_TOKENS, 1, DATA_BYTES + SCALE_BYTES)
            evidence["forward_view"] = identity(forward)
            if tuple(forward.stride()) != (
                PAGE_BYTES,
                DATA_BYTES + SCALE_BYTES,
                DATA_BYTES + SCALE_BYTES,
                1,
            ):
                raise ValueError(
                    "Unexpected FlashMLA packed cache view strides: "
                    + json.dumps(evidence, sort_keys=True)
                )
            result.append(
                {
                    **evidence,
                    "new_slot_last_data_byte": 133 * DATA_BYTES - 1,
                    "new_slot_last_scale_byte": 147456 + 133 * SCALE_BYTES - 1,
                }
            )
        return result

    def metadata(self):
        """Validate the actual replay-produced backend indices, not inferred map values."""
        if self.graft.runner.attn_backend is not self.backend:
            raise ValueError("Original selected attention backend changed")
        core = self.backend.forward_metadata.core_attn_metadata
        indices, lengths, output = (
            core.swa_page_indices,
            core.swa_topk_lengths,
            core.swa_out_cache_loc,
        )
        device = self.owners["out_cache_loc"].device
        require_tensor(self.torch, indices, self.torch.int32, (5, 192), device)
        require_tensor(self.torch, lengths, self.torch.int32, (5,), device)
        require_tensor(self.torch, output, self.torch.int32, (5,), device)
        expected = [*range(3, 128), 0, 1, 2, *DRAFT_SLOTS]
        rows = indices.cpu().tolist()
        if (
            lengths.cpu().tolist() != [133] * 5
            or output.cpu().tolist() != list(DRAFT_SLOTS)
            or any(row[:133] != expected for row in rows)
        ):
            raise ValueError("Original graph did not rebuild disjoint133-key attention metadata")
        return {
            "passed": True,
            "swa_topk_lengths": [133] * 5,
            "swa_out_cache_loc": list(DRAFT_SLOTS),
            "effective_indices_row": expected,
            "all_five_rows_equal": True,
            "unique_physical_keys_per_query": 133,
            "metadata_owners": {
                "indices": identity(indices),
                "lengths": identity(lengths),
                "output": identity(output),
            },
        }

    def verify(self, *, require_metadata=True):
        """Check preserved prefix bytes and exact mapping identity after an untimed sync."""
        if self.torch.cuda.is_current_stream_capturing():
            raise ValueError("Disjoint fixture verification is forbidden in capture")
        now = self.live_owners()
        if set(now) != set(self.owners) or any(
            now[n] is not owner or identity(owner) != self.owner_identity[n]
            for n, owner in self.owners.items()
        ):
            raise ValueError("Disjoint fixture owner or layout changed")
        if self.values() != self.expected:
            raise ValueError("Disjoint fixture live mapping or input scalar changed")
        if self.storage_layout() != self.reader_layout:
            raise ValueError("Original packed cache reader view changed")
        actual = tuple(prefix_bytes(now[f"swa{n}"]) for n in range(3))
        if actual != self.prefix:
            raise ValueError("Original128-token packed prefix was overwritten")
        return {
            "passed": True,
            "prefix_bytes_equal": True,
            "prefix_sha256": [digest(v) for v in actual],
            "prefix_bytes_per_layer": 128 * (DATA_BYTES + SCALE_BYTES),
            "mapping_values_equal": True,
            "owner_identity_equal": True,
            "draft_prefix_disjoint": True,
            "draft_swa_locations": list(DRAFT_SLOTS),
            "backend_metadata": (
                self.metadata() if require_metadata else {"deferred_until_original_replay": True}
            ),
        }

    def record(self):
        """Return compact JSON evidence while retaining raw prefix snapshots privately."""
        return {
            "schema": "dspark.disjoint_draft_kv_fixture.v1",
            "scope": "Corrected cache placement; new fixed-state reference baseline",
            "source": self.source,
            "storage_layout": self.reader_layout,
            "owners": self.owner_identity,
            "prefix_logical_ring_order": [128, 129, 130, *range(3, 128)],
            "prefix_swa_locations": list(PREFIX_SLOTS),
            "draft_logical_positions": list(range(131, 136)),
            "previous_draft_swa_locations": self.before["out_cache_loc"],
            "draft_full_locations": list(DRAFT_SLOTS),
            "draft_swa_locations": list(DRAFT_SLOTS),
            "target_injection": "Already completed at logical130 / physical2; not timed",
            "graph_operations_changed": False,
            "native_current_target_row": (
                "Native recomputes logical130; this receipt checks packed source preservation, "
                "not numerical equality of that recomputed BF16 row"
            ),
            "verification": self.verify(require_metadata=False),
        }


def apply(torch, graft, prepared):
    """Opt into disjoint draft storage once, after fixture.prepare and before captures."""
    return Binding(torch, graft, prepared)
