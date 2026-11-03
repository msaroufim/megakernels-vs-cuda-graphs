from __future__ import annotations

from pathlib import Path

import torch

_KV_BYTES = 584
_KV_DATA_BYTES = 576
_KV_SCALE_BYTES = 8


def load_prefill_snapshot(path: Path) -> dict:
    """Load and structurally validate an offline SGLang prefill fixture."""

    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "seq_lens",
        "full_locs",
        "bonus_tokens",
        "draft_block_ids",
        "draft_tokens",
        "target_pool",
        "draft_pool",
    }
    missing = required.difference(snapshot)
    if missing:
        raise ValueError(f"prefill snapshot is missing {sorted(missing)}")
    seq_lens = snapshot["seq_lens"]
    if tuple(seq_lens.shape) != (1,) or int(seq_lens[0]) <= 0:
        raise ValueError(f"prefill snapshot requires one live prefix, got {seq_lens}")
    if snapshot["full_locs"].numel() != int(seq_lens[0]):
        raise ValueError("prefill full-location count does not match the prefix")
    if tuple(snapshot["draft_tokens"].shape) != (1, 5):
        raise ValueError("prefill snapshot requires five draft tokens")
    return snapshot


def verify_ids(snapshot: dict, *, device: torch.device | str = "cuda") -> torch.Tensor:
    """Return the anchor plus five draft candidates consumed by target verify."""

    return (
        torch.cat((snapshot["draft_block_ids"][:, :1], snapshot["draft_tokens"]), dim=1)
        .flatten()
        .to(device=device, dtype=torch.int64)
    )


def _lookup_page(group: dict, page_id: int) -> torch.Tensor:
    matches = torch.nonzero(group["page_ids"] == page_id).flatten()
    if matches.numel() != 1:
        raise ValueError(f"captured pool does not contain page {page_id}")
    return group["pages"][int(matches[0])].view(torch.uint8)


def unpack_packed_kv_rows(
    group: dict,
    locations: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Convert SGLang's page-split FlashMLA layout to page-size-one rows."""

    output = torch.zeros((locations.numel(), 1152), dtype=torch.uint8)
    for output_row, location_value in enumerate(locations.tolist()):
        location = int(location_value)
        page = _lookup_page(group, location // page_size)
        offset = location % page_size
        data_base = offset * _KV_DATA_BYTES
        scale_base = page_size * _KV_DATA_BYTES + offset * _KV_SCALE_BYTES
        output[output_row, :_KV_DATA_BYTES].copy_(page[data_base : data_base + _KV_DATA_BYTES])
        output[output_row, _KV_DATA_BYTES:_KV_BYTES].copy_(
            page[scale_base : scale_base + _KV_SCALE_BYTES]
        )
    return output


def unpack_index_pages(group: dict, *, rows: int) -> torch.Tensor:
    """Materialize a dense page-size-64 C4 index cache from captured pages."""

    pages = max((rows + 63) // 64, 1)
    output = torch.zeros((pages, 64 * 132), dtype=torch.uint8)
    for page_id in group["page_ids"].tolist():
        page_id = int(page_id)
        if page_id < pages:
            output[page_id].copy_(_lookup_page(group, page_id))
    return output


def seed_target_layer(
    snapshot: dict,
    *,
    layer: int,
    ratio: int,
    vendor_raw_cache: torch.Tensor,
    vendor_index_raw_cache: torch.Tensor | None,
    target_state: torch.Tensor | None,
    index_state: torch.Tensor | None,
) -> None:
    """Translate one target layer's captured prefix into the standalone ABI."""

    pool = snapshot["target_pool"]
    seq_len = int(snapshot["seq_lens"][0])
    swa_locations = pool["swa_locs"].to(torch.int64)
    packed_swa = unpack_packed_kv_rows(pool["swa"][layer], swa_locations, page_size=256)
    live_start = max(0, seq_len - 128)
    for position in range(live_start, seq_len):
        vendor_raw_cache[position & 127].copy_(packed_swa[position].to("cuda"))

    if ratio:
        compressed_count = seq_len // ratio
        if compressed_count:
            emitted_full_locations = snapshot["full_locs"][
                ratio - 1 : compressed_count * ratio : ratio
            ].to(torch.int64)
            locations = emitted_full_locations // ratio
            packed_extra = unpack_packed_kv_rows(
                pool["extra"][layer], locations, page_size=256 // ratio
            )
            vendor_raw_cache[128 : 128 + compressed_count].copy_(packed_extra.to("cuda"))
        if target_state is None:
            raise ValueError("compressed target layer has no state allocation")
        captured_state = pool["attention_state"][layer]["rows"]
        if ratio == 4:
            # Standalone legacy plans use the first 8-row ring. The captured
            # request lives in SWA page 1, whose production 16-row ring was
            # saved in logical order.
            target_state.view(8, -1).copy_(captured_state[:8].to("cuda"))
        else:
            target_state.view(128, -1).copy_(captured_state[:128].to("cuda"))

    if ratio == 4:
        if vendor_index_raw_cache is None or index_state is None:
            raise ValueError("C4 target layer has no index cache/state allocation")
        index_rows = max(seq_len // 4, 1)
        index_locations = snapshot["full_locs"][3 : index_rows * 4 : 4].to(torch.int64) // 4
        index_rows_packed = torch.stack(
            [
                _lookup_page(pool["index"][layer], int(location) // 64)[
                    (int(location) % 64) * 132 : (int(location) % 64 + 1) * 132
                ]
                for location in index_locations
            ]
        )
        vendor_index_raw_cache.zero_()
        vendor_index_raw_cache.view(-1, 132)[:index_rows].copy_(index_rows_packed.to("cuda"))
        index_state.view(8, -1).copy_(pool["index_state"][layer]["rows"][:8].to("cuda"))


def seed_graft_pool(snapshot: dict, graft) -> None:
    """Seed the captured three-layer graft graph with the same prompt KV."""

    pool = snapshot["draft_pool"]
    seq_len = int(snapshot["seq_lens"][0])
    locations = pool["swa_locs"].to(torch.int64)
    live_start = max(0, seq_len - 128)
    draft_pool = graft.bundle.draft_model_runner.token_to_kv_pool
    if draft_pool._unified_kv:
        raise ValueError("captured graft unexpectedly uses unified KV")
    for layer, destination in enumerate(draft_pool.swa_kv_pool.kv_buffer):
        source = unpack_packed_kv_rows(pool["swa"][layer], locations, page_size=256)
        compact = torch.zeros((128, 1152), dtype=torch.uint8)
        for position in range(live_start, seq_len):
            compact[position & 127].copy_(source[position])
        # The standalone pool maps full cache rows 0..127 into SWA page zero.
        data = destination[0].view(torch.uint8)
        data.zero_()
        for row in range(128):
            data[row * _KV_DATA_BYTES : (row + 1) * _KV_DATA_BYTES].copy_(
                compact[row, :_KV_DATA_BYTES].to("cuda")
            )
            scale = 256 * _KV_DATA_BYTES + row * _KV_SCALE_BYTES
            data[scale : scale + _KV_SCALE_BYTES].copy_(
                compact[row, _KV_DATA_BYTES:_KV_BYTES].to("cuda")
            )
    mapping = draft_pool.full_to_swa_index_mapping
    mapping[:128].copy_(torch.arange(128, dtype=mapping.dtype, device=mapping.device))
    req_to_token = graft.bundle.draft_model_runner.req_to_token_pool.req_to_token
    logical = torch.arange(
        req_to_token.shape[1], dtype=req_to_token.dtype, device=req_to_token.device
    )
    req_to_token[0].copy_(logical.remainder(128))
