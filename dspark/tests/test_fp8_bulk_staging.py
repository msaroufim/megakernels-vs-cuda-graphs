from __future__ import annotations

import pytest
import torch

from scripts.megakernel.boltin_pack import pack_fp8_bulk_staging


def test_fp8_bulk_staging_matches_mma_shared_addresses() -> None:
    raw = torch.arange(256 * 256, dtype=torch.int32).to(torch.uint8).reshape(256, 256)
    packed = pack_fp8_bulk_staging(raw.view(torch.float8_e4m3fn)).flatten()
    row = torch.arange(256)[:, None]
    column = torch.arange(256)[None, :]
    address = (
        (((row // 128) * 2 + column // 128) * 4 + column % 128 // 32) * 4096
        + column % 32 // 16 * 2048
        + row % 128 * 16
        + column % 16
    )
    assert torch.equal(packed[address], raw)


@pytest.mark.parametrize("shape", [(128, 32), (127, 128), (2, 128, 128)])
def test_fp8_bulk_staging_rejects_partial_tiles(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        pack_fp8_bulk_staging(torch.empty(shape, dtype=torch.float8_e4m3fn))
