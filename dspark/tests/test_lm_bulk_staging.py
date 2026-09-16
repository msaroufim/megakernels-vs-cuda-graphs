from __future__ import annotations

import pytest
import torch

from scripts.megakernel.boltin_pack import pack_lm_bulk_staging


def test_bulk_layout_matches_mma_operand_addresses() -> None:
    # Include every BF16 bit pattern, including NaNs: staging must copy bits.
    bits = torch.arange(256 * 256, dtype=torch.int32).to(torch.int16).reshape(256, 256)
    packed = pack_lm_bulk_staging(bits.view(torch.bfloat16)).view(torch.int16).flatten()
    for row in (0, 1, 63, 127, 128, 255):
        for column in (0, 7, 8, 15, 16, 63, 127, 255):
            # Each M128/K16 UMMA operand is two consecutive 2-KiB halves.
            tile = row // 128
            chunk = column // 16
            operand_base = (tile * 16 + chunk) * 2048
            address = operand_base + (column % 16 // 8) * 1024 + (row % 128) * 8
            assert packed[address + column % 8].item() == bits[row, column].item()


@pytest.mark.parametrize("shape", [(129, 32), (128, 17), (128, 2, 16)])
def test_bulk_layout_rejects_partial_tiles(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        pack_lm_bulk_staging(torch.empty(shape, dtype=torch.bfloat16))


def test_bulk_layout_rejects_implicit_dtype_conversion() -> None:
    with pytest.raises(ValueError, match="BF16 matrix"):
        pack_lm_bulk_staging(torch.empty((128, 16), dtype=torch.float32))
