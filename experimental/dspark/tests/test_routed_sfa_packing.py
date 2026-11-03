from __future__ import annotations

import pytest
import torch

from scripts.megakernel.boltin_pack import pack_routed_sfa


def test_packed_scales_match_utccp_source_words() -> None:
    scales = torch.arange(256 * 128, dtype=torch.int32).to(torch.uint8).reshape(256, 128)
    packed = pack_routed_sfa(scales).flatten()
    for tile in range(2):
        for group in range(32):
            block = packed[(tile * 32 + group) * 512 : (tile * 32 + group + 1) * 512]
            for word in range(128):
                # UTCCP word 4*lane+quarter supplies row 32*quarter+lane.
                row = tile * 128 + (word % 4) * 32 + word // 4
                assert torch.equal(
                    block[word * 4 : word * 4 + 4], scales[row, group * 4 : group * 4 + 4]
                )


@pytest.mark.parametrize("shape", [(127, 32), (128, 5), (2, 128, 32)])
def test_packed_scales_reject_partial_blocks(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        pack_routed_sfa(torch.empty(shape, dtype=torch.uint8))


def test_packed_scales_never_converts_values() -> None:
    with pytest.raises(ValueError, match="uint8"):
        pack_routed_sfa(torch.empty((128, 32), dtype=torch.float32))
