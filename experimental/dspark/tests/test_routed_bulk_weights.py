from __future__ import annotations

import pytest
import torch

from scripts.megakernel.boltin_pack import pack_routed_bulk_weights


@pytest.mark.parametrize("shape", [(128, 256), (2048, 2048), (4096, 1024)])
def test_bulk_stage_matches_four_tensor_map_boxes(shape: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(42)
    weight = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    packed = pack_routed_bulk_weights(weight).flatten()
    stages = shape[1] // 256
    for tile in range(shape[0] // 128):
        for stage in range(stages):
            offset = (tile * stages + stage) * 32768
            # Each original tensor-map box copies 64 adjacent bytes from
            # each of 128 rows. Four boxes must arrive unchanged in SMEM.
            for micro in range(4):
                expected = weight[
                    tile * 128 : (tile + 1) * 128,
                    stage * 256 + micro * 64 : stage * 256 + (micro + 1) * 64,
                ]
                assert torch.equal(
                    packed[offset + micro * 8192 : offset + (micro + 1) * 8192],
                    expected.contiguous().flatten(),
                )


@pytest.mark.parametrize("shape", [(127, 256), (128, 255), (2, 128, 256)])
def test_bulk_weights_require_complete_tiles(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        pack_routed_bulk_weights(torch.empty(shape, dtype=torch.uint8))
