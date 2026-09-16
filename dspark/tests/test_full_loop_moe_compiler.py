import pytest
import torch

from deepspec.megakernel.full_loop_moe_compiler import (
    compile_routed_moe_lowering,
    gated_matrix_a_row_permutation,
    matrix_a_row_permutation,
    pack_topk_ids_and_weights,
)


def test_gb300_routed_lowering_exposes_measured_w13_claims():
    lowering = compile_routed_moe_lowering()
    assert lowering.program_items == 1152
    assert lowering.persistent_tasks == 144
    assert lowering.w13_claim == 8
    assert lowering.warp_reduce


def test_routed_lowering_rejects_non_pair_claims():
    with pytest.raises(ValueError):
        compile_routed_moe_lowering(w13_claim=3)
    with pytest.raises(ValueError, match="barrier lifecycle"):
        compile_routed_moe_lowering(w13_claim=4)


def test_matrix_a_row_permutation_matches_epilogue_tile_128_contract():
    assert matrix_a_row_permutation(32).tolist() == [
        0,
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        29,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        31,
    ]


def test_gated_row_compiler_composes_halves_before_mma_shuffle():
    permutation = gated_matrix_a_row_permutation(64)
    assert permutation[:16].tolist() == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
    ]
    assert sorted(permutation.tolist()) == list(range(64))


def test_packed_topk_round_trips_ids_and_bf16_weights():
    ids = torch.tensor([[3, 255], [17, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [1.0, -0.5]], dtype=torch.float32)
    packed = pack_topk_ids_and_weights(ids, weights)
    assert torch.equal(packed >> 16, ids)
    unpacked = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    torch.testing.assert_close(unpacked.float(), weights.bfloat16().float())


def test_moe_compiler_rejects_ambiguous_layouts():
    with pytest.raises(ValueError):
        gated_matrix_a_row_permutation(63)
    with pytest.raises(TypeError):
        pack_topk_ids_and_weights(
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.float32),
        )
