import pytest
import torch

from deepspec.megakernel import (
    build_v4_launch_abi,
    expected_v4_weights,
    pack_v4_weights,
    plan_v4_weight_arena,
)
from deepspec.megakernel.v4_weights import (
    V4ExpertWeightSlot,
    V4GlobalWeightSlot,
    V4LayerWeightSlot,
    V4WeightTableLayout,
)


def test_v4_runtime_weight_manifest_is_complete_unique_and_aligned():
    expected = expected_v4_weights()
    plan = plan_v4_weight_arena()
    assert len(expected) == len(plan.entries) == 4704
    assert len({item.name for item in expected}) == 4704
    assert len({item.offset_index for item in expected}) == 4704
    assert sum(offset < 0 for offset in plan.offset_table) == 20
    assert plan.arena_bytes == 14_206_958_848
    assert plan.arena_bytes % plan.alignment == 0
    for entry in plan.entries:
        assert entry.arena_offset % plan.alignment == 0
        assert plan.offset_table[entry.offset_index] == entry.arena_offset
        assert entry.nbytes > 0

    by_name = {item.name: item for item in expected}
    assert by_name["head.weight"].dtype == "float32"
    assert by_name["mtp.0.attn.wo_a.weight"].dtype == "bfloat16"
    assert by_name["mtp.1.ffn.experts.255.w2.weight"].shape == (4096, 1024)
    assert by_name["mtp.2.markov_head.markov_w2.weight"].shape == (129280, 256)


def test_v4_offset_table_has_dense_semantic_expert_sections():
    abi = build_v4_launch_abi()
    spec = abi.spec
    layout = V4WeightTableLayout.flash(spec)
    assert layout.global_slot(V4GlobalWeightSlot.LM_HEAD) == 1
    assert layout.layer_slot(2, V4LayerWeightSlot.CONFIDENCE) == 97
    first_expert = layout.routed_expert_slot(
        0,
        0,
        V4ExpertWeightSlot.W1,
        experts=spec.num_routed_experts,
    )
    last_expert = layout.routed_expert_slot(
        2,
        255,
        V4ExpertWeightSlot.W3_SCALE,
        experts=spec.num_routed_experts,
    )
    assert first_expert == 98
    assert last_expert + 1 == layout.shared_expert_base
    assert layout.total_slots == 4724
    assert abi.weight_arena.source_tensor_count == 4707
    assert abi.weight_arena.runtime_tensor_count == 4704


def test_v4_packer_rejects_an_unrelated_module_before_allocating_arena():
    module = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="V4 parameter mismatch"):
        pack_v4_weights(module)
