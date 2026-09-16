import pytest

from deepspec.modeling.dspark.deepseek_v4.official import (
    OFFICIAL_V4_FLASH_SHARDS,
    official_snapshot_patterns,
    required_checkpoint_shards,
    selected_weight_map,
)


def test_selected_weight_map_keeps_only_v4_proposal_tensors():
    index = {
        "weight_map": {
            "embed.weight": OFFICIAL_V4_FLASH_SHARDS[0],
            "layers.0.attn.weight": "model-00002-of-00048.safetensors",
            "head.weight": OFFICIAL_V4_FLASH_SHARDS[1],
            "mtp.0.attn.wq_a.weight": OFFICIAL_V4_FLASH_SHARDS[2],
            "mtp.1.ffn.experts.0.w1.weight": OFFICIAL_V4_FLASH_SHARDS[3],
            "mtp.2.confidence_head.proj.weight": OFFICIAL_V4_FLASH_SHARDS[4],
        }
    }

    selected = selected_weight_map(index)

    assert set(selected) == {
        "embed.weight",
        "head.weight",
        "mtp.0.attn.wq_a.weight",
        "mtp.1.ffn.experts.0.w1.weight",
        "mtp.2.confidence_head.proj.weight",
    }
    assert required_checkpoint_shards(selected) == OFFICIAL_V4_FLASH_SHARDS


def test_selected_weight_map_rejects_non_v4_checkpoint():
    with pytest.raises(ValueError, match="embedding or LM head"):
        selected_weight_map({"weight_map": {"mtp.0.x": "model.safetensors"}})


def test_official_snapshot_is_pinned_to_only_the_five_proposal_shards():
    patterns = official_snapshot_patterns()
    assert tuple(pattern for pattern in patterns if pattern.endswith(".safetensors")) == (
        OFFICIAL_V4_FLASH_SHARDS
    )
