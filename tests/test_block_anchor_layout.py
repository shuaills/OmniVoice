from types import SimpleNamespace

import torch

from omnivoice.blockdiff_dual import build_block_anchor_layout
from omnivoice.data.collator import PackingDataCollator


def test_anchor_layout_uses_only_committed_block_boundary():
    clean = torch.arange(2 * 24, dtype=torch.long).view(2, 24)
    positions, boundaries = build_block_anchor_layout(
        clean,
        [0, 8, 24, 32],
        prefix_len=5,
        block_size=32,
        mask_id=1024,
    )

    assert positions.shape == (3, 32)
    assert torch.equal(positions[0, :8], torch.arange(29, 37))
    assert positions[0, 8:].eq(-1).all()
    assert torch.equal(positions[1, :16], torch.arange(37, 53))
    assert torch.equal(positions[2, :8], torch.arange(53, 61))
    assert boundaries[0].eq(1024).all()
    assert torch.equal(boundaries[1], clean[:, 7])
    assert torch.equal(boundaries[2], clean[:, 23])

    changed = clean.clone()
    changed[:, 8:] += 10000
    _, changed_boundaries = build_block_anchor_layout(
        changed,
        [0, 8, 24, 32],
        prefix_len=5,
        block_size=32,
        mask_id=1024,
    )
    assert torch.equal(changed_boundaries[1], boundaries[1])


def _packed_sample(length, anchor_positions, boundary_value):
    codebooks = 2
    return {
        "input_ids": torch.zeros(codebooks, length, dtype=torch.long),
        "labels": torch.full((codebooks, length), -100, dtype=torch.long),
        "audio_mask": torch.ones(length, dtype=torch.bool),
        "position_ids": torch.arange(length),
        "length": length,
        "copy_tag": torch.full((length,), 2, dtype=torch.int32),
        "block_idx": torch.zeros(length, dtype=torch.int32),
        "loss_kind": torch.zeros(codebooks, length, dtype=torch.uint8),
        "anchor_positions": torch.tensor(
            [anchor_positions], dtype=torch.long
        ),
        "anchor_boundary_ids": torch.full(
            (1, codebooks), boundary_value, dtype=torch.long
        ),
    }


def test_packing_offsets_anchor_positions_and_keeps_documents_separate():
    processor = SimpleNamespace(
        text_tokenizer=SimpleNamespace(pad_token_id=0),
        audio_mask_id=1024,
    )
    collator = PackingDataCollator(processor, batch_tokens=16)
    batch = collator(
        [
            _packed_sample(5, [1, 2, -1], 11),
            _packed_sample(4, [0, 3, -1], 22),
        ]
    )

    assert torch.equal(
        batch["anchor_positions"],
        torch.tensor([[[1, 2, -1], [5, 8, -1]]]),
    )
    assert torch.equal(
        batch["anchor_boundary_ids"],
        torch.tensor([[[11, 11], [22, 22]]]),
    )
    assert torch.equal(
        batch["document_ids"][0, :9],
        torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
    )


def test_packing_rejects_half_present_anchor_contract():
    processor = SimpleNamespace(
        text_tokenizer=SimpleNamespace(pad_token_id=0),
        audio_mask_id=1024,
    )
    collator = PackingDataCollator(processor, batch_tokens=8)
    sample = _packed_sample(4, [0, -1], 11)
    sample.pop("anchor_boundary_ids")
    try:
        collator([sample])
    except AssertionError as error:
        assert "must appear together" in str(error)
    else:
        raise AssertionError("half-present anchor contract did not fail")
