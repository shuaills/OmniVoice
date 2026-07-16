"""CPU gates for the opt-in CFG C/U_shared/U_drop_ref training contract."""

import hashlib
import json
import random
import types

import pytest
import torch

from omnivoice.blockdiff import block_eos_id
from omnivoice.blockdiff_dual import (
    KIND_EOS,
    TAG_CLEAN,
    TAG_NOISY,
    OmniVoiceBlockDualSampleProcessor,
)
from omnivoice.training.config import TrainingConfig
from omnivoice.training.split_loss import category_counts


C = 8
MASK = 1024
BS = 32


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, _text, return_tensors="pt"):
        return types.SimpleNamespace(
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long)
        )


def _processor(**overrides):
    kwargs = {
        "text_tokenizer": _Tokenizer(),
        "num_channels": C,
        "audio_mask_id": MASK,
        "prompt_ratio_range": (0.3, 0.3),
        "mask_ratio_range": (0.0, 1.0),
        "drop_cond_ratio": 0.1,
        "language_ratio": 0.0,
        "use_pinyin_ratio": 0.0,
        "instruct_ratio": 0.0,
        "only_instruct_ratio": 0.0,
        "block_size": BS,
        "eos_decouple_silence": True,
        "eos_band_k": 4,
        "silence_void_window": 32,
    }
    kwargs.update(overrides)
    return OmniVoiceBlockDualSampleProcessor(**kwargs)


def _sample(sample_id, T=96):
    values = torch.arange(C * T, dtype=torch.long).reshape(C, T) % MASK
    return {
        "audio_tokens": values,
        "label": {"id": sample_id, "text": "fixed cfg branch fixture"},
    }


def _tensor_digest(output):
    digest = hashlib.sha256()
    for key in (
        "input_ids",
        "labels",
        "audio_mask",
        "position_ids",
        "copy_tag",
        "block_idx",
        "loss_kind",
    ):
        tensor = output[key].contiguous()
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _ragged_boundaries(target_len, q):
    boundaries = [0, q]
    while boundaries[-1] <= target_len:
        boundaries.append(boundaries[-1] + BS)
    return boundaries


def _ragged_block_ids(length, q):
    positions = torch.arange(length)
    return torch.where(
        positions < q,
        torch.zeros_like(positions),
        1 + (positions - q) // BS,
    ).to(torch.int32)


def test_disabled_contract_is_old_rng_and_tensor_bit_identity():
    random.seed(20260717)
    torch.manual_seed(20260717)
    output = _processor()(_sample("golden", T=70))

    assert set(output) == {
        "input_ids",
        "labels",
        "audio_mask",
        "length",
        "position_ids",
        "copy_tag",
        "block_idx",
        "loss_kind",
    }
    assert output["length"] == 166
    assert _tensor_digest(output) == (
        "d9965466302d21d1b68aa2d81596fae72f69e12735f20377dd70b82b7e661690"
    )
    assert random.random() == 0.10679318030608098
    assert torch.equal(
        torch.rand(4),
        torch.tensor(
            [
                0.7584938406944275,
                0.25654345750808716,
                0.5458587408065796,
                0.8998725414276123,
            ]
        ),
    )


def test_keyed_branch_stream_is_deterministic_global_rng_neutral_and_90_5_5():
    processor = _processor(cfg_branch_training=True, cfg_branch_seed=314159)
    before = random.getstate()
    first = processor._choose_cfg_branch(processor._cfg_rng(_sample("same")))
    second = processor._choose_cfg_branch(processor._cfg_rng(_sample("same")))
    assert first == second
    assert random.getstate() == before

    counts = {"C": 0, "U_shared": 0, "U_drop_ref": 0}
    for index in range(20_000):
        rng = processor._cfg_rng(_sample(f"branch-{index}"))
        counts[processor._choose_cfg_branch(rng)] += 1

    assert 0.885 <= counts["C"] / 20_000 <= 0.915
    assert 0.04 <= counts["U_shared"] / 20_000 <= 0.06
    assert 0.04 <= counts["U_drop_ref"] / 20_000 <= 0.06


def test_enabled_c_branch_keeps_control_tensors_and_global_rng_bit_identical():
    source = _sample("matching-c", T=70)
    core_keys = (
        "input_ids",
        "labels",
        "audio_mask",
        "position_ids",
        "copy_tag",
        "block_idx",
        "loss_kind",
    )

    random.seed(101)  # legacy branch draw is conditional, not drop_cond
    torch.manual_seed(101)
    control = _processor()(source)
    control_python_state = random.getstate()
    control_torch_state = torch.random.get_rng_state().clone()

    random.seed(101)
    torch.manual_seed(101)
    candidate = _processor(
        cfg_branch_training=True,
        cfg_branch_cond_ratio=1.0,
        cfg_branch_shared_ratio=0.0,
        cfg_branch_drop_ref_ratio=0.0,
    )(source)

    assert candidate["cfg_branch"] == "C"
    for key in core_keys:
        assert torch.equal(control[key], candidate[key]), key
    assert control["length"] == candidate["length"]
    assert random.getstate() == control_python_state
    assert torch.equal(torch.random.get_rng_state(), control_torch_state)


def test_drop_ref_phase_mixture_covers_all_q_and_oversamples_one_to_four():
    processor = _processor(cfg_branch_training=True, cfg_branch_seed=271828)
    counts = {q: 0 for q in range(1, BS + 1)}
    for index in range(20_000):
        sample = _sample(f"phase-{index}", T=320)
        rng = processor._cfg_rng(sample)
        assert processor._choose_cfg_branch(rng) in {
            "C",
            "U_shared",
            "U_drop_ref",
        }
        phase = processor._choose_drop_ref_phase(320, rng)
        assert phase["requested_q"] == phase["actual_q"]
        assert phase["rebucketed"] is False
        counts[phase["actual_q"]] += 1

    assert all(counts.values())
    short_share = sum(counts[q] for q in range(1, 5)) / 20_000
    assert 0.54 <= short_share <= 0.585
    assert min(counts[q] for q in range(1, 5)) > max(
        counts[q] for q in range(5, BS + 1)
    )


def test_u_shared_keeps_reference_and_target_in_the_same_original_timeline():
    source = _sample("shared", T=70)
    truth = source["audio_tokens"].clone()
    random.seed(101)
    torch.manual_seed(101)
    output = _processor(
        cfg_branch_training=True,
        cfg_branch_cond_ratio=0.0,
        cfg_branch_shared_ratio=1.0,
        cfg_branch_drop_ref_ratio=0.0,
        cfg_branch_seed=1,
    )(source)

    S = 21
    clean_len, canvas_len = 64, 96
    assert output["cfg_branch"] == "U_shared"
    assert output["cfg_prompt_cut"] == S
    assert output["cfg_target_frames"] == 70 - S
    assert output["cfg_reference_frames"] == S
    assert output["length"] == clean_len + canvas_len
    assert not (output["copy_tag"] == 0).any()
    assert torch.equal(output["input_ids"][:, :clean_len], truth[:, :clean_len])

    noisy_input = output["input_ids"][:, clean_len:]
    noisy_labels = output["labels"][:, clean_len:]
    assert torch.equal(noisy_input[:, :S], truth[:, :S])
    assert (noisy_labels[:, :S] == -100).all()
    assert (output["block_idx"][clean_len : clean_len + BS] == 0).all()


@pytest.mark.parametrize("q", range(1, BS + 1))
def test_u_drop_ref_ragged_q_geometry_has_no_reference_and_one_band4_event(q):
    source = _sample(f"drop-q{q}", T=96)
    original = source["audio_tokens"].clone()
    random.seed(9000 + q)
    torch.manual_seed(9000 + q)
    output = _processor(
        cfg_branch_training=True,
        cfg_branch_cond_ratio=0.0,
        cfg_branch_shared_ratio=0.0,
        cfg_branch_drop_ref_ratio=1.0,
        cfg_branch_seed=100 + q,
        cfg_drop_ref_q_min=q,
        cfg_drop_ref_q_max=q,
    )(source)

    S = output["cfg_prompt_cut"]
    target = original[:, S:]
    target_len = target.shape[1]
    boundaries = _ragged_boundaries(target_len, q)
    clean_len, canvas_len = boundaries[-2], boundaries[-1]

    assert output["cfg_branch"] == "U_drop_ref"
    assert output["cfg_requested_q"] == q
    assert output["cfg_actual_q"] == q
    assert output["cfg_rebucketed"] is False
    assert output["cfg_reference_frames"] == 0
    assert output["cfg_reference_leak_tokens"] == 0
    assert 1 <= S < original.shape[1]
    observed_q = BS - (S % BS) if S % BS else BS
    assert observed_q == q
    assert output["length"] == clean_len + canvas_len

    tags = output["copy_tag"]
    positions = output["position_ids"]
    blocks = output["block_idx"]
    assert (tags[:clean_len] == TAG_CLEAN).all()
    assert (tags[clean_len:] == TAG_NOISY).all()
    assert torch.equal(positions[:clean_len], torch.arange(clean_len))
    assert torch.equal(positions[clean_len:], torch.arange(canvas_len))
    assert torch.equal(blocks[:clean_len], _ragged_block_ids(clean_len, q))
    assert torch.equal(blocks[clean_len:], _ragged_block_ids(canvas_len, q))

    clean_input = output["input_ids"][:, :clean_len]
    noisy_input = output["input_ids"][:, clean_len:]
    noisy_labels = output["labels"][:, clean_len:]
    assert torch.equal(clean_input, target[:, :clean_len])
    masked = noisy_input[:, :target_len] == MASK
    assert torch.equal(noisy_labels[:, :target_len][masked], target[masked])
    assert torch.equal(noisy_input[:, :target_len][~masked], target[~masked])
    assert (noisy_labels[:, :target_len][~masked] == -100).all()

    eos_width = min(4, canvas_len - target_len)
    assert output["cfg_eos_band_width"] == eos_width
    assert (
        noisy_labels[0, target_len : target_len + eos_width]
        == block_eos_id(MASK)
    ).all()
    assert (
        output["loss_kind"][
            0, clean_len + target_len : clean_len + target_len + eos_width
        ]
        == KIND_EOS
    ).all()
    document_ids = torch.zeros((1, output["length"]), dtype=torch.int32)
    counts = category_counts(output["loss_kind"].unsqueeze(0), document_ids)
    assert counts.eos_count.item() == 1
    assert counts.invariant_errors.item() == 0


def test_training_config_parses_opt_in_cfg_contract(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            {
                "cfg_branch_training": True,
                "cfg_branch_cond_ratio": 0.9,
                "cfg_branch_shared_ratio": 0.05,
                "cfg_branch_drop_ref_ratio": 0.05,
                "cfg_branch_seed": 17,
                "cfg_drop_ref_short_bucket_ratio": 0.5,
                "cfg_drop_ref_q_min": 1,
                "cfg_drop_ref_q_max": 32,
                "cfg_drop_ref_short_q_max": 4,
            }
        )
    )

    config = TrainingConfig.from_json(path)
    assert config.cfg_branch_training is True
    assert config.cfg_branch_cond_ratio == 0.9
    assert config.cfg_branch_shared_ratio == 0.05
    assert config.cfg_branch_drop_ref_ratio == 0.05
    assert config.cfg_branch_seed == 17
    assert config.cfg_drop_ref_short_bucket_ratio == 0.5
    assert (config.cfg_drop_ref_q_min, config.cfg_drop_ref_q_max) == (1, 32)
    assert config.cfg_drop_ref_short_q_max == 4
