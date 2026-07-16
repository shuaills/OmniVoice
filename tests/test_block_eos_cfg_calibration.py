"""CPU tests for blockwise EOS/CFG score calibration."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from omnivoice.blockdiff import (
    EOS_CFG_CALIBRATION_LEGACY,
    EOS_CFG_CALIBRATION_MASS_PRESERVING,
    EOS_CFG_CALIBRATION_RENORM,
    _calibrate_blockwise_eos_log_probs,
    _predict_tokens_blockwise,
    block_eos_id,
)


MASK_ID = 4
EOS_ID = block_eos_id(MASK_ID)


def _model():
    return SimpleNamespace(config=SimpleNamespace(audio_mask_id=MASK_ID))


def _gen_config(**overrides):
    values = {
        "guidance_scale": 2.0,
        "class_temperature": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _predict(conditional, unconditional, gen_config):
    # _predict_tokens_blockwise lazily imports two sampling helpers from the
    # full model module.  Temperature is zero in these tests, so a tiny stub
    # keeps this scoring-only test independent of transformers/torchaudio.
    scoring_helpers = ModuleType("omnivoice.models.omnivoice")
    scoring_helpers._filter_top_k = lambda scores, ratio: scores
    scoring_helpers._gumbel_sample = lambda scores, temperature: scores
    with patch.dict(
        sys.modules,
        {"omnivoice.models.omnivoice": scoring_helpers},
    ):
        return _predict_tokens_blockwise(
            _model(), conditional, unconditional, gen_config
        )


def _logits():
    # Shape: [batch, codebook, frame, vocabulary].  The two codebooks also
    # exercise the contract that EOS is legal only on codebook 0.
    conditional = torch.tensor(
        [[[[2.0, 0.0, -1.0, -2.0, -20.0, 3.0, -20.0]],
          [[0.5, 2.0, -0.5, 0.0, -20.0, 1.0, -20.0]]]],
        dtype=torch.float64,
    )
    unconditional = torch.tensor(
        [[[[-2.0, 0.0, 0.5, 1.0, -20.0, 6.0, -20.0]],
          [[1.0, -1.0, 0.0, 0.5, -20.0, 2.0, -20.0]]]],
        dtype=torch.float64,
    )
    return conditional, unconditional


def _base_scores(conditional, unconditional, guidance_scale):
    conditional_lp = torch.log_softmax(conditional, dim=-1)
    if guidance_scale != 0:
        unconditional_lp = torch.log_softmax(unconditional, dim=-1)
        scores = torch.log_softmax(
            conditional_lp
            + guidance_scale * (conditional_lp - unconditional_lp),
            dim=-1,
        )
    else:
        scores = conditional_lp.clone()
    return conditional_lp, scores


def _legacy_scores(conditional, unconditional, guidance_scale):
    conditional_lp, scores = _base_scores(
        conditional, unconditional, guidance_scale
    )
    scores[..., MASK_ID] = -float("inf")
    scores[..., EOS_ID + 1 :] = -float("inf")
    scores[:, 1:, :, EOS_ID] = -float("inf")
    scores[:, 0:1, :, EOS_ID] = conditional_lp[:, 0:1, :, EOS_ID]
    return scores


def test_legacy_default_locks_mixed_score_behavior_and_non_normalization():
    conditional, unconditional = _logits()
    for guidance_scale in (0.0, 2.0):
        expected = _legacy_scores(
            conditional, unconditional, guidance_scale=guidance_scale
        )
        predicted, confidence = _predict(
            conditional,
            unconditional,
            _gen_config(guidance_scale=guidance_scale),
        )

        assert torch.equal(predicted, expected.argmax(dim=-1))
        torch.testing.assert_close(
            confidence,
            expected.max(dim=-1).values,
            rtol=0.0,
            atol=0.0,
        )

    # Regression evidence for the calibration defect: after replacing only
    # the EOS entry, codebook 0 no longer represents a normalized log-prob
    # distribution.  This test intentionally locks that historical default.
    expected = _legacy_scores(conditional, unconditional, guidance_scale=2.0)
    allowed_mass = expected[:, 0].exp().sum(dim=-1)
    assert not torch.allclose(allowed_mass, torch.ones_like(allowed_mass))


def test_explicit_legacy_mode_is_bit_exact_with_historical_score_surgery():
    conditional, unconditional = _logits()
    conditional_lp, cfg_lp = _base_scores(
        conditional, unconditional, guidance_scale=2.0
    )

    actual = _calibrate_blockwise_eos_log_probs(
        cfg_lp,
        conditional_lp,
        MASK_ID,
        EOS_ID,
        mode=EOS_CFG_CALIBRATION_LEGACY,
    )

    assert torch.equal(actual, _legacy_scores(conditional, unconditional, 2.0))


def test_renorm_mode_normalizes_legacy_splice_without_changing_winners():
    conditional, unconditional = _logits()
    conditional_lp, cfg_lp = _base_scores(
        conditional, unconditional, guidance_scale=2.0
    )
    legacy = _calibrate_blockwise_eos_log_probs(
        cfg_lp, conditional_lp, MASK_ID, EOS_ID, EOS_CFG_CALIBRATION_LEGACY
    )
    renorm = _calibrate_blockwise_eos_log_probs(
        cfg_lp, conditional_lp, MASK_ID, EOS_ID, EOS_CFG_CALIBRATION_RENORM
    )

    torch.testing.assert_close(
        renorm[:, 0].exp().sum(dim=-1),
        torch.ones_like(renorm[:, 0, :, 0]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.equal(renorm.argmax(dim=-1), legacy.argmax(dim=-1))
    torch.testing.assert_close(
        renorm[:, 1:], legacy[:, 1:], rtol=0.0, atol=0.0
    )

    # Codebook 0 renormalization subtracts one row-wise constant, preserving
    # all finite within-row score differences from the legacy splice.
    finite = torch.isfinite(legacy[0, 0, 0])
    delta = renorm[0, 0, 0, finite] - legacy[0, 0, 0, finite]
    torch.testing.assert_close(
        delta,
        delta[0].expand_as(delta),
        rtol=1e-12,
        atol=1e-12,
    )


def test_mass_preserving_mode_keeps_conditional_eos_and_cfg_non_eos_shape():
    conditional, unconditional = _logits()
    conditional_lp, cfg_lp = _base_scores(
        conditional, unconditional, guidance_scale=2.0
    )
    calibrated = _calibrate_blockwise_eos_log_probs(
        cfg_lp,
        conditional_lp,
        MASK_ID,
        EOS_ID,
        EOS_CFG_CALIBRATION_MASS_PRESERVING,
    )

    torch.testing.assert_close(
        calibrated[:, 0].exp().sum(dim=-1),
        torch.ones_like(calibrated[:, 0, :, 0]),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        calibrated[:, 0:1, :, EOS_ID],
        conditional_lp[:, 0:1, :, EOS_ID],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isneginf(calibrated[:, 1:, :, EOS_ID]).all()
    assert torch.isneginf(calibrated[..., MASK_ID]).all()
    assert torch.isneginf(calibrated[..., EOS_ID + 1 :]).all()
    legacy = _calibrate_blockwise_eos_log_probs(
        cfg_lp,
        conditional_lp,
        MASK_ID,
        EOS_ID,
        EOS_CFG_CALIBRATION_LEGACY,
    )
    torch.testing.assert_close(
        calibrated[:, 1:], legacy[:, 1:], rtol=0.0, atol=0.0
    )

    # Removing the allocated EOS mass must leave the legal ordinary classes
    # in exactly their CFG relative proportions on codebook 0.
    torch.testing.assert_close(
        torch.softmax(calibrated[:, 0:1, :, :MASK_ID], dim=-1),
        torch.softmax(cfg_lp[:, 0:1, :, :MASK_ID], dim=-1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_predictor_routes_explicit_experimental_modes_and_rejects_typos():
    conditional, unconditional = _logits()
    conditional_lp, cfg_lp = _base_scores(
        conditional, unconditional, guidance_scale=2.0
    )
    for mode in (
        EOS_CFG_CALIBRATION_RENORM,
        EOS_CFG_CALIBRATION_MASS_PRESERVING,
    ):
        expected = _calibrate_blockwise_eos_log_probs(
            cfg_lp, conditional_lp, MASK_ID, EOS_ID, mode
        )
        predicted, confidence = _predict(
            conditional,
            unconditional,
            _gen_config(eos_cfg_calibration=mode),
        )
        assert torch.equal(predicted, expected.argmax(dim=-1))
        torch.testing.assert_close(
            confidence,
            expected.max(dim=-1).values,
            rtol=0.0,
            atol=0.0,
        )

    try:
        _predict(
            conditional,
            unconditional,
            _gen_config(eos_cfg_calibration="renrom"),
        )
    except ValueError as exc:
        assert "unknown eos_cfg_calibration" in str(exc)
    else:
        raise AssertionError("invalid calibration mode did not fail fast")
