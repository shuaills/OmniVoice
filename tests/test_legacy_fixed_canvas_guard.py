"""Regression gates for block-checkpoint generation routing."""

from types import MethodType, SimpleNamespace

import pytest

from omnivoice.cli.infer import get_parser
from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationConfig,
)


def _model_config(**overrides):
    values = {
        "audio_mask_id": 1024,
        "audio_vocab_size": 1025,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _assert_legacy_allowed(model_config, *, allow=False):
    model = SimpleNamespace(config=model_config)
    generation_config = OmniVoiceGenerationConfig(
        allow_block_checkpoint_fixed_canvas=allow
    )
    OmniVoice._assert_legacy_fixed_canvas_allowed(model, generation_config)


@pytest.mark.parametrize(
    "model_config",
    [
        _model_config(
            audio_vocab_size=1026,
            block_migrated_from="/checkpoints/OmniVoice-official",
        ),
        # Older derived checkpoints may lose migration provenance while still
        # retaining the block-only [eos] vocabulary contract.
        _model_config(audio_vocab_size=1026),
    ],
)
def test_block_checkpoint_refuses_legacy_fixed_canvas_by_default(model_config):
    with pytest.raises(RuntimeError, match="block checkpoint.*fixed-canvas"):
        _assert_legacy_allowed(model_config)


def test_block_checkpoint_fixed_canvas_requires_explicit_opt_in():
    _assert_legacy_allowed(
        _model_config(
            audio_vocab_size=1026,
            block_migrated_from="/checkpoints/OmniVoice-official",
        ),
        allow=True,
    )


def test_generate_checks_block_routing_before_inference_resources():
    model = SimpleNamespace(
        config=_model_config(audio_vocab_size=1026),
        audio_tokenizer=None,
        text_tokenizer=None,
    )
    model._assert_block_only_generation_options = MethodType(
        OmniVoice._assert_block_only_generation_options, model
    )
    model._assert_legacy_fixed_canvas_allowed = MethodType(
        OmniVoice._assert_legacy_fixed_canvas_allowed, model
    )

    with pytest.raises(RuntimeError, match="block checkpoint.*fixed-canvas"):
        OmniVoice.generate(model, "hello")

    with pytest.raises(RuntimeError, match="audio/text tokenizers"):
        OmniVoice.generate(
            model,
            "hello",
            allow_block_checkpoint_fixed_canvas=True,
        )


def test_fixed_canvas_generate_rejects_block_only_eos_calibration():
    model = SimpleNamespace(
        config=_model_config(),
        audio_tokenizer=None,
        text_tokenizer=None,
    )
    model._assert_block_only_generation_options = MethodType(
        OmniVoice._assert_block_only_generation_options, model
    )
    model._assert_legacy_fixed_canvas_allowed = MethodType(
        OmniVoice._assert_legacy_fixed_canvas_allowed, model
    )

    with pytest.raises(ValueError, match="block-causal decoder option"):
        OmniVoice.generate(model, "hello", eos_cfg_calibration="renorm")


@pytest.mark.parametrize(
    "model_config",
    [
        _model_config(),
        _model_config(
            audio_vocab_size=1027,
            elastic_migrated_from="/checkpoints/OmniVoice-official",
        ),
    ],
)
def test_official_and_elastic_checkpoints_keep_legacy_generation(model_config):
    _assert_legacy_allowed(model_config)


def test_cli_fixed_canvas_opt_in_is_off_by_default_and_explicit_when_requested():
    parser = get_parser()
    required = ["--text", "hello", "--output", "out.wav"]

    assert parser.parse_args(required).allow_block_checkpoint_fixed_canvas is False
    assert (
        parser.parse_args(required + ["--allow_block_checkpoint_fixed_canvas"])
        .allow_block_checkpoint_fixed_canvas
        is True
    )
