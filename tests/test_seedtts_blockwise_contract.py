import numpy as np
import pytest

from omnivoice.eval.seedtts_blockwise_contract import (
    prepare_official_emilia_audio,
    prepare_reference_text,
    resolve_prompt_contract,
    restore_official_emilia_output_rms,
)


def test_current_contract_keeps_existing_defaults() -> None:
    zh = resolve_prompt_contract(
        "current",
        lang_arg=None,
        tsv_path="/datasets/seedtts/zh/test.tsv",
    )
    en = resolve_prompt_contract(
        "current",
        lang_arg=None,
        tsv_path="/datasets/seedtts/en/test.tsv",
    )

    assert zh.language == "zh"
    assert en.language == "en"
    assert zh.ref_text_punctuation == "add"
    assert prepare_reference_text("参考文本", zh.ref_text_punctuation) == "参考文本。"
    assert prepare_reference_text("Reference text", en.ref_text_punctuation) == (
        "Reference text."
    )


def test_current_contract_can_explicitly_disable_language() -> None:
    resolved = resolve_prompt_contract(
        "current",
        lang_arg="None",
        tsv_path="/datasets/seedtts/zh/test.tsv",
        ref_text_punctuation_arg="preserve",
    )

    assert resolved.language is None
    assert resolved.ref_text_punctuation == "preserve"


def test_official_emilia_prompt_can_keep_current_language_for_prompt_only_ab() -> None:
    resolved = resolve_prompt_contract(
        "official-emilia",
        lang_arg=None,
        tsv_path="/datasets/seedtts/zh/test.tsv",
    )

    assert resolved.language == "zh"
    assert resolved.ref_text_punctuation == "preserve"


def test_official_emilia_prompt_combines_with_explicit_language_none() -> None:
    resolved = resolve_prompt_contract(
        "official-emilia",
        lang_arg="None",
        tsv_path="/datasets/seedtts/zh/test.tsv",
    )

    assert resolved.language is None
    assert resolved.ref_text_punctuation == "preserve"
    assert prepare_reference_text("参考文本", resolved.ref_text_punctuation) == "参考文本"


@pytest.mark.parametrize(
    ("lang_arg", "punctuation"),
    [("zh", "add"), ("en", "add"), (None, "add")],
)
def test_official_emilia_rejects_non_parity_overrides(
    lang_arg: str | None,
    punctuation: str,
) -> None:
    with pytest.raises(ValueError, match="official-emilia"):
        resolve_prompt_contract(
            "official-emilia",
            lang_arg=lang_arg,
            tsv_path="/datasets/seedtts/zh/test.tsv",
            ref_text_punctuation_arg=punctuation,
        )


def test_official_emilia_audio_normalizes_low_rms_then_hop_aligns() -> None:
    waveform = np.full((1, 10), 0.01, dtype=np.float32)

    result = prepare_official_emilia_audio(waveform, hop_length=4)

    assert result.original_rms == pytest.approx(0.01)
    assert result.truncated_samples == 2
    assert result.waveform.shape == (1, 8)
    assert result.waveform.dtype == np.float32
    np.testing.assert_allclose(result.waveform, 0.1, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(waveform, np.full((1, 10), 0.01, dtype=np.float32))


def test_official_emilia_audio_does_not_scale_rms_at_threshold() -> None:
    waveform = np.full((1, 8), 0.1, dtype=np.float32)

    result = prepare_official_emilia_audio(waveform, hop_length=4)

    assert result.original_rms == pytest.approx(0.1)
    assert result.truncated_samples == 0
    assert result.waveform is waveform


def test_official_emilia_audio_fails_if_hop_alignment_empties_prompt() -> None:
    waveform = np.ones((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="becomes empty"):
        prepare_official_emilia_audio(waveform, hop_length=4)


def test_official_emilia_restores_quiet_reference_volume() -> None:
    generated = np.array([0.5, -0.25], dtype=np.float32)

    restored = restore_official_emilia_output_rms(
        generated,
        original_ref_rms=0.02,
    )

    np.testing.assert_allclose(restored, np.array([0.1, -0.05], dtype=np.float32))
    np.testing.assert_array_equal(generated, np.array([0.5, -0.25], dtype=np.float32))


def test_official_emilia_does_not_rescale_loud_reference_output() -> None:
    generated = np.array([0.5, -0.25], dtype=np.float32)

    restored = restore_official_emilia_output_rms(
        generated,
        original_ref_rms=0.1,
    )

    assert restored is generated


def test_prompt_contract_helpers_do_not_advance_numpy_rng() -> None:
    np.random.seed(1234)
    before = np.random.get_state()

    resolved = resolve_prompt_contract(
        "official-emilia",
        lang_arg=None,
        tsv_path="/datasets/seedtts/en/test.tsv",
    )
    prepare_reference_text("Reference", resolved.ref_text_punctuation)
    prepare_official_emilia_audio(
        np.full((1, 8), 0.01, dtype=np.float32),
        hop_length=4,
    )

    after = np.random.get_state()
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]
