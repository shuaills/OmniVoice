"""Prompt contracts for the blockwise Seed-TTS generator.

This module is intentionally CPU-only so the prompt preparation contract can be
locked without importing the GPU generator script (which parses its CLI at
module import time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from omnivoice.utils.text import add_punctuation


PromptContractName = Literal["current", "official-emilia"]
RefTextPunctuation = Literal["add", "preserve"]


@dataclass(frozen=True)
class PromptContract:
    """Resolved prompt behavior used for one generator invocation."""

    name: PromptContractName
    language: str | None
    ref_text_punctuation: RefTextPunctuation


@dataclass(frozen=True)
class PreparedReferenceAudio:
    """Official prompt audio plus provenance needed by evaluation metadata."""

    waveform: np.ndarray
    original_rms: float
    truncated_samples: int


def _is_none_language(value: str) -> bool:
    return value.strip().casefold() in {"none", "null"}


def resolve_prompt_contract(
    name: str,
    *,
    lang_arg: str | None,
    tsv_path: str,
    ref_text_punctuation_arg: str = "contract",
) -> PromptContract:
    """Resolve prompt preprocessing and the independently selected language.

    ``lang_arg is None`` means the CLI flag was omitted.  The literal CLI value
    ``--lang None`` is represented by the string ``"None"`` and therefore can
    explicitly disable language conditioning.  Language is intentionally
    orthogonal to prompt preprocessing so experiments can isolate prompt-only,
    language-only, and combined effects.
    """

    if name not in {"current", "official-emilia"}:
        raise ValueError(
            f"unknown prompt contract {name!r}; expected current or official-emilia"
        )
    if ref_text_punctuation_arg not in {"contract", "add", "preserve"}:
        raise ValueError(
            "ref_text_punctuation_arg must be contract, add, or preserve; "
            f"got {ref_text_punctuation_arg!r}"
        )

    if lang_arg is None:
        language = "zh" if "/zh/" in tsv_path else "en"
    elif _is_none_language(lang_arg):
        language = None
    else:
        language = lang_arg

    if name == "current":
        punctuation: RefTextPunctuation = (
            "add"
            if ref_text_punctuation_arg == "contract"
            else ref_text_punctuation_arg
        )
        return PromptContract(
            name="current",
            language=language,
            ref_text_punctuation=punctuation,
        )

    if ref_text_punctuation_arg == "add":
        raise ValueError(
            "--prompt-contract official-emilia preserves ref_text exactly; "
            "use --ref-text-punctuation preserve (or contract)"
        )
    return PromptContract(
        name="official-emilia",
        language=language,
        ref_text_punctuation="preserve",
    )


def prepare_reference_text(
    ref_text: str,
    punctuation: RefTextPunctuation,
) -> str:
    """Apply the resolved reference-text boundary policy."""

    if punctuation == "add":
        return add_punctuation(ref_text)
    if punctuation == "preserve":
        return ref_text
    raise ValueError(f"unknown reference-text punctuation policy {punctuation!r}")


def prepare_official_emilia_audio(
    waveform: np.ndarray,
    *,
    hop_length: int,
) -> PreparedReferenceAudio:
    """Mirror ``create_voice_clone_prompt(..., preprocess_prompt=False)``.

    The caller must provide the same mono, resampled ``float32`` waveform that
    :func:`omnivoice.utils.audio.load_audio` returns.  The released path first
    raises low-RMS prompts to 0.1, then truncates the tail to an exact tokenizer
    hop boundary.  Silence removal and long-audio trimming are deliberately not
    part of this contract.
    """

    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"waveform must be a numpy array, got {type(waveform)!r}")
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            "waveform must have mono channels-first shape (1, samples), got "
            f"{waveform.shape}"
        )
    if waveform.dtype != np.float32:
        raise ValueError(
            "official-emilia waveform must be float32 as returned by load_audio; "
            f"got {waveform.dtype}"
        )
    if waveform.shape[-1] == 0:
        raise ValueError("reference audio is empty")
    if not np.isfinite(waveform).all():
        raise ValueError("reference audio contains NaN or infinity")
    if not isinstance(hop_length, (int, np.integer)) or int(hop_length) <= 0:
        raise ValueError(f"hop_length must be a positive integer, got {hop_length!r}")

    # Keep the operation order and numpy float32 arithmetic aligned with
    # OmniVoice.create_voice_clone_prompt().
    original_rms = float(np.sqrt(np.mean(waveform**2)))
    prepared = waveform
    if 0 < original_rms < 0.1:
        prepared = prepared * 0.1 / original_rms

    truncated_samples = int(prepared.shape[-1] % int(hop_length))
    if truncated_samples > 0:
        prepared = prepared[:, :-truncated_samples]
    if prepared.shape[-1] == 0:
        raise ValueError(
            "reference audio becomes empty after tokenizer hop alignment: "
            f"samples={waveform.shape[-1]} hop_length={int(hop_length)}"
        )

    return PreparedReferenceAudio(
        waveform=prepared,
        original_rms=original_rms,
        truncated_samples=truncated_samples,
    )


def restore_official_emilia_output_rms(
    waveform: np.ndarray,
    *,
    original_ref_rms: float,
) -> np.ndarray:
    """Mirror the reference-volume step in ``_post_process_audio``.

    The official prompt path temporarily raises quiet references to RMS 0.1
    before tokenization, then scales generated audio back to the original
    reference level.  This helper intentionally does not claim full output
    post-processing parity (silence removal, fade, or edge padding).
    """

    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"waveform must be a numpy array, got {type(waveform)!r}")
    if not np.isfinite(original_ref_rms) or original_ref_rms < 0:
        raise ValueError(
            "original_ref_rms must be a finite non-negative value, got "
            f"{original_ref_rms!r}"
        )
    if original_ref_rms < 0.1:
        return waveform * original_ref_rms / 0.1
    return waveform
