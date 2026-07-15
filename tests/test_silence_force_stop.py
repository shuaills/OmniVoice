"""CPU tests for the blockwise digital-silence force-stop detector."""

import torch

from omnivoice.blockdiff_dual import (
    SILENCE_FRAME_TOKENS,
    _choose_termination,
    _decode_block_causal,
    _find_silence_run_start,
)


def _tokens(frames: int = 80) -> torch.Tensor:
    values = torch.arange(8 * frames, dtype=torch.long).reshape(8, frames)
    return values % 1024


def test_detects_earliest_run_and_ignores_later_junk():
    tokens = _tokens()
    tokens[:2, 20:45] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    tokens[:, 45:] = 17  # a click/junk after the qualifying quiet interval
    assert _find_silence_run_start(tokens, 25, match_codebooks=2) == 20


def test_upper_codebook_dither_does_not_hide_silence():
    tokens = _tokens(50)
    tokens[:2, 10:35] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    tokens[2:, 10:35] = torch.arange(6).unsqueeze(1) + torch.arange(25)
    assert _find_silence_run_start(tokens, 25, match_codebooks=2) == 10
    assert _find_silence_run_start(tokens, 25, match_codebooks=8) is None


def test_subthreshold_and_interrupted_runs_do_not_stop():
    tokens = _tokens(60)
    tokens[:2, 5:29] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    assert _find_silence_run_start(tokens, 25, match_codebooks=2) is None
    tokens[:2, 30:55] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    tokens[0, 42] = 999
    assert _find_silence_run_start(tokens, 25, match_codebooks=2) is None


def test_run_can_cross_a_block_boundary():
    tokens = _tokens(64)
    tokens[:2, 20:45] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    # The qualifying run crosses the 32-frame boundary.  Calling once on the
    # committed prefix cannot stop; the next committed block can.
    assert _find_silence_run_start(
        tokens[:, :32], 25, match_codebooks=2
    ) is None
    assert _find_silence_run_start(tokens, 25, match_codebooks=2) == 20


def test_prompt_silence_is_outside_the_detector_input():
    prompt = _tokens(40)
    prompt[:2] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    generated = _tokens(60)
    generated[:2, 24:49] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    committed = torch.cat([prompt, generated], dim=1)
    assert _find_silence_run_start(
        committed[:, prompt.shape[1]:], 25, match_codebooks=2
    ) == 24


def test_minimum_generation_region_is_never_trimmed():
    tokens = _tokens(70)
    tokens[:2, :40] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    assert (
        _find_silence_run_start(
            tokens, 25, match_codebooks=2, start_frame=24
        )
        is None
    )
    tokens[:2, 24:49] = SILENCE_FRAME_TOKENS[:2].unsqueeze(1)
    assert (
        _find_silence_run_start(
            tokens, 25, match_codebooks=2, start_frame=24
        )
        == 24
    )


def test_invalid_detector_configuration_fails_fast():
    tokens = _tokens(10)
    for kwargs in (
        {"min_run_frames": 0},
        {"min_run_frames": 2, "match_codebooks": 0},
        {"min_run_frames": 2, "match_codebooks": 9},
    ):
        try:
            _find_silence_run_start(tokens, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_stop_order_uses_observation_time_but_trims_silence_to_onset():
    # Silence [20, 44] is only known at 44.  EOS at 40 wins even though a
    # silence-triggered return would have trimmed farther back to 20.
    assert _choose_termination(40, 20, 25) == ("eos", 40, 40)
    # EOS after the threshold loses; the fallback trims the verified wait.
    assert _choose_termination(50, 20, 25) == ("silence", 20, 44)
    # At the final silence frame, prefer normal model termination.
    assert _choose_termination(44, 20, 25) == ("eos", 44, 44)


def test_stop_choice_handles_single_signal_and_disabled_path():
    assert _choose_termination(17, None, 0) == ("eos", 17, 17)
    assert _choose_termination(None, 12, 4) == ("silence", 12, 15)
    assert _choose_termination(None, None, 0) == (None, None, None)


def test_disabled_decoder_path_is_token_and_rng_identical():
    from transformers import Qwen3Config

    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceConfig,
        OmniVoiceGenerationConfig,
    )

    llm_cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=2048,
        max_position_embeddings=128,
    )
    cfg = OmniVoiceConfig(
        audio_vocab_size=1026,
        audio_mask_id=1024,
        num_audio_codebook=8,
        llm_config=llm_cfg,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(91)
    model = OmniVoice(cfg).double().eval()
    gen = OmniVoiceGenerationConfig()
    gen.guidance_scale = 0.0
    gen.class_temperature = 0.0
    gen.position_temperature = 0.0
    prefix = torch.randint(1, 200, (8, 4))

    rng = torch.get_rng_state()
    implicit, implicit_stats = _decode_block_causal(
        model,
        prefix,
        gen,
        block_size=4,
        max_blocks=2,
        num_step_per_block=2,
        use_kv_cache=False,
    )
    implicit_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    explicit, explicit_stats = _decode_block_causal(
        model,
        prefix,
        gen,
        block_size=4,
        max_blocks=2,
        num_step_per_block=2,
        use_kv_cache=False,
        silence_run_frames=0,
        silence_match_codebooks=2,
    )
    explicit_rng = torch.get_rng_state()

    assert torch.equal(implicit, explicit)
    assert implicit_stats == explicit_stats
    assert torch.equal(implicit_rng, explicit_rng)
