#!/usr/bin/env python3
"""CPU tests for E1.1 targeted corruption + scheduler mix (no GPU needed).

Run:  python tests/test_elastic_targeted.py
"""

import random

import torch

from omnivoice.elastic import (
    corrupt_audio_region_targeted,
    elastic_ids,
    execute_structure_ops,
)

C, MASK = 4, 16
EXPAND, DELETE = elastic_ids(MASK)


def _mk(T, prompt, rng):
    tokens = torch.randint(0, MASK, (C, T))
    inputs, labels = tokens.clone(), tokens.clone()
    m = torch.rand(C, T - prompt) < rng.random()
    inputs[:, prompt:][m] = MASK
    labels[:, prompt:][~m] = -100
    labels[:, :prompt] = -100
    return inputs, labels


def test_delta_distribution():
    """Canvas/content length ratio must cover U(0.7, 1.3), both tails."""
    rng = random.Random(7)
    ratios = []
    for _ in range(1000):
        T = rng.randint(40, 300)
        prompt = rng.randint(0, T // 3)
        inputs, labels = _mk(T, prompt, rng)
        ci, cl, _ = corrupt_audio_region_targeted(
            inputs, labels, prompt, MASK, delta_max=0.3, rng=rng
        )
        gen_content = T - prompt
        gen_canvas = ci.shape[1] - prompt
        ratios.append(gen_canvas / gen_content)
    ratios.sort()
    q = lambda p: ratios[int(p * len(ratios))]
    assert 0.68 <= q(0.01) <= 0.80, f"P1={q(0.01)}"
    assert 1.20 <= q(0.99) <= 1.32, f"P99={q(0.99)}"
    assert q(0.10) < 0.88, f"short tail missing: P10={q(0.10)}"
    assert q(0.90) > 1.12, f"long tail missing: P90={q(0.90)}"
    print(f"test_delta_distribution OK  P1/P10/P50/P90/P99 = "
          f"{q(0.01):.3f}/{q(0.10):.3f}/{q(0.50):.3f}/{q(0.90):.3f}/{q(0.99):.3f}")


def test_oracle_roundtrip():
    """Committing all labels then executing ops must recover content length."""
    rng = random.Random(11)
    for _ in range(300):
        T = rng.randint(20, 200)
        prompt = rng.randint(0, T // 3)
        inputs, labels = _mk(T, prompt, rng)
        ci, cl, _ = corrupt_audio_region_targeted(
            inputs, labels, prompt, MASK, delta_max=0.3, rng=rng
        )
        # oracle model: write cb0 special labels onto the canvas, fill every
        # other masked cell with a real token
        canvas = ci.clone()
        gen = canvas[:, prompt:]
        gl = cl[:, prompt:]
        spec = (gl[0] == EXPAND) | (gl[0] == DELETE)
        gen[0, spec] = gl[0, spec]
        gen[gen == MASK] = 0
        out, n_e, n_d = execute_structure_ops(gen, MASK)
        # each executed expand adds net +1 col (1->2 masks), each delete -1;
        # canvas +- ops must land exactly on the true content length
        expected = T - prompt
        assert out.shape[1] == expected, (
            f"roundtrip {out.shape[1]} != {expected} (T={T} prompt={prompt} "
            f"e={n_e} d={n_d})"
        )
    print("test_oracle_roundtrip OK")


def test_uncond_carries_corruption():
    """drop_cond (uncond/CFG branch) samples must still carry special targets."""
    from omnivoice.data.processor import OmniVoiceElasticSampleProcessor

    class FakeTok:
        pad_token_id = 0

        def __call__(self, text, return_tensors=None):
            class R:
                input_ids = torch.randint(1, 50, (1, max(1, len(text) // 4)))

            return R()

    proc = OmniVoiceElasticSampleProcessor(
        text_tokenizer=FakeTok(),
        num_channels=C,
        audio_mask_id=MASK,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=1.0,  # force uncond
        language_ratio=0.0,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
        p_elastic=1.0,
        elastic_mode="targeted",
    )
    hit = 0
    for i in range(50):
        s = proc({
            "audio_tokens": torch.randint(0, MASK, (C, 120)),
            "label": {"text": "x" * 40, "language_id": "zh"},
        })
        lab = s["labels"]
        if ((lab == EXPAND) | (lab == DELETE)).any():
            hit += 1
    assert hit >= 40, f"uncond corruption too rare: {hit}/50"
    print(f"test_uncond_carries_corruption OK ({hit}/50 samples carry specials)")


def test_scheduler_mix_regime():
    """With mix=1.0 every elastic sample must be in the low-mask regime."""
    from unittest.mock import patch
    from omnivoice.data.processor import OmniVoiceElasticSampleProcessor

    seen = []
    orig_uniform = random.uniform

    def spy(a, b):
        v = orig_uniform(a, b)
        if (a, b) == (0.0, 0.5):
            seen.append(v)
        return v

    class FakeTok:
        pad_token_id = 0

        def __call__(self, text, return_tensors=None):
            class R:
                input_ids = torch.randint(1, 50, (1, 8))

            return R()

    proc = OmniVoiceElasticSampleProcessor(
        text_tokenizer=FakeTok(),
        num_channels=C,
        audio_mask_id=MASK,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.0,
        language_ratio=0.0,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
        p_elastic=1.0,
        elastic_mode="targeted",
        elastic_scheduler_mix=1.0,
    )
    with patch("omnivoice.data.processor.random.uniform", side_effect=spy):
        for _ in range(20):
            proc({
                "audio_tokens": torch.randint(0, MASK, (C, 80)),
                "label": {"text": "y" * 30, "language_id": "zh"},
            })
    assert len(seen) == 20, f"low-mask draw not hit every sample: {len(seen)}/20"
    print("test_scheduler_mix_regime OK")


if __name__ == "__main__":
    test_delta_distribution()
    test_oracle_roundtrip()
    test_uncond_carries_corruption()
    test_scheduler_mix_regime()
    print("ALL targeted tests OK")
