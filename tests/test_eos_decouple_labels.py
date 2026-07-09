"""Label-construction tests for the EOS/padding decoupling scheme
(eos_decouple_silence). Runnable via plain python (no pytest):

    PYTHONPATH=. python -c "import tests.test_eos_decouple_labels as t; t.run()"

Covers: (a) flag-off == v1 byte-identical tail (single-column-window regression),
(b) flag-on placement on all codebooks, (c) canvas-edge clipping (void=1),
(d) T at block boundary (void=32), (e) drop_text/uncond branch unaffected.
"""
import random
import types

import torch

from omnivoice.blockdiff_dual import (
    OmniVoiceBlockDualSampleProcessor,
    SILENCE_FRAME_TOKENS,
    block_eos_id,
)

BS = 32
MASK_ID = 1024
EOS = block_eos_id(MASK_ID)
C = 8


class _FakeTok:
    def __call__(self, s, return_tensors="pt"):
        return types.SimpleNamespace(input_ids=torch.ones(1, 3, dtype=torch.long))


def _proc(decouple, drop_cond_ratio=0.0, window=32):
    p = OmniVoiceBlockDualSampleProcessor.__new__(OmniVoiceBlockDualSampleProcessor)
    p.block_size = BS
    p.turn_boundary_prompt_prob = 0.0
    p.eos_decouple_silence = decouple
    p.silence_void_window = window
    p.audio_mask_id = MASK_ID
    p.num_channels = C
    p.drop_cond_ratio = drop_cond_ratio
    p.prompt_ratio_range = (0.3, 0.3)
    p.mask_ratio_range = (0.0, 1.0)
    p.language_ratio = 0.0
    p.use_pinyin_ratio = 0.0
    p.instruct_ratio = 0.0
    p.only_instruct_ratio = 0.0
    p.text_tokenizer = _FakeTok()
    return p


def _sample(T):
    torch.manual_seed(7)
    return {
        "audio_tokens": torch.randint(0, 1024, (C, T)),
        "label": {"text": "hello world"},
    }


def _tail(out, T):
    labels = out["labels"]
    canvas_len = ((T // BS) + 1) * BS
    noisy_labels = labels[:, -canvas_len:]
    return noisy_labels, canvas_len


def _run_pair(T, window=32, drop_cond_ratio=0.0):
    s = _sample(T)
    random.seed(123)
    torch.manual_seed(123)
    off = _proc(False, drop_cond_ratio)(dict(s))
    random.seed(123)
    torch.manual_seed(123)
    on = _proc(True, drop_cond_ratio, window)(dict(s))
    return off, on


def test_flag_off_is_v1():
    T = 70  # void = 26
    off, _ = _run_pair(T)
    nl, canvas_len = _tail(off, T)
    assert nl[0, T] == EOS and nl[0, T + 1] == EOS
    assert (nl[0, T:T + 4] == EOS).all()
    assert (nl[0, T + 4:] == -100).all()
    assert (nl[1:, T:] == -100).all()


def test_flag_on_placement_and_content_region_unchanged():
    T = 70
    off, on = _run_pair(T)
    nl_off, canvas_len = _tail(off, T)
    nl_on, _ = _tail(on, T)
    # identical RNG => content region labels identical
    assert torch.equal(nl_off[:, :T], nl_on[:, :T])
    # single eos column on cb0
    assert nl_on[0, T] == EOS
    assert (nl_on[0, T + 1:] != EOS).all()
    # silence supervision on ALL codebooks over the void
    v_hi = min(T + 1 + 32, canvas_len)
    for cb in range(C):
        assert (nl_on[cb, T + 1:v_hi] == SILENCE_FRAME_TOKENS[cb]).all(), cb
    assert (nl_on[:, v_hi:] == -100).all()
    # inputs beyond T are mask on the noisy canvas (both flags)
    ii = on["input_ids"][:, -canvas_len:]
    assert (ii[:, T:] == MASK_ID).all()


def test_edge_void_of_one():
    T = BS * 2 - 1  # 63 -> canvas 64, void = 1 (only the eos column fits)
    _, on = _run_pair(T)
    nl, canvas_len = _tail(on, T)
    assert canvas_len == T + 1
    assert nl[0, T] == EOS
    # no silence columns exist; nothing out of bounds happened
    assert nl.shape[1] == canvas_len


def test_block_boundary_void_of_32():
    T = BS * 2  # 64 -> canvas 96, void = 32
    _, on = _run_pair(T)
    nl, canvas_len = _tail(on, T)
    assert canvas_len == T + BS
    assert nl[0, T] == EOS
    for cb in range(C):
        assert (nl[cb, T + 1:canvas_len] == SILENCE_FRAME_TOKENS[cb]).all()


def test_drop_text_branch():
    T = 70
    _, on = _run_pair(T, drop_cond_ratio=1.0)  # forces drop_cond/drop_text
    nl, canvas_len = _tail(on, T)
    assert nl[0, T] == EOS
    v_hi = min(T + 1 + 32, canvas_len)
    for cb in range(C):
        assert (nl[cb, T + 1:v_hi] == SILENCE_FRAME_TOKENS[cb]).all()
    # drop_text => no text prefix: total length = clean_len + canvas_len
    clean_len = (T // BS) * BS
    assert on["input_ids"].shape[1] == clean_len + canvas_len


def run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(len(fns), "tests passed")


if __name__ == "__main__":
    run()
