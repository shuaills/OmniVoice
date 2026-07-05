#!/usr/bin/env python3
"""CPU smoke tests for the elastic canvas (no GPU / no checkpoint required).

Run:  python tests/test_elastic_smoke.py
"""

import random

import torch

from omnivoice.elastic import (
    NUM_ELASTIC_CLASSES,
    corrupt_audio_region,
    elastic_ids,
    execute_structure_ops,
    migrate_state_dict,
)

C, V, MASK = 4, 17, 16  # toy: vocab 17 (16 codes + mask), expand=17, delete=18
EXPAND, DELETE = elastic_ids(MASK)


def test_corruption_invariants():
    rng = random.Random(0)
    for trial in range(200):
        T = rng.randint(6, 80)
        prompt = rng.randint(0, T // 2)
        tokens = torch.randint(0, MASK, (C, T))
        inputs, labels = tokens.clone(), tokens.clone()
        m = torch.rand(C, T - prompt) < 0.6
        inputs[:, prompt:][m] = MASK
        labels[:, prompt:][~m] = -100
        labels[:, :prompt] = -100

        ci, cl, w = corrupt_audio_region(
            inputs, labels, prompt, MASK, merge_prob=0.2, insert_prob=0.1,
            end_append_max_ratio=0.3, rng=rng,
        )
        assert ci.shape == cl.shape == w.shape and ci.shape[0] == C
        # prompt region untouched
        assert torch.equal(ci[:, :prompt], inputs[:, :prompt])
        # special labels only on row 0; their columns fully masked in input
        specials = (cl == EXPAND) | (cl == DELETE)
        assert not specials[1:].any()
        for j in specials[0].nonzero(as_tuple=True)[0].tolist():
            assert (ci[:, j] == MASK).all()
            assert (cl[1:, j] == -100).all()
        # delete weights sum to <= 1 (1/N each)
        dw = w[cl == DELETE]
        if dw.numel():
            assert abs(dw.sum().item() - 1.0) < 1e-5
        # non-special supervised cells keep weight 1
        normal = (cl != -100) & ~specials
        assert (w[normal] == 1.0).all()
    print("test_corruption_invariants OK")


def test_structure_ops():
    t = torch.randint(0, MASK, (C, 6))
    t[0, 1] = EXPAND
    t[0, 4] = DELETE
    t[:, 5] = MASK  # fully-masked suffix after the delete -> broadcast
    out, ne, nd = execute_structure_ops(t, MASK)
    assert ne == 1 and nd == 1
    # col1 -> 2 mask cols (len +1), col4+col5 dropped by broadcast (len -2)
    assert out.shape[1] == 6 + 1 - 2
    assert (out[:, 1] == MASK).all() and (out[:, 2] == MASK).all()
    # no specials left
    assert not ((out[0] == EXPAND) | (out[0] == DELETE)).any()
    # idempotent when no specials
    out2, ne2, nd2 = execute_structure_ops(out, MASK)
    assert ne2 == nd2 == 0 and torch.equal(out, out2)
    print("test_structure_ops OK")


def test_migration_preserves_logits():
    H = 8
    sd = {
        "audio_embeddings.weight": torch.randn(C * V, H),
        "audio_heads.weight": torch.randn(C * V, H),
        "audio_heads.bias": torch.randn(C * V),
    }
    new = migrate_state_dict({k: v.clone() for k, v in sd.items()}, C, V)
    NV = V + NUM_ELASTIC_CLASSES
    assert new["audio_embeddings.weight"].shape[0] == C * NV
    h = torch.randn(3, H)
    for c in range(C):
        # embeddings: every original row lands at the remapped index
        for v in (0, V // 2, V - 1):
            assert torch.equal(
                sd["audio_embeddings.weight"][c * V + v],
                new["audio_embeddings.weight"][c * NV + v],
            )
        # head logits identical on the original vocab slice
        old_logits = h @ sd["audio_heads.weight"].T + sd["audio_heads.bias"]
        new_logits = h @ new["audio_heads.weight"].T + new["audio_heads.bias"]
        assert torch.allclose(
            old_logits[:, c * V : (c + 1) * V],
            new_logits[:, c * NV : c * NV + V],
        )
    print("test_migration_preserves_logits OK")


def test_elastic_processor_and_collator():
    from omnivoice.data.collator import PaddingDataCollator
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
        drop_cond_ratio=0.1,
        language_ratio=0.5,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.5,
        only_instruct_ratio=0.5,
        p_elastic=1.0,
    )
    samples = []
    for i in range(4):
        s = proc(
            {
                "audio_tokens": torch.randint(0, MASK, (C, 30 + 7 * i)),
                "label": {"text": "hello world example", "language_id": "en"},
            }
        )
        assert "loss_weights" in s and s["loss_weights"].shape == s["labels"].shape
        samples.append(s)
    batch = PaddingDataCollator(proc, 4096)(samples)
    assert "loss_weights" in batch
    assert batch["loss_weights"].shape == batch["labels"].shape
    # padded region carries zero weight
    for i, s in enumerate(samples):
        assert (batch["loss_weights"][i, :, s["length"] :] == 0).all()
    print("test_elastic_processor_and_collator OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    test_corruption_invariants()
    test_structure_ops()
    test_migration_preserves_logits()
    test_elastic_processor_and_collator()
    print("ALL ELASTIC SMOKE TESTS PASSED")
