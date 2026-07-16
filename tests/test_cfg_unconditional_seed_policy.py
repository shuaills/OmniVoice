"""CPU/mock contracts for CFG unconditional reference-seed policies."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import omnivoice.blockdiff as blockdiff  # noqa: E402
import omnivoice.blockdiff_dual as dual  # noqa: E402
import omnivoice.models.omnivoice as omnivoice_model  # noqa: E402
import transformers  # noqa: E402


C = 8
MASK = 1024
VOCAB = MASK + 2
BLOCK_SIZE = 4
PREFIX_LEN = 3


class _FakeCache:
    def __init__(self):
        self.length = 0
        self.role = None

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = length


class _FakeModel:
    device = torch.device("cpu")
    config = SimpleNamespace(
        num_audio_codebook=C,
        audio_mask_id=MASK,
        audio_vocab_size=VOCAB,
    )


def _gen_config():
    return SimpleNamespace(
        guidance_scale=2.0,
        t_shift=1.0,
        layer_penalty_factor=0.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )


def _inputs():
    prefix = torch.arange(C * PREFIX_LEN).view(C, PREFIX_LEN) + 20
    # Distinct high-valued sentinels make any accidental full or partial
    # reference exposure in the drop-ref u branch directly observable.
    seed = torch.arange(C * 6).view(C, 6) + 700
    return prefix, seed


def _expected_generated():
    # Conditional positions are P + seed_total .. P + final generated frame.
    positions = torch.arange(PREFIX_LEN + 6, PREFIX_LEN + 12)
    codebook_offsets = 3 * torch.arange(C).view(C, 1)
    return 10 + (positions.view(1, -1) + codebook_offsets) % 100


def _install_mock_decode(patches):
    records = {"forward": [], "cfg_shapes": [], "masks": []}

    patches.setattr(transformers, "DynamicCache", _FakeCache)
    patches.setattr(
        omnivoice_model,
        "_get_time_steps",
        lambda t_start, t_end, num_step, t_shift: torch.linspace(
            t_start, t_end, num_step + 1
        ),
    )

    def logits_for_positions(positions):
        length = positions.numel()
        logits = torch.full((1, C, length, VOCAB), -20.0)
        for cb in range(C):
            targets = 10 + (positions + 3 * cb) % 100
            logits[0, cb, torch.arange(length), targets] = 20.0
        return logits

    def forward_text_prefix(model, text_ids, positions, attn4d, cache):
        assert cache.role is None
        cache.role = "c"
        cache.length += text_ids.size(1)

    def forward_slices(
        model, ids, positions, attn4d, past_key_values=None
    ):
        role = "recompute"
        before = None
        cache_id = None
        if past_key_values is not None:
            if past_key_values.role is None:
                past_key_values.role = "u"
            role = past_key_values.role
            before = past_key_values.length
            cache_id = id(past_key_values)
            past_key_values.length += ids.size(1)
        records["forward"].append(
            {
                "role": role,
                "cache_id": cache_id,
                "before": before,
                "ids": ids.detach().clone(),
                "positions": positions.detach().clone(),
            }
        )
        return logits_for_positions(positions)

    def forward_slices_mixed(model, ids, prefix_len, positions, attn4d):
        records["forward"].append(
            {
                "role": "conditional_recompute",
                "cache_id": None,
                "before": None,
                "ids": ids.detach().clone(),
                "positions": positions.detach().clone(),
            }
        )
        return logits_for_positions(positions)

    def predict(model, c_logits, u_logits, gen_config):
        records["cfg_shapes"].append(
            (c_logits.shape[2], u_logits.shape[2])
        )
        return c_logits.argmax(dim=-1), c_logits.max(dim=-1).values

    original_mask_builder = dual.build_block_causal_attn_mask

    def record_mask(document_ids, copy_tags, block_ids):
        records["masks"].append(
            {
                "tags": copy_tags.detach().clone(),
                "blocks": block_ids.detach().clone(),
            }
        )
        return original_mask_builder(document_ids, copy_tags, block_ids)

    patches.setattr(dual, "_forward_text_prefix", forward_text_prefix)
    patches.setattr(dual, "_forward_slices", forward_slices)
    patches.setattr(dual, "_forward_slices_mixed", forward_slices_mixed)
    patches.setattr(
        dual, "build_block_causal_attn_mask", record_mask
    )
    patches.setattr(blockdiff, "_predict_tokens_blockwise", predict)
    return records


def _decode(*, use_cache, policy=None):
    prefix, seed = _inputs()
    kwargs = {}
    if policy is not None:
        kwargs["cfg_unconditional_seed_policy"] = policy
    return dual._decode_block_causal(
        _FakeModel(),
        prefix,
        _gen_config(),
        block_size=BLOCK_SIZE,
        max_blocks=3,
        num_step_per_block=1,
        use_kv_cache=use_cache,
        seed_audio=seed,
        min_gen_frames=100,
        **kwargs,
    )


class TestCfgUnconditionalSeedPolicy(unittest.TestCase):
    def setUp(self):
        self._patches = []

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()

    def setattr(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self._patches.append(patcher)

    def test_shared_default_freezes_legacy_output_and_cache_geometry(self):
        records = _install_mock_decode(self)

        cache_out, cache_stats = _decode(use_cache=True)
        shared_out, shared_stats = _decode(use_cache=True, policy="shared")
        recompute_out, recompute_stats = _decode(use_cache=False)

        expected = _expected_generated()
        self.assertTrue(torch.equal(cache_out, expected))
        self.assertTrue(torch.equal(shared_out, expected))
        self.assertTrue(torch.equal(recompute_out, expected))
        self.assertEqual(cache_stats, shared_stats)
        self.assertEqual(cache_stats, recompute_stats)
        self.assertEqual(
            cache_stats,
            {
                "n_blocks": 3,
                "stopped_by_eos": False,
                "eos_col": None,
                "stopped_by_silence": False,
                "silence_col": None,
                "silence_trigger_col": None,
                "stop_reason": None,
            },
        )

        # Each shared cache decode has the legacy u sequence: full seed block,
        # partial-seed-containing first current block, then the next full block.
        u_calls = [r for r in records["forward"] if r["role"] == "u"]
        self.assertEqual(
            [r["ids"].size(1) for r in u_calls], [4, 4, 4, 4, 4] * 2
        )
        self.assertEqual(
            [r["before"] for r in u_calls], [0, 4, 4, 8, 8] * 2
        )
        self.assertEqual(
            [int(r["positions"][0]) for r in u_calls],
            [0, 4, 4, 8, 8] * 2,
        )
        self.assertTrue(
            all(
                c_shape == u_shape
                for c_shape, u_shape in records["cfg_shapes"]
            )
        )


    def test_drop_ref_uses_generated_only_cache_and_partial_timeline(self):
        records = _install_mock_decode(self)
        cache_out, cache_stats = _decode(
            use_cache=True, policy="drop_ref"
        )

        expected = _expected_generated()
        self.assertTrue(torch.equal(cache_out, expected))
        self.assertEqual(cache_stats["n_blocks"], 3)

        u_calls = [r for r in records["forward"] if r["role"] == "u"]
        # The first target suffix has only 2 frames because the reference
        # supplies the first 2 conditional columns.  It starts a fresh u
        # timeline at 0..1; the next target block starts at position 2.
        self.assertEqual(
            [r["ids"].size(1) for r in u_calls], [2, 2, 4, 4]
        )
        self.assertEqual([r["before"] for r in u_calls], [0, 0, 2, 2])
        self.assertEqual(
            [r["positions"].tolist() for r in u_calls],
            [[0, 1], [0, 1], [2, 3, 4, 5], [2, 3, 4, 5]],
        )
        seed_values = set(_inputs()[1].flatten().tolist())
        for record in u_calls:
            self.assertTrue(
                seed_values.isdisjoint(record["ids"].flatten().tolist())
            )
        self.assertEqual(records["cfg_shapes"], [(2, 2), (4, 4)])


    def test_drop_ref_cache_matches_recompute_and_block_ids(self):
        records = _install_mock_decode(self)
        cache_out, cache_stats = _decode(
            use_cache=True, policy="drop_ref"
        )
        recompute_out, recompute_stats = _decode(
            use_cache=False, policy="drop_ref"
        )

        self.assertTrue(torch.equal(cache_out, recompute_out))
        self.assertEqual(cache_stats, recompute_stats)

        u_masks = [
            record
            for record in records["masks"]
            if not (record["tags"] == dual.TAG_PREFIX).any()
        ]
        self.assertEqual(len(u_masks), 2)
        self.assertEqual(
            u_masks[0]["tags"].tolist(), [dual.TAG_NOISY] * 2
        )
        self.assertEqual(u_masks[0]["blocks"].tolist(), [0, 0])
        self.assertEqual(
            u_masks[1]["tags"].tolist(),
            [
                dual.TAG_CLEAN,
                dual.TAG_CLEAN,
                *([dual.TAG_NOISY] * 4),
            ],
        )
        self.assertEqual(
            u_masks[1]["blocks"].tolist(), [0, 0, 1, 1, 1, 1]
        )


    def test_rejects_unknown_seed_policy(self):
        _install_mock_decode(self)
        with self.assertRaisesRegex(
            ValueError, "must be 'shared' or 'drop_ref'"
        ):
            _decode(use_cache=True, policy="unknown")


if __name__ == "__main__":
    unittest.main()
