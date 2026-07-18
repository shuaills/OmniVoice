#!/usr/bin/env python3
"""Static and behavioral checks for CFG90100 performance receipts."""

import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from omnivoice.data.batching import PackingIterableDataset
except ModuleNotFoundError:
    if os.environ.get("REQUIRE_OMNIVOICE_RUNTIME") == "1":
        raise
    PackingIterableDataset = None


CONFIG_DIR = ROOT / "examples/config"
REFERENCE = CONFIG_DIR / "train_config_cfg90100_band4_300k.json"
BUILDER = ROOT / "omnivoice/training/builder.py"
CONFIGS = {
    "base": CONFIG_DIR / "train_config_cfg90100_band4_perf8_base.json",
    "nogc": CONFIG_DIR / "train_config_cfg90100_band4_perf8_nogc.json",
    "nogc_bal8": CONFIG_DIR / "train_config_cfg90100_band4_perf8_nogc_bal8.json",
    "small_nogc": CONFIG_DIR / "train_config_cfg90100_band4_perf8_small_nogc.json",
    "small_nogc_bal8": CONFIG_DIR
    / "train_config_cfg90100_band4_perf8_small_nogc_bal8.json",
    "perf16_nogc": CONFIG_DIR / "train_config_cfg90100_band4_perf300_16g.json",
}


class _Dataset:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch


class ReceiptContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text())
        cls.configs = {name: json.loads(path.read_text()) for name, path in CONFIGS.items()}

    def test_only_declared_contract_fields_change(self):
        allowed = {
            "batch_tokens",
            "steps",
            "warmup_type",
            "warmup_ratio",
            "warmup_steps",
            "logging_steps",
            "save_steps",
            "keep_last_n_checkpoints",
            "output_dir",
            "perf_grad_checkpoint",
            "perf_balanced_packing",
        }
        for arm, config in self.configs.items():
            for key in sorted(set(self.reference) | set(config)):
                if key not in allowed:
                    self.assertEqual(config.get(key), self.reference.get(key), f"{arm}:{key}")

    def test_arm_axes_are_exact(self):
        expected = {
            "base": (15648, True, 0),
            "nogc": (15648, False, 0),
            "nogc_bal8": (15648, False, 8),
            "small_nogc": (7824, False, 0),
            "small_nogc_bal8": (7824, False, 8),
            "perf16_nogc": (7824, False, 0),
        }
        for arm, (batch_tokens, checkpointing, balanced) in expected.items():
            config = self.configs[arm]
            self.assertEqual(config["batch_tokens"], batch_tokens, arm)
            self.assertEqual(config["perf_grad_checkpoint"], checkpointing, arm)
            self.assertEqual(config["perf_balanced_packing"], balanced, arm)
            self.assertEqual(config["gradient_accumulation_steps"], 1, arm)
            self.assertEqual(config["steps"], 300, arm)
            self.assertEqual(config["save_steps"], 300, arm)

    def test_global_batch_contracts(self):
        self.assertEqual(8 * self.configs["base"]["batch_tokens"], 125184)
        self.assertEqual(16 * self.configs["perf16_nogc"]["batch_tokens"], 125184)

    def test_runtime_axes_are_forced_to_visible_output(self):
        builder = BUILDER.read_text()
        self.assertIn("def _emit_perf_contract", builder)
        self.assertIn("print(message, flush=True)", builder)
        self.assertIn('f"active={gradient_checkpointing_active}"', builder)
        self.assertIn('f"PERF: balanced packing window={balanced_window}"', builder)

    @unittest.skipIf(PackingIterableDataset is None, "OmniVoice runtime dependencies unavailable")
    def test_balanced_packer_conserves_samples_and_capacity(self):
        samples = [
            {"id": index, "length": length}
            for index, length in enumerate((7, 5, 4, 4, 3, 2, 2, 1, 1))
        ]
        packer = PackingIterableDataset(
            _Dataset(samples), lambda sample: dict(sample), batch_tokens=10, balanced_window=3
        )
        packs = list(packer)
        ids = [sample["id"] for pack in packs for sample in pack]
        self.assertCountEqual(ids, [sample["id"] for sample in samples])
        self.assertEqual(len(ids), len(set(ids)))
        for pack in packs:
            self.assertLessEqual(sum(sample["length"] for sample in pack), 10)


if __name__ == "__main__":
    unittest.main()
