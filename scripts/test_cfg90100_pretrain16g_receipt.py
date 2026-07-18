#!/usr/bin/env python3
"""Static contract test for the two-node CFG90100 Band-4 smoke receipt."""

import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "examples/config/train_config_cfg90100_band4_300k.json"
SMOKE_CONFIG = ROOT / "examples/config/train_config_cfg90100_band4_smoke300_16g.json"
PERF_CONFIG = ROOT / "examples/config/train_config_cfg90100_band4_perf300_16g.json"
LAUNCHER = ROOT / "cfg90100_band4_pretrain_16g.sh"
SUMMARIZER = ROOT / "scripts/summarize_cfg90100_perf16.py"


class ReceiptContractTest(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(REFERENCE_CONFIG.read_text())
        self.smoke = json.loads(SMOKE_CONFIG.read_text())
        self.perf = json.loads(PERF_CONFIG.read_text())
        self.launcher = LAUNCHER.read_text()

    def test_model_contract_matches_current_8g_pretrain(self):
        allowed_changes = {
            "batch_tokens",
            "gradient_accumulation_steps",
            "steps",
            "warmup_type",
            "warmup_ratio",
            "warmup_steps",
            "logging_steps",
            "save_steps",
            "keep_last_n_checkpoints",
            "output_dir",
            "perf_balanced_packing",
        }
        for key in sorted(set(self.reference) | set(self.smoke)):
            if key not in allowed_changes:
                self.assertEqual(self.smoke.get(key), self.reference.get(key), key)

    def test_global_batch_matches_current_8g_pretrain(self):
        reference_global_batch = (
            8
            * self.reference["batch_tokens"]
            * self.reference["gradient_accumulation_steps"]
        )
        smoke_global_batch = (
            16
            * self.smoke["batch_tokens"]
            * self.smoke["gradient_accumulation_steps"]
        )
        self.assertEqual(reference_global_batch, 125184)
        self.assertEqual(smoke_global_batch, reference_global_batch)

    def test_perf_arm_is_only_gradient_checkpointing_off(self):
        allowed_changes = {
            "output_dir",
            "perf_grad_checkpoint",
        }
        for key in sorted(set(self.smoke) | set(self.perf)):
            if key not in allowed_changes:
                self.assertEqual(self.perf.get(key), self.smoke.get(key), key)
        self.assertTrue(self.smoke["perf_grad_checkpoint"])
        self.assertFalse(self.perf["perf_grad_checkpoint"])
        self.assertEqual(self.perf["perf_balanced_packing"], 0)

    def test_smoke_and_distributed_contract(self):
        self.assertEqual(self.smoke["steps"], 300)
        self.assertEqual(self.smoke["save_steps"], 300)
        self.assertEqual(self.smoke["batch_tokens"], 7824)
        self.assertEqual(self.smoke["gradient_accumulation_steps"], 1)
        self.assertIn("expected_num_machines=2", self.launcher)
        self.assertIn("expected_gpus_per_machine=8", self.launcher)
        self.assertIn("expected_world_size=16", self.launcher)
        self.assertIn("global_rank=node_rank*8+local_rank", self.launcher)
        self.assertIn('--num_machines "$num_machines"', self.launcher)
        self.assertIn('--machine_rank "$node_rank"', self.launcher)
        self.assertIn('--num_processes "$world_size"', self.launcher)
        self.assertIn("RUN_ID is required", self.launcher)
        self.assertIn("VC_WORKER_NUM mismatch", self.launcher)
        self.assertIn("VC_WORKER_HOSTS count mismatch", self.launcher)
        self.assertIn("--rdzv_backend static", self.launcher)
        self.assertIn("--max_restarts 0", self.launcher)
        self.assertIn("smoke300|perf300", self.launcher)
        self.assertIn("refusing socket fallback", self.launcher)
        self.assertIn("Using network IB", self.launcher)
        self.assertIn("NET/IB/[0-7]/GDRDMA", self.launcher)
        self.assertIn(str(SUMMARIZER.relative_to(ROOT)), self.launcher)
        self.assertNotIn("sleep infinity", self.launcher)


if __name__ == "__main__":
    unittest.main()
