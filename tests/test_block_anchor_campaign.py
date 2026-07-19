import hashlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_seed0_configs_and_launcher_contract():
    causal_path = ROOT / "examples/config/train_config_cfg90100_anchor_causal_s300.json"
    stateless_path = ROOT / "examples/config/train_config_cfg90100_anchor_stateless_s300.json"
    causal = json.loads(causal_path.read_text())
    stateless = json.loads(stateless_path.read_text())
    differences = {
        key for key in set(causal) | set(stateless) if causal.get(key) != stateless.get(key)
    }
    assert differences == {"block_anchor_mode", "output_dir"}
    assert causal["block_anchor_mode"] == "causal"
    assert stateless["block_anchor_mode"] == "stateless"
    for config in (causal, stateless):
        assert config["steps"] == 300
        assert config["seed"] == 42
        assert config["block_markov_rank"] == 0
        assert config["block_anchor_scan_dim"] == 64
        assert config["block_anchor_proposal_dim"] == 32
        assert config["block_anchor_stride"] == 8
        assert config["block_anchor_freeze_base"] is True
        assert config["perf_grad_checkpoint"] is False
        assert "block_anchor_topk" not in config

    launcher_path = ROOT / "dspark_anchor_scan_300proof.sh"
    launcher = launcher_path.read_text()
    subprocess.run(["bash", "-n", str(launcher_path)], check=True)
    assert "run_arm causal " in launcher
    assert "run_arm stateless " in launcher
    assert "run_arm causal_replay " in launcher
    assert "--num-packs 32" in launcher
    assert "--benchmark-packs 4 --benchmark-warmup 2 --benchmark-repeats 5" in launcher
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher
    assert "seed_index=0" in launcher
    assert "english_only_mechanism_probe=1" in launcher
    assert "BLOCK_ANCHOR_NO_AUTOMATIC_FOLLOWUP" in launcher
    assert "RUN_ID must use the shuai- prefix" in launcher
    assert "liger_kernel_tree_sha256" in launcher
    assert "data manifest content drifted during proof" in launcher
    assert "submit-longrun" not in launcher
    assert "PROMOTE_TO_10K" not in launcher


def test_contract_accepts_only_mode_and_output_drift():
    if importlib.util.find_spec("torch") is None:
        import pytest

        pytest.skip("local campaign test environment has no torch")
    causal = json.loads(
        (ROOT / "examples/config/train_config_cfg90100_anchor_causal_s300.json").read_text()
    )
    stateless = json.loads(
        (ROOT / "examples/config/train_config_cfg90100_anchor_stateless_s300.json").read_text()
    )
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        output = temporary / "main"
        checkpoint = output / "checkpoint-300000"
        checkpoint.mkdir(parents=True)
        for name in (
            "model.safetensors",
            "optimizer.bin",
            "scheduler.bin",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (checkpoint / name).write_bytes(b"receipt")
        (checkpoint / "config.json").write_text(
            json.dumps({"block_markov_rank": 0, "block_anchor_scan_dim": 0})
        )
        for rank in range(8):
            (checkpoint / f"random_states_{rank}.pkl").write_bytes(b"state")
        base_config = ROOT / "examples/config/train_config_cfg90100_band4_300k.json"
        data_config = ROOT / "examples/config/data_config_emilia_full_blockparity.json"
        recipe = json.loads(base_config.read_text())
        recipe["output_dir"] = str(output)
        (checkpoint / "train_config.json").write_text(json.dumps(recipe))
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = temporary / "main.manifest.txt"
        manifest.write_text(
            f"output={output}\nsource_commit=deadbeef\n"
            f"config_sha256={sha(base_config)}\n"
            f"data_config_sha256={sha(data_config)}\n"
            "train_rc=0\ntee_rc=0\nrc=0\n"
        )
        causal["init_from_checkpoint"] = str(checkpoint.resolve())
        stateless["init_from_checkpoint"] = str(checkpoint.resolve())
        causal_path = temporary / "causal.json"
        stateless_path = temporary / "stateless.json"
        causal_path.write_text(json.dumps(causal))
        stateless_path.write_text(json.dumps(stateless))
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_block_anchor_contract.py"),
                "--causal-config",
                str(causal_path),
                "--stateless-config",
                str(stateless_path),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--base-train-config",
                str(base_config),
                "--data-config",
                str(data_config),
                "--expected-main-source-commit",
                "deadbeef",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert "BLOCK_ANCHOR_CONTRACT_OK" in completed.stdout

        causal["drop_cond_ratio"] = 0.2
        causal_path.write_text(json.dumps(causal))
        rejected = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_block_anchor_contract.py"),
                "--causal-config",
                str(causal_path),
                "--stateless-config",
                str(stateless_path),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--base-train-config",
                str(base_config),
                "--data-config",
                str(data_config),
                "--expected-main-source-commit",
                "deadbeef",
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        assert rejected.returncode != 0
        assert "config drift" in rejected.stderr


def test_paired_bootstrap_is_directional():
    if importlib.util.find_spec("torch") is None:
        import pytest

        pytest.skip("local campaign test environment has no torch")
    namespace = runpy.run_path(str(ROOT / "scripts/eval_block_anchor_nll.py"))
    result = namespace["_bootstrap_delta"](
        [0.9, 1.0, 1.1, 1.2] * 8,
        [1.0, 1.1, 1.2, 1.3] * 8,
        samples=1000,
    )
    assert result["mean"] < 0
    assert result["ci95_high"] < 0


def test_independently_trained_stateless_pair_gates_promotion():
    source = (ROOT / "scripts/eval_block_anchor_nll.py").read_text()
    assert "trained_causal_vs_stateless_soft_ci = _bootstrap_delta(" in source
    assert 'stateless_on["soft_only_suffix"]["pack_weighted_nll"]' in source
    assert "independently-trained causal/stateless paired noninferiority gate failed" in source
    assert '"same_causal_checkpoint_mode_override_role": "mechanism_diagnostic_only"' in source
    assert "TRAINED_STATELESS_NONINFERIORITY_MARGIN_PCT = 0.05" in source
    assert "same-weight causal propagation advantage/CI gate failed" in source


def test_final_verdict_severity_is_fail_closed():
    namespace = runpy.run_path(str(ROOT / "scripts/finalize_block_anchor_receipt.py"))
    resolve = namespace["resolve_final_verdict"]
    assert resolve("PASS", "PROMOTE_TO_3SEED") == "PROMOTE_TO_3SEED"
    assert resolve("ENGINEERING_BLOCK", "INVALID_IMPLEMENTATION") == "INVALID_IMPLEMENTATION"
    assert resolve("INVALID_IMPLEMENTATION", "ENGINEERING_BLOCK") == "INVALID_IMPLEMENTATION"
    assert resolve("PASS", "ENGINEERING_BLOCK") == "ENGINEERING_BLOCK"


def test_replay_and_label_independence_checks_are_not_vacuous():
    training_source = (ROOT / "scripts/report_block_anchor_training.py").read_text()
    nll_source = (ROOT / "scripts/eval_block_anchor_nll.py").read_text()
    assert "causal_loss_trace = [(step, loss)" in training_source
    assert "replay_loss_trace = [(step, loss)" in training_source
    assert "REPLAY_HEAD_MAX_ABS_TOL = 5e-6" in training_source
    assert "REPLAY_LOSS_MAX_ABS_TOL = 1e-5" in training_source
    assert '"causal_replay_bitwise_exact"' in training_source
    assert '"replay_numeric_noise_floor"' in training_source
    assert '"labels",' in nll_source
    assert '"loss_kind",' in nll_source
    assert "valid_target = mutated_labels.ne(-100)" in nll_source
    assert "mutated_labels[valid_target] = replacement[valid_target]" in nll_source


def test_bf16_attach_and_forward_microbenchmark_are_hard_gates():
    attach = (ROOT / "scripts/check_block_anchor_checkpoint_attach.py").read_text()
    nll = (ROOT / "scripts/eval_block_anchor_nll.py").read_text()
    assert "diagnostic = head.to(device=device, dtype=torch.bfloat16)" in attach
    assert 'device = torch.device("cuda:0")' in attach
    assert "causal.dtype != torch.float32" in attach
    assert "partition_delta > 5e-6" in attach
    assert "full_model_synthetic_packed_forward=PASS" in attach
    assert "full_on = model(**packed).logits" in attach
    assert "full_off = model(**packed).logits" in attach
    assert "config.perf_flex_bf16_qkv = True" in attach
    assert "def _benchmark_head_on_off(" in nll
    assert "MIN_HEAD_ON_THROUGHPUT_RATIO = 0.95" in nll
    assert "0.08 * off_peak_mib" in nll
    assert "MAX_HEAD_ON_MEMORY_DELTA_MIB" in nll
    assert '"measurement_order": "alternating_head_off_first/head_on_first"' in nll


def test_eval_preserves_required_bf16_flex_attention_contract():
    if importlib.util.find_spec("torch") is None:
        import pytest

        pytest.skip("local campaign test environment has no torch")
    from scripts.eval_block_anchor_nll import _config

    for enabled in (False, True):
        config = _config(
            ROOT / "examples/config/train_config_cfg90100_anchor_causal_s300.json",
            Path("/tmp/block-anchor-contract-only-checkpoint"),
            enabled=enabled,
        )
        assert config.attn_implementation == "flex_attention"
        assert config.perf_flex_bf16_qkv is True


def test_nll_and_recovery_receipt_bind_evaluation_inputs():
    nll = (ROOT / "scripts/eval_block_anchor_nll.py").read_text()
    receipt = (ROOT / "scripts/finalize_block_anchor_receipt.py").read_text()
    assert '"provenance": {' in nll
    assert '"source_commit": source_commit' in nll
    assert '"model_sha256": _file_sha256' in nll
    assert '"configs": config_provenance' in nll
    assert 'choices=("standard", "recovery_eval_only")' in receipt
    assert '"training_reexecuted": False' in receipt
    assert '"flex_attention_qkv_dtype_mismatch_v1"' in receipt
    assert "NLL checkpoint provenance mismatch" in receipt
    assert "fixed 32-pack snapshot receipt" in receipt


def test_layout_proves_exact_noisy_coverage_and_committed_boundary():
    if importlib.util.find_spec("torch") is None:
        import pytest

        pytest.skip("local campaign test environment has no torch")
    import torch

    from omnivoice.blockdiff_dual import TAG_CLEAN, TAG_NOISY
    from scripts.eval_block_anchor_nll import _validate_layout

    mask_id = 5
    batch = {
        "input_ids": torch.tensor(
            [
                [
                    [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5],
                    [2, 3, 4, 1, 5, 5, 5, 5, 5, 5, 5, 5],
                ]
            ]
        ),
        "copy_tags": torch.tensor(
            [[TAG_CLEAN] * 4 + [TAG_NOISY] * 8], dtype=torch.int32
        ),
        "document_ids": torch.zeros((1, 12), dtype=torch.int32),
        "block_ids": torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
            dtype=torch.int32,
        ),
        "anchor_positions": torch.tensor([[[4, 5, 6, 7], [8, 9, 10, 11]]]),
        "anchor_boundary_ids": torch.tensor([[[5, 5], [4, 1]]]),
    }
    _validate_layout(batch, mask_id=mask_id)

    leaked = {name: value.clone() for name, value in batch.items()}
    leaked["anchor_boundary_ids"][0, 1] = torch.tensor([3, 4])
    try:
        _validate_layout(leaked, mask_id=mask_id)
    except RuntimeError as error:
        assert "immediately preceding committed CLEAN frame" in str(error)
    else:
        raise AssertionError("future/stale boundary leak was accepted")

    holed = {name: value.clone() for name, value in batch.items()}
    holed["anchor_positions"][0, 0] = torch.tensor([4, -1, 6, 7])
    try:
        _validate_layout(holed, mask_id=mask_id)
    except RuntimeError as error:
        assert "exact padded coverage" in str(error)
    else:
        raise AssertionError("holed anchor layout was accepted")


def test_receipt_has_closed_verdict_vocabulary_and_no_followup():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        manifest = temporary / "run.manifest.txt"
        manifest.write_text(
            "run_id=test\n"
            "source_commit=deadbeef\n"
            "seed_index=0\n"
            "train_seed=42\n"
            "eval_seed=20260719\n"
            "english_only_mechanism_probe=1\n"
            "automatic_followup_submitted=0\n"
        )
        artifact = temporary / "verdict.json"
        training = temporary / "training.json"
        training.write_text('{"verdict":"PASS"}\n')
        nll = temporary / "nll.json"
        nll.write_text('{"verdict":"INCONCLUSIVE_1K"}\n')
        artifact.write_text(
            json.dumps(
                {
                    "verdict": "INCONCLUSIVE_1K",
                    "training_verdict": "PASS",
                    "nll_verdict": "INCONCLUSIVE_1K",
                    "generation_status": "NEEDS_GENERATION",
                    "automatic_followup_submitted": False,
                    "seed_index": 0,
                    "train_seed": 42,
                    "eval_seed": 20260719,
                    "english_only_mechanism_probe": True,
                }
            )
            + "\n"
        )
        environment = temporary / "environment.json"
        environment.write_text(
            json.dumps(
                {
                    "accelerate_version": "1",
                    "cuda_version": "12",
                    "cudnn_version": 9000,
                    "gpu_inventory": ["0, H100"],
                    "liger_kernel_file_count": 1,
                    "liger_kernel_source_root": "/tmp/liger_kernel",
                    "liger_kernel_tree_sha256": "a" * 64,
                    "nvidia_driver_version": ["1"],
                    "python_executable": sys.executable,
                    "python_version": "3",
                    "source_commit": "deadbeef",
                    "torch_version": "2",
                    "transformers_version": "4",
                }
            )
        )
        checkpoint = temporary / "checkpoint-300"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"model")
        (checkpoint / "optimizer.bin").write_bytes(b"optimizer")
        receipt = temporary / "receipt.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/finalize_block_anchor_receipt.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(receipt),
            "--scientific-verdict",
            "INCONCLUSIVE_1K",
            "--verdict-report",
            str(artifact),
            "--training-report",
            str(training),
            "--nll-report",
            str(nll),
            "--environment-report",
            str(environment),
            "--checkpoint",
            f"causal={checkpoint}",
            "--seed-index",
            "0",
            "--train-seed",
            "42",
            "--eval-seed",
            "20260719",
            "--english-only-mechanism-probe",
            "--artifact",
            f"verdict_report={artifact}",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "BLOCK_ANCHOR_RECEIPT_FINALIZED" in completed.stdout
        payload = json.loads(receipt.read_text())
        assert payload["schema"] == "block_anchor_seed0_300proof_receipt_v2"
        assert "recovery" not in payload
        assert payload["scientific_verdict"] == "INCONCLUSIVE_1K"
        assert payload["generation_status"] == "NEEDS_GENERATION"
        assert payload["automatic_followup_submitted"] is False
        assert payload["seed_index"] == 0
        assert payload["train_seed"] == 42
        assert payload["eval_seed"] == 20260719
        assert payload["english_only_mechanism_probe"] is True
        assert payload["checkpoint_inventories"]["causal"]["file_count"] == 2
        assert payload["environment"]["source_commit"] == "deadbeef"
        assert "automatic_followup_submitted=0" in manifest.read_text()
        keys = [line.split("=", 1)[0] for line in manifest.read_text().splitlines()]
        assert len(keys) == len(set(keys))


def test_receipt_rejects_verdict_report_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        manifest = temporary / "run.manifest.txt"
        manifest.write_text("run_id=test\n")
        verdict = temporary / "verdict.json"
        verdict.write_text(
            '{"verdict":"SCIENTIFIC_KILL","training_verdict":"PASS",'
            '"nll_verdict":"SCIENTIFIC_KILL",'
            '"generation_status":"NEEDS_GENERATION",'
            '"automatic_followup_submitted":false}\n'
        )
        training = temporary / "training.json"
        training.write_text('{"verdict":"PASS"}\n')
        nll = temporary / "nll.json"
        nll.write_text('{"verdict":"SCIENTIFIC_KILL"}\n')
        environment = temporary / "environment.json"
        environment.write_text("{}\n")
        checkpoint = temporary / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model").write_bytes(b"x")
        rejected = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/finalize_block_anchor_receipt.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(temporary / "receipt.json"),
                "--scientific-verdict",
                "PROMOTE_TO_3SEED",
                "--verdict-report",
                str(verdict),
                "--training-report",
                str(training),
                "--nll-report",
                str(nll),
                "--environment-report",
                str(environment),
                "--checkpoint",
                f"causal={checkpoint}",
                "--seed-index",
                "0",
                "--train-seed",
                "42",
                "--eval-seed",
                "20260719",
                "--english-only-mechanism-probe",
                "--artifact",
                f"verdict_report={verdict}",
            ],
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "scientific verdict mismatch" in rejected.stderr
