import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CANARY_ARMS = ("shared_g0", "shared_g0p25", "shared_g0p5", "shared_g1", "shared_g2")


def _write_canary_variant(
    temporary: Path,
    name: str,
    *,
    ids_suffix: str = "fixed",
    rtf: float = 0.5,
    wer: float = 8.0,
    sim: float = 0.60,
    duration_mean: float = 1.05,
    duration_p95: float = 1.20,
) -> tuple[Path, Path]:
    result = temporary / name
    for language in ("zh", "en"):
        inputs = result / "inputs" / language
        inputs.mkdir(parents=True)
        (inputs / "manifest.json").write_text(
            json.dumps(
                {"ordered_ids_sha256": f"{language}-{ids_suffix}"}
            )
        )
    rows = []
    for language in ("zh", "en"):
        for index, arm in enumerate(CANARY_ARMS):
            rows.append(
                {
                    "lang": language,
                    "arm": arm,
                    "guidance_scale": (0, 0.25, 0.5, 1, 2)[index],
                    "runaway": 0,
                    "max_blocks_count": 0,
                    "token_decode_rtf_median": rtf,
                    "wer_percent": wer,
                    "sim": sim,
                    "duration_ratio_mean": duration_mean,
                    "duration_ratio_p95": duration_p95,
                }
            )
    summary = temporary / f"{name}.json"
    summary.write_text(json.dumps({"rows": rows}))
    return result, summary


def test_campaign_configs_and_fail_closed_checkpoint_contract():
    control_path = (
        ROOT
        / "examples/config/train_config_cfg90100_markov_control_s300.json"
    )
    head_path = (
        ROOT
        / "examples/config/train_config_cfg90100_markov_head_r32_s300.json"
    )
    control = json.loads(control_path.read_text())
    head = json.loads(head_path.read_text())
    differences = {
        key
        for key in set(control) | set(head)
        if control.get(key) != head.get(key)
    }
    assert differences == {"block_markov_rank", "output_dir"}
    assert control["block_markov_rank"] == 0
    assert head["block_markov_rank"] == 32
    launcher = (ROOT / "dspark_markov_300ab.sh").read_text()
    subprocess.run(
        ["bash", "-n", str(ROOT / "dspark_markov_300ab.sh")],
        check=True,
    )
    assert '--control-replay-checkpoint "$control_replay_checkpoint"' in launcher
    for receipt_key in (
        "control_model",
        "head_model",
        "control_replay_model",
        "training_report",
        "nll_report",
        "generation_report",
        "verdict_report",
        "base_provenance",
        "control_provenance",
        "head_provenance",
    ):
        assert f'--artifact "{receipt_key}=' in launcher

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
            json.dumps({"block_markov_rank": 0})
        )
        for rank in range(8):
            (checkpoint / f"random_states_{rank}.pkl").write_bytes(b"state")
        base_config_path = (
            ROOT / "examples/config/train_config_cfg90100_band4_300k.json"
        )
        data_config_path = (
            ROOT / "examples/config/data_config_emilia_full_blockparity.json"
        )
        saved_recipe = json.loads(base_config_path.read_text())
        saved_recipe["output_dir"] = str(output)
        (checkpoint / "train_config.json").write_text(json.dumps(saved_recipe))

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        manifest = temporary / "main.manifest.txt"
        manifest.write_text(
            f"output={output}\nsource_commit=deadbeef\n"
            f"config_sha256={sha(base_config_path)}\n"
            f"data_config_sha256={sha(data_config_path)}\n"
            "train_rc=0\ntee_rc=0\nrc=0\n"
        )

        fake_control = dict(control)
        fake_head = dict(head)
        fake_control["init_from_checkpoint"] = str(checkpoint.resolve())
        fake_head["init_from_checkpoint"] = str(checkpoint.resolve())
        fake_control_path = temporary / "control.json"
        fake_head_path = temporary / "head.json"
        fake_control_path.write_text(json.dumps(fake_control))
        fake_head_path.write_text(json.dumps(fake_head))
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_block_markov_ab_contract.py"),
                "--control-config",
                str(fake_control_path),
                "--head-config",
                str(fake_head_path),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--base-train-config",
                str(base_config_path),
                "--data-config",
                str(data_config_path),
                "--expected-main-source-commit",
                "deadbeef",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert "BLOCK_MARKOV_AB_CONTRACT_OK" in completed.stdout


def test_eval_helpers_lock_offsets_and_paired_bootstrap():
    import runpy

    namespace = runpy.run_path(str(ROOT / "scripts/eval_block_markov_ab.py"))
    offsets = namespace["_block_offsets"](
        {
            "document_ids": torch.tensor([[0, 0, 0, 0, 1, 1]]),
            "copy_tags": torch.tensor([[0, 2, 2, 2, 2, 2]]),
            "block_ids": torch.tensor([[-1, 0, 0, 1, 0, 0]]),
        }
    )
    assert torch.equal(offsets, torch.tensor([[-1, 0, 1, 0, 0, 1]]))
    bootstrap = namespace["_bootstrap_delta"](
        [0.9, 1.0, 1.1, 1.2] * 2,
        [1.0, 1.1, 1.2, 1.3] * 2,
        samples=1000,
    )
    assert bootstrap["mean"] < 0
    assert bootstrap["ci95_high"] < 0


def test_training_report_reads_real_log_and_safetensors_shapes():
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        checkpoints = {}
        for arm, rank in (
            ("control", 0),
            ("head", 32),
            ("control_replay", 0),
        ):
            checkpoint = temporary / arm / "checkpoint-300"
            checkpoint.mkdir(parents=True)
            for name in (
                "optimizer.bin",
                "scheduler.bin",
                "train_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (checkpoint / name).write_bytes(b"receipt")
            (checkpoint / "config.json").write_text(
                json.dumps({"block_markov_rank": rank})
            )
            for worker in range(2):
                (checkpoint / f"random_states_{worker}.pkl").write_bytes(
                    b"state"
                )
            checkpoints[arm] = checkpoint
        save_file(
            {"dummy": torch.ones(1)},
            str(checkpoints["control"] / "model.safetensors"),
        )
        save_file(
            {"dummy": torch.ones(1)},
            str(checkpoints["control_replay"] / "model.safetensors"),
        )
        save_file(
            {
                "block_markov_head.output.weight": torch.ones(8, 4),
                "block_markov_head.prev_embeddings.weight": torch.ones(8, 4),
            },
            str(checkpoints["head"] / "model.safetensors"),
        )

        logs = {}
        memories = {}
        for arm, rate, memory in (
            ("control", 2.0, 1000),
            ("head", 1.9, 1100),
            ("control_replay", 2.0, 1000),
        ):
            logs[arm] = temporary / f"{arm}.log"
            logs[arm].write_text(
                "\n".join(
                    f"Step {step} | train/loss: 3.9 | "
                    f"train/steps_per_sec: {rate}"
                    for step in range(100, 301, 10)
                )
            )
            memories[arm] = temporary / f"{arm}.memory.csv"
            memories[arm].write_text(
                "".join(
                    f"2026/07/19 00:00:{second:02d}, {gpu}, {memory}\n"
                    for second in range(12)
                    for gpu in (4, 7)
                )
            )
        report = temporary / "training.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/report_block_markov_training_ab.py"),
                "--control-log",
                str(logs["control"]),
                "--head-log",
                str(logs["head"]),
                "--control-replay-log",
                str(logs["control_replay"]),
                "--control-memory",
                str(memories["control"]),
                "--head-memory",
                str(memories["head"]),
                "--control-replay-memory",
                str(memories["control_replay"]),
                "--control-checkpoint",
                str(checkpoints["control"]),
                "--head-checkpoint",
                str(checkpoints["head"]),
                "--control-replay-checkpoint",
                str(checkpoints["control_replay"]),
                "--output",
                str(report),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert "BLOCK_MARKOV_TRAINING_VERDICT" in completed.stdout
        payload = json.loads(report.read_text())
        assert payload["verdict"] == "PASS"
        assert payload["control"]["memory"]["samples_per_gpu"] == {
            "4": 12,
            "7": 12,
        }


def test_generation_canary_compares_identical_bilingual_matrix():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        results = {}
        summaries = {}
        for variant, rtf in (("base", 0.50), ("control", 0.51), ("head", 0.90)):
            results[variant], summaries[variant] = _write_canary_variant(
                temporary, variant, rtf=rtf
            )

        output = temporary / "generation.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/eval_block_markov_generation_canary.py"),
                "--base-summary",
                str(summaries["base"]),
                "--control-summary",
                str(summaries["control"]),
                "--head-summary",
                str(summaries["head"]),
                "--base-result",
                str(results["base"]),
                "--control-result",
                str(results["control"]),
                "--head-result",
                str(results["head"]),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "BLOCK_MARKOV_GENERATION_CANARY" in completed.stdout
        payload = json.loads(output.read_text())
        assert payload["verdict"] == "PASS"
        assert payload["timing_comparisons_descriptive_only"]


def test_generation_canary_fails_common_absolute_disaster():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        variants = {
            name: _write_canary_variant(temporary, name, wer=60.0)
            for name in ("base", "control", "head")
        }
        output = temporary / "generation.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/eval_block_markov_generation_canary.py"),
                "--base-summary",
                str(variants["base"][1]),
                "--control-summary",
                str(variants["control"][1]),
                "--head-summary",
                str(variants["head"][1]),
                "--base-result",
                str(variants["base"][0]),
                "--control-result",
                str(variants["control"][0]),
                "--head-result",
                str(variants["head"][0]),
                "--output",
                str(output),
            ],
            check=True,
        )
        payload = json.loads(output.read_text())
        assert payload["verdict"] == "FAIL"
        assert any("absolute WER disaster" in item for item in payload["failures"])


def test_generation_canary_gates_head_against_frozen_base():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        variants = {
            "base": _write_canary_variant(temporary, "base", wer=5.0),
            "control": _write_canary_variant(temporary, "control", wer=20.0),
            "head": _write_canary_variant(temporary, "head", wer=20.0),
        }
        output = temporary / "generation.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/eval_block_markov_generation_canary.py"),
                "--base-summary",
                str(variants["base"][1]),
                "--control-summary",
                str(variants["control"][1]),
                "--head-summary",
                str(variants["head"][1]),
                "--base-result",
                str(variants["base"][0]),
                "--control-result",
                str(variants["control"][0]),
                "--head-result",
                str(variants["head"][0]),
                "--output",
                str(output),
            ],
            check=True,
        )
        payload = json.loads(output.read_text())
        assert payload["verdict"] == "FAIL"
        assert any("WER disaster vs base" in item for item in payload["failures"])


def test_generation_canary_rejects_common_missing_matrix_row():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        variants = {
            name: _write_canary_variant(temporary, name)
            for name in ("base", "control", "head")
        }
        for _, summary in variants.values():
            payload = json.loads(summary.read_text())
            payload["rows"].pop()
            summary.write_text(json.dumps(payload))
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/eval_block_markov_generation_canary.py"),
                "--base-summary",
                str(variants["base"][1]),
                "--control-summary",
                str(variants["control"][1]),
                "--head-summary",
                str(variants["head"][1]),
                "--base-result",
                str(variants["base"][0]),
                "--control-result",
                str(variants["control"][0]),
                "--head-result",
                str(variants["head"][0]),
                "--output",
                str(temporary / "generation.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "registered bilingual matrix" in completed.stderr


def test_receipt_finalizer_hashes_artifacts_and_preserves_scientific_kill():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        manifest = temporary / "run.manifest.txt"
        manifest.write_text("run_id=test\nsource_commit=deadbeef\n")
        artifact = temporary / "evidence.json"
        artifact.write_text('{"verdict":"FAIL"}\n')
        receipt = temporary / "receipt.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/finalize_block_markov_receipt.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(receipt),
                "--scientific-verdict",
                "KILL",
                "--artifact",
                f"verdict_report={artifact}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "BLOCK_MARKOV_RECEIPT_FINALIZED" in completed.stdout
        payload = json.loads(receipt.read_text())
        expected_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert payload["scientific_verdict"] == "KILL"
        assert payload["artifacts"]["verdict_report"]["sha256"] == expected_sha
        manifest_text = manifest.read_text()
        assert "scientific_verdict=KILL" in manifest_text
        assert "infrastructure_rc=0" in manifest_text
        assert "rc=0" in manifest_text

        repeated = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/finalize_block_markov_receipt.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(receipt),
                "--scientific-verdict",
                "KILL",
                "--artifact",
                f"verdict_report={artifact}",
            ],
            capture_output=True,
            text=True,
        )
        assert repeated.returncode != 0
