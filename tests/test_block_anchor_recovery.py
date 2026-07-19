import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "dspark_anchor_scan_recover_eval.sh"
SOURCE_CHANGED_PATHS = sorted(
    (
        "dspark_anchor_scan_recover_eval.sh",
        "scripts/check_block_anchor_checkpoint_attach.py",
        "scripts/eval_block_anchor_nll.py",
        "scripts/finalize_block_anchor_receipt.py",
        "tests/test_block_anchor_campaign.py",
        "tests/test_block_anchor_recovery.py",
    )
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _checkpoint(path: Path, random_states: int) -> Path:
    path.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "train_config.json",
    ):
        (path / name).write_bytes(f"{path.name}:{name}".encode())
    for index in range(random_states):
        (path / f"random_states_{index}.pkl").write_bytes(
            f"{path.name}:state:{index}".encode()
        )
    return path


def _inventory(root: Path) -> dict:
    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _source_repository(root: Path) -> tuple[str, str, str, str]:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test Author"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
    )
    for relative in (
        "examples/config/train_config_cfg90100_anchor_causal_s300.json",
        "examples/config/train_config_cfg90100_anchor_stateless_s300.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": relative}, sort_keys=True) + "\n")
    (root / "BASELINE").write_text("parent\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "parent"], cwd=root, check=True)
    parent_commit = _git(root, "rev-parse", "HEAD")

    for relative in SOURCE_CHANGED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"recovery:{relative}\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "recovery"], cwd=root, check=True)
    recovery_commit = _git(root, "rev-parse", "HEAD")
    return (
        parent_commit,
        recovery_commit,
        _git(root, "rev-parse", "HEAD^{tree}"),
        _git(root, "show", "-s", "--format=%an <%ae>", "HEAD"),
    )


def _make_recovery_case(tmp_path: Path) -> SimpleNamespace:
    source_root = tmp_path / "source"
    old_commit, new_commit, source_tree, source_author = _source_repository(source_root)

    parent = tmp_path / "parent"
    parent_reports = parent / "reports"
    parent_logs = parent / "logs"
    parent_reports.mkdir(parents=True)
    parent_logs.mkdir()
    parent_run_id = "shuai-parent-v1"
    base = _checkpoint(tmp_path / "base", 8)
    causal_output = parent / f"causal_{parent_run_id}"
    stateless_output = parent / f"stateless_{parent_run_id}"
    replay_output = parent / f"causal_replay_{parent_run_id}"
    causal = _checkpoint(causal_output / "checkpoint-300", 2)
    stateless = _checkpoint(stateless_output / "checkpoint-300", 2)
    replay = _checkpoint(replay_output / "checkpoint-300", 2)

    training = parent_reports / f"{parent_run_id}.training.json"
    _write_json(
        training,
        {
            "verdict": "PASS",
            "frozen_backbone_hash_exact": True,
            "causal_replay_bitwise_exact": True,
        },
    )
    parent_manifest = parent / f"{parent_run_id}.manifest.txt"
    parent_manifest.write_text(
        "\n".join(
            (
                f"run_id={parent_run_id}",
                "scope=seed0_300step_only",
                "seed_index=0",
                "train_seed=42",
                "eval_seed=20260719",
                "english_only_mechanism_probe=1",
                f"source_root={source_root}",
                f"source_commit={old_commit}",
                f"base_checkpoint={base}",
                "causal_config=examples/config/train_config_cfg90100_anchor_causal_s300.json",
                "stateless_config=examples/config/train_config_cfg90100_anchor_stateless_s300.json",
                f"causal_output={causal_output}",
                f"stateless_output={stateless_output}",
                f"causal_replay_output={replay_output}",
                "automatic_followup_submitted=0",
                "failed_phase=nll",
                "infrastructure_rc=1",
            )
        )
        + "\n"
    )
    failure = tmp_path / "parent_failure.log"
    failure.write_text(
        "_benchmark_head_on_off\n_forward_device\n"
        "ValueError: Expected query, key, and value to have the same dtype; "
        "query.dtype: torch.float32, key.dtype: torch.float32, "
        "value.dtype: torch.bfloat16\n"
    )
    parent_evidence = {
        "causal_log": parent_logs / f"causal_{parent_run_id}.log",
        "stateless_log": parent_logs / f"stateless_{parent_run_id}.log",
        "causal_replay_log": parent_logs / f"causal_replay_{parent_run_id}.log",
        "causal_memory": parent_logs / f"causal_{parent_run_id}.memory.csv",
        "stateless_memory": parent_logs / f"stateless_{parent_run_id}.memory.csv",
        "causal_replay_memory": parent_logs / f"causal_replay_{parent_run_id}.memory.csv",
        "failure_log": failure,
    }
    for name, path in parent_evidence.items():
        if name != "failure_log":
            path.write_text(f"{name}\n")

    recovery_root = tmp_path / "recovery"
    recovery_reports = recovery_root / "reports"
    recovery_reports.mkdir(parents=True)
    recovery_run_id = "shuai-recovery-v1"
    manifest = recovery_root / f"{recovery_run_id}.manifest.txt"
    manifest.write_text(
        "\n".join(
            (
                f"run_id={recovery_run_id}",
                "scope=seed0_300step_eval_only_recovery",
                "recovery_scope=eval_only",
                "training_reused=1",
                "training_reexecuted=0",
                "seed_index=0",
                "train_seed=42",
                "eval_seed=20260719",
                "english_only_mechanism_probe=1",
                f"source_root={source_root}",
                f"source_commit={new_commit}",
                f"source_author={source_author}",
                f"parent_root={parent}",
                f"parent_run_id={parent_run_id}",
                f"parent_source_commit={old_commit}",
                f"parent_manifest={parent_manifest}",
                f"parent_manifest_sha256={_sha(parent_manifest)}",
                f"parent_training_report={training}",
                f"parent_training_report_sha256={_sha(training)}",
                "parent_failed_phase=nll",
                "parent_infrastructure_rc=1",
                f"parent_failure_log={failure}",
                f"parent_failure_log_sha256={_sha(failure)}",
                "automatic_followup_submitted=0",
            )
        )
        + "\n"
    )
    environment = recovery_reports / f"{recovery_run_id}.environment.json"
    _write_json(
        environment,
        {
            "accelerate_version": "test",
            "cuda_version": "test",
            "cudnn_version": 1,
            "gpu_inventory": ["H100"],
            "liger_kernel_file_count": 1,
            "liger_kernel_source_root": "/test",
            "liger_kernel_tree_sha256": "c" * 64,
            "nvidia_driver_version": ["test"],
            "python_executable": sys.executable,
            "python_version": "test",
            "source_commit": new_commit,
            "parent_source_commit": old_commit,
            "recovery_scope": "eval_only",
            "torch_version": "test",
            "transformers_version": "test",
        },
    )
    regenerated = recovery_reports / f"{recovery_run_id}.training.json"
    regenerated.write_bytes(training.read_bytes())
    causal_config = source_root / "examples/config/train_config_cfg90100_anchor_causal_s300.json"
    stateless_config = source_root / "examples/config/train_config_cfg90100_anchor_stateless_s300.json"
    nll = recovery_reports / f"{recovery_run_id}.nll.json"
    nll_payload = {
        "verdict": "SCIENTIFIC_KILL",
        "seed_index": 0,
        "train_seed": 42,
        "eval_seed": 20260719,
        "english_only_mechanism_probe": True,
        "num_packs": 32,
        "snapshot_sha256": "d" * 64,
        "provenance": {
            "source_root": str(source_root.resolve()),
            "source_commit": new_commit,
            "configs": {
                "causal": {
                    "path": str(causal_config.resolve()),
                    "sha256": _sha(causal_config),
                },
                "stateless": {
                    "path": str(stateless_config.resolve()),
                    "sha256": _sha(stateless_config),
                },
            },
            "checkpoints": {
                name: {
                    "path": str(path.resolve()),
                    "model_sha256": _sha(path / "model.safetensors"),
                }
                for name, path in (
                    ("base", base),
                    ("causal", causal),
                    ("stateless", stateless),
                )
            },
        },
    }
    _write_json(nll, nll_payload)
    verdict = recovery_reports / f"{recovery_run_id}.verdict.json"
    verdict_payload = {
        "verdict": "SCIENTIFIC_KILL",
        "training_verdict": "PASS",
        "nll_verdict": "SCIENTIFIC_KILL",
        "generation_status": "NEEDS_GENERATION",
        "automatic_followup_submitted": False,
        "recovery_scope": "eval_only",
        "training_reused": True,
        "training_reexecuted": False,
        "seed_index": 0,
        "train_seed": 42,
        "eval_seed": 20260719,
        "english_only_mechanism_probe": True,
    }
    _write_json(verdict, verdict_payload)

    checkpoint_inventories = {
        "causal": _inventory(causal),
        "stateless": _inventory(stateless),
        "causal_replay": _inventory(replay),
    }
    parent_lock = {
        "schema": "block_anchor_eval_recovery_parent_lock_v1",
        "parent_root": str(parent.resolve()),
        "parent_run_id": parent_run_id,
        "parent_source_root": str(source_root),
        "parent_source_commit": old_commit,
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": _sha(parent_manifest),
        "parent_training_report": str(training.resolve()),
        "parent_training_report_sha256": _sha(training),
        "parent_failure_log": str(failure.resolve()),
        "parent_failure_log_sha256": _sha(failure),
        "parent_failed_phase": "nll",
        "parent_infrastructure_rc": 1,
        "parent_automatic_followup_submitted": False,
        "training_reused": True,
        "training_reexecuted": False,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha(path)}
            for name, path in sorted(parent_evidence.items())
        },
        "checkpoint_inventories": checkpoint_inventories,
    }
    parent_pre = recovery_reports / f"{recovery_run_id}.parent_preflight.json"
    parent_post = recovery_reports / f"{recovery_run_id}.parent_postflight.json"
    _write_json(parent_pre, parent_lock)
    parent_post.write_bytes(parent_pre.read_bytes())

    source_lock = {
        "schema": "block_anchor_eval_recovery_source_lock_v1",
        "source_root": str(source_root.resolve()),
        "source_commit": new_commit,
        "source_tree": source_tree,
        "source_author": source_author,
        "parent_source_commit": old_commit,
        "changed_paths": SOURCE_CHANGED_PATHS,
        "working_tree_clean": True,
    }
    source_pre = recovery_reports / f"{recovery_run_id}.source_preflight.json"
    source_post = recovery_reports / f"{recovery_run_id}.source_postflight.json"
    _write_json(source_pre, source_lock)
    source_post.write_bytes(source_pre.read_bytes())

    output = recovery_reports / f"{recovery_run_id}.receipt.json"
    artifacts = {
        "parent_manifest": parent_manifest,
        "parent_training_report": training,
        "parent_failure_log": failure,
        "parent_preflight_report": parent_pre,
        "parent_postflight_report": parent_post,
        "source_preflight_report": source_pre,
        "source_postflight_report": source_post,
        "regenerated_training_report": regenerated,
        "nll_report": nll,
        "verdict_report": verdict,
        "environment_report": environment,
    }
    for lock_name, path in parent_evidence.items():
        artifacts[f"parent_{lock_name}"] = path
    command = [
        sys.executable,
        str(ROOT / "scripts/finalize_block_anchor_receipt.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--scientific-verdict",
        "SCIENTIFIC_KILL",
        "--verdict-report",
        str(verdict),
        "--training-report",
        str(regenerated),
        "--nll-report",
        str(nll),
        "--environment-report",
        str(environment),
        "--receipt-mode",
        "recovery_eval_only",
        "--parent-manifest",
        str(parent_manifest),
        "--parent-failure-log",
        str(failure),
        "--seed-index",
        "0",
        "--train-seed",
        "42",
        "--eval-seed",
        "20260719",
        "--english-only-mechanism-probe",
    ]
    for name, path in (
        ("base", base),
        ("causal", causal),
        ("stateless", stateless),
        ("causal_replay", replay),
    ):
        command.extend(("--checkpoint", f"{name}={path}"))
    for name, path in artifacts.items():
        command.extend(("--artifact", f"{name}={path}"))
    return SimpleNamespace(
        command=command,
        manifest=manifest,
        original_manifest=manifest.read_bytes(),
        output=output,
        parent_pre=parent_pre,
        parent_post=parent_post,
        parent_lock=parent_lock,
        source_pre=source_pre,
        source_post=source_post,
        source_lock=source_lock,
        nll=nll,
        verdict=verdict,
        checkpoint_inventories=checkpoint_inventories,
    )


def _run(case: SimpleNamespace) -> subprocess.CompletedProcess:
    return subprocess.run(case.command, capture_output=True, text=True)


def _assert_rejected(case: SimpleNamespace, needle: str) -> None:
    completed = _run(case)
    assert completed.returncode != 0
    assert needle in completed.stderr
    assert not case.output.exists()
    assert not case.output.with_name(case.output.name + ".tmp").exists()
    assert case.manifest.read_bytes() == case.original_manifest


def test_recovery_launcher_is_eval_only_and_fail_closed():
    source = LAUNCHER.read_text()
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    assert 'mkdir "$output_root"' in source
    assert 'mkdir -p "$output_root"' not in source
    assert "recovery RUN_ID must differ from PARENT_RUN_ID" in source
    assert "recovery root must differ from parent root" in source
    for contract in (
        "EXPECTED_PARENT_MANIFEST_SHA256",
        "EXPECTED_PARENT_TRAINING_REPORT_SHA256",
        "EXPECTED_PARENT_FAILURE_LOG_SHA256",
        "EXPECTED_PARENT_CAUSAL_INVENTORY_SHA256",
        "EXPECTED_PARENT_STATELESS_INVENTORY_SHA256",
        "EXPECTED_PARENT_REPLAY_INVENTORY_SHA256",
    ):
        assert contract in source
    assert 'exactly("failed_phase", "nll")' in source
    assert 'exactly("infrastructure_rc", "1")' in source
    assert 'exactly("automatic_followup_submitted", "0")' in source
    assert 'if forbidden in values:' in source
    assert "checkpoint inventory drift" in source
    assert 'cmp -s "$preflight_snapshot" "$parent_preflight_report"' in source
    assert 'cmp -s "$parent_preflight_report" "$parent_postflight_report"' in source
    assert 'cmp -s "$source_preflight_report" "$source_postflight_report"' in source
    assert "scripts/report_block_anchor_training.py" in source
    assert 'cmp -s "$parent_training_report" "$regenerated_training_report"' in source
    assert "training_report_reused_byte_exact=1" in source
    assert "training_reexecuted=0" in source
    assert "recovery source diff escaped allowlist" in source
    assert "scripts/check_block_anchor_contract.py" in source
    assert "contract_post_report" in source
    assert "data manifest content drifted during recovery" in source
    assert "scripts/check_block_anchor_checkpoint_attach.py" in source
    assert "full_model_synthetic_packed_forward=PASS" in source
    assert "scripts/eval_block_anchor_nll.py" in source
    assert "scripts/finalize_block_anchor_receipt.py" in source
    assert "--receipt-mode recovery_eval_only" in source
    assert 'artifact "parent_failure_log=$parent_failure_log"' in source

    for forbidden in (
        "accelerate launch",
        "omnivoice.cli.train",
        "run_arm ",
        "submit-longrun",
        "oms.sh submit",
        "PROMOTE_TO_10K",
        "kubectl delete",
    ):
        assert forbidden not in source
    assert "BLOCK_ANCHOR_RECOVERY_NO_AUTOMATIC_FOLLOWUP" in source


def test_recovery_launcher_rejects_positional_arguments_before_side_effects():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "unexpected"], capture_output=True, text=True
    )
    assert completed.returncode == 2
    assert "usage: dspark_anchor_scan_recover_eval.sh" in completed.stderr


def test_recovery_receipt_derives_and_binds_all_lineage(tmp_path):
    case = _make_recovery_case(tmp_path)
    completed = _run(case)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(case.output.read_text())
    assert payload["schema"] == "block_anchor_seed0_300proof_recovery_receipt_v1"
    assert payload["recovery"]["training_reused"] is True
    assert payload["recovery"]["training_reexecuted"] is False
    assert (
        payload["recovery"]["failure_signature"]["id"]
        == "flex_attention_qkv_dtype_mismatch_v1"
    )
    for name, inventory in case.checkpoint_inventories.items():
        assert payload["checkpoint_inventories"][name] == inventory
    assert payload["recovery"]["source"]["tree"] == case.source_lock["source_tree"]
    assert payload["recovery"]["source"]["changed_paths"] == SOURCE_CHANGED_PATHS
    assert payload["recovery"]["source"]["preflight_lock_sha256"] == _sha(
        case.source_pre
    )
    keys = [line.split("=", 1)[0] for line in case.manifest.read_text().splitlines()]
    assert len(keys) == len(set(keys))
    for key in (
        "artifact_receipt",
        "artifact_receipt_sha256",
        "scientific_verdict",
        "generation_status",
    ):
        assert keys.count(key) == 1


def test_recovery_rejects_semantically_tampered_parent_lock(tmp_path):
    case = _make_recovery_case(tmp_path)
    tampered = json.loads(case.parent_pre.read_text())
    tampered["checkpoint_inventories"]["causal"]["inventory_sha256"] = "f" * 64
    _write_json(case.parent_pre, tampered)
    case.parent_post.write_bytes(case.parent_pre.read_bytes())
    _assert_rejected(case, "parent lock checkpoint inventory mismatch for causal")


def test_recovery_rejects_source_pre_post_drift(tmp_path):
    case = _make_recovery_case(tmp_path)
    tampered = json.loads(case.source_post.read_text())
    tampered["source_tree"] = "f" * 40
    _write_json(case.source_post, tampered)
    _assert_rejected(case, "source evidence changed during eval-only recovery")


def test_recovery_rejects_semantically_dirty_source_lock(tmp_path):
    case = _make_recovery_case(tmp_path)
    tampered = json.loads(case.source_pre.read_text())
    tampered["working_tree_clean"] = False
    _write_json(case.source_pre, tampered)
    case.source_post.write_bytes(case.source_pre.read_bytes())
    _assert_rejected(case, "source lock working_tree_clean is not strict true")


def test_recovery_rejects_numeric_source_lock_boolean(tmp_path):
    case = _make_recovery_case(tmp_path)
    tampered = json.loads(case.source_pre.read_text())
    tampered["working_tree_clean"] = 1
    _write_json(case.source_pre, tampered)
    case.source_post.write_bytes(case.source_pre.read_bytes())
    _assert_rejected(case, "source lock working_tree_clean is not strict true")


def test_recovery_rejects_numeric_parent_lock_boolean(tmp_path):
    case = _make_recovery_case(tmp_path)
    tampered = json.loads(case.parent_pre.read_text())
    tampered["training_reused"] = 1
    _write_json(case.parent_pre, tampered)
    case.parent_post.write_bytes(case.parent_pre.read_bytes())
    _assert_rejected(case, "parent lock boolean is not strict for training_reused")


@pytest.mark.parametrize(
    ("mutation", "needle"),
    (
        ("source_commit", "NLL source commit differs from recovery environment"),
        ("config_sha", "NLL config provenance mismatch for causal"),
        ("checkpoint_sha", "NLL checkpoint provenance mismatch for causal"),
        ("pack_count", "fixed 32-pack snapshot receipt"),
        ("eval_seed", "NLL seed/probe contract mismatch"),
    ),
)
def test_recovery_rejects_nll_provenance_tampering(tmp_path, mutation, needle):
    case = _make_recovery_case(tmp_path)
    payload = json.loads(case.nll.read_text())
    if mutation == "source_commit":
        payload["provenance"]["source_commit"] = "f" * 40
    elif mutation == "config_sha":
        payload["provenance"]["configs"]["causal"]["sha256"] = "f" * 64
    elif mutation == "checkpoint_sha":
        payload["provenance"]["checkpoints"]["causal"]["model_sha256"] = "f" * 64
    elif mutation == "eval_seed":
        payload["eval_seed"] = 999
    else:
        payload["num_packs"] = 31
    _write_json(case.nll, payload)
    _assert_rejected(case, needle)


@pytest.mark.parametrize("field", ("training_reused", "training_reexecuted"))
def test_recovery_rejects_non_boolean_verdict_contract(tmp_path, field):
    case = _make_recovery_case(tmp_path)
    payload = json.loads(case.verdict.read_text())
    payload[field] = "true" if field == "training_reused" else "false"
    _write_json(case.verdict, payload)
    _assert_rejected(case, "strict reused eval-only recovery")
