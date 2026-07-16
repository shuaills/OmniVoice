import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
WORKLOAD_PATH = ROOT / "training_contract_cfg_guidance_probe.sh"
CANONICAL_PATH = ROOT / "training_contract_probe_first100.sh"


def array(script: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\((.*?)\)$", script, flags=re.MULTILINE)
    assert match is not None
    return shlex.split(match.group(1))


def assignment(script: str, name: str) -> str:
    match = re.search(rf"^{name}=\$\{{{name}:-([^}}]+)}}$", script, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_cfg_guidance_runner_has_only_the_targeted_matrix() -> None:
    script = WORKLOAD_PATH.read_text()

    assert "MODE=${MODE:-calibrate}" in script
    assert array(script, "CALIBRATE_ARM_NAMES") == [
        "shared_g2",
        "drop_ref_g0p5",
        "drop_ref_g1",
        "drop_ref_g1p5",
        "drop_ref_g2",
    ]
    assert array(script, "CALIBRATE_CFG_POLICIES") == [
        "shared",
        "drop_ref",
        "drop_ref",
        "drop_ref",
        "drop_ref",
    ]
    assert array(script, "CALIBRATE_GUIDANCES") == [
        "2.0",
        "0.5",
        "1.0",
        "1.5",
        "2.0",
    ]
    assert "BASELINE_ARM=shared_g2" in script
    assert "PROMPT_CONTRACT=current" in script
    assert "LANG_POLICY=dataset" in script
    assert "--lang None" not in script
    assert "--prompt-contract \"$PROMPT_CONTRACT\"" in script
    assert "--cfg-unconditional-seed-policy \"$cfg_policy\"" in script


def test_shared_sweep_matrix_is_shared_reference_only_and_includes_zero() -> None:
    script = WORKLOAD_PATH.read_text()

    assert array(script, "SHARED_SWEEP_ARM_NAMES") == [
        "shared_g0",
        "shared_g0p25",
        "shared_g0p5",
        "shared_g1",
        "shared_g2",
    ]
    assert array(script, "SHARED_SWEEP_CFG_POLICIES") == ["shared"] * 5
    assert array(script, "SHARED_SWEEP_GUIDANCES") == [
        "0",
        "0.25",
        "0.5",
        "1.0",
        "2.0",
    ]
    assert 'MODE=shared_sweep requires EXPECTED_COUNT in {5,100}' in script
    assert 'MODE=shared_sweep rejects PROMOTED_GUIDANCE' in script
    assert 'MODE=shared_sweep requires STEPS_PER_BLOCK=16' in script
    assert '--eos-cfg-calibration legacy' in script
    assert '--measure-token-decode' in script


def test_cfg_guidance_runner_locks_subset_shards_seed_and_fixed_decode() -> None:
    script = WORKLOAD_PATH.read_text()
    canonical = CANONICAL_PATH.read_text()

    assert "EXPECTED_COUNT=${EXPECTED_COUNT:-100}" in script
    assert 'MODE=calibrate requires EXPECTED_COUNT=100' in script
    assert 'MODE=promote requires EXPECTED_COUNT=300' in script
    assert "GPU_IDS=${GPU_IDS:-0,1,2}" in script
    assert 'MODE=$MODE requires exactly three GPUs' in script
    assert 'MODE=shared_sweep requires at least two GPUs' in script
    assert "SEED_BASE=20260707" in script
    assert "global_subset_row_index" in script
    for name in ("MAIN", "E", "L", "DL", "BASE", "MODELS", "CK", "CONFIG_SRC", "TREND"):
        assert assignment(script, name) == assignment(canonical, name)
    for fixed in (
        "STEPS_PER_BLOCK=${STEPS_PER_BLOCK:-16}",
        "BLOCK_SIZE=${BLOCK_SIZE:-32}",
        "MAX_BLOCKS=${MAX_BLOCKS:-24}",
        "ITEM_ERROR_POLICY=${ITEM_ERROR_POLICY:-fail-at-end}",
        "--dtype bf16",
        "--silence-stop-seconds 0",
        '--shard "$shard/${#GPU_ARRAY[@]}"',
    ):
        assert fixed in script


def run_contract(
    result_root: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "calibrate",
            "EXPECTED_COUNT": "100",
            "PROMOTED_GUIDANCE": "",
            "RUN_ID": "pytest-contract",
            "RESULT_ROOT": str(result_root),
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(WORKLOAD_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"MODE": "unknown"}, "MODE must be calibrate, promote, or shared_sweep"),
        (
            {"MODE": "calibrate", "EXPECTED_COUNT": "300"},
            "MODE=calibrate requires EXPECTED_COUNT=100",
        ),
        (
            {"MODE": "calibrate", "PROMOTED_GUIDANCE": "0.5"},
            "MODE=calibrate rejects PROMOTED_GUIDANCE",
        ),
        (
            {
                "MODE": "promote",
                "EXPECTED_COUNT": "100",
                "PROMOTED_GUIDANCE": "0.5",
            },
            "MODE=promote requires EXPECTED_COUNT=300",
        ),
        (
            {"MODE": "promote", "EXPECTED_COUNT": "300"},
            "MODE=promote requires PROMOTED_GUIDANCE",
        ),
        (
            {"MODE": "shared_sweep", "EXPECTED_COUNT": "6"},
            "MODE=shared_sweep requires EXPECTED_COUNT in {5,100}",
        ),
        (
            {
                "MODE": "shared_sweep",
                "EXPECTED_COUNT": "5",
                "PROMOTED_GUIDANCE": "0.5",
            },
            "MODE=shared_sweep rejects PROMOTED_GUIDANCE",
        ),
        (
            {
                "MODE": "shared_sweep",
                "EXPECTED_COUNT": "5",
                "STEPS_PER_BLOCK": "8",
            },
            "MODE=shared_sweep requires STEPS_PER_BLOCK=16",
        ),
        (
            {
                "MODE": "promote",
                "EXPECTED_COUNT": "300",
                "PROMOTED_GUIDANCE": "1",
            },
            "MODE=promote requires PROMOTED_GUIDANCE",
        ),
    ],
)
def test_cfg_guidance_runner_rejects_illegal_mode_combinations(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    result = run_contract(tmp_path, **overrides)

    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    ("guidance", "arm"),
    [
        ("0.5", "drop_ref_g0p5"),
        ("1.0", "drop_ref_g1"),
        ("1.5", "drop_ref_g1p5"),
        ("2.0", "drop_ref_g2"),
    ],
)
def test_promote_maps_every_allowed_guidance_to_its_candidate(
    tmp_path: Path, guidance: str, arm: str
) -> None:
    script = WORKLOAD_PATH.read_text()
    promote_case = re.search(
        r'case "\$PROMOTED_GUIDANCE" in(?P<body>.*?)\n    esac',
        script,
        flags=re.DOTALL,
    )

    assert promote_case is not None
    assert re.search(
        rf"^      {re.escape(guidance)}\) promoted_arm={re.escape(arm)} ;;$",
        promote_case.group("body"),
        flags=re.MULTILINE,
    )
    missing_repo = tmp_path / "missing-repo"
    result = run_contract(
        tmp_path,
        MODE="promote",
        EXPECTED_COUNT="300",
        PROMOTED_GUIDANCE=guidance,
        C=str(missing_repo),
    )
    assert result.returncode != 0
    assert f"required directory not found: {missing_repo}" in result.stderr


def test_promote_matrix_contains_only_baseline_and_selected_candidate() -> None:
    script = WORKLOAD_PATH.read_text()

    assert 'ARM_NAMES=(shared_g2 "$promoted_arm")' in script
    assert "ARM_CFG_POLICIES=(shared drop_ref)" in script
    assert 'ARM_GUIDANCES=(2.0 "$PROMOTED_GUIDANCE")' in script
    assert "expected_arm_count=2" in script


def test_shared_sweep_accepts_two_shards_but_legacy_modes_keep_three(
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-repo"
    shared = run_contract(
        tmp_path,
        MODE="shared_sweep",
        EXPECTED_COUNT="5",
        GPU_IDS="0,1",
        C=str(missing_repo),
    )
    assert shared.returncode != 0
    assert f"required directory not found: {missing_repo}" in shared.stderr

    too_few = run_contract(
        tmp_path,
        MODE="shared_sweep",
        EXPECTED_COUNT="5",
        GPU_IDS="0",
    )
    assert too_few.returncode != 0
    assert "MODE=shared_sweep requires at least two GPUs" in too_few.stderr

    calibrate = run_contract(tmp_path, GPU_IDS="0,1")
    assert calibrate.returncode != 0
    assert "MODE=calibrate requires exactly three GPUs" in calibrate.stderr


def test_cfg_guidance_runner_uses_same_reporter_and_scorers_with_paired_output() -> None:
    script = WORKLOAD_PATH.read_text()

    assert "REPORTER=$C/scripts/training_contract_probe_report.py" in script
    for command in ("prepare-inputs", "validate-generation", "summarize-arm", "aggregate"):
        assert f'python "$REPORTER" {command}' in script
    assert "omnivoice/eval/wer/seedtts.py" in script
    assert "omnivoice/eval/speaker_similarity/sim.py" in script
    assert '--batch-size 4' in script
    assert '--baseline-arm "$BASELINE_ARM"' in script
    assert '--output-tsv "$RES/SUMMARY.tsv"' in script
    assert '--output-json "$RES/SUMMARY.json"' in script
    assert '--output-md "$RES/SUMMARY.md"' in script


def test_cfg_guidance_runner_is_fresh_fail_closed_and_self_terminating() -> None:
    script = WORKLOAD_PATH.read_text()

    assert 'mkdir "$RES"' in script
    assert 'mkdir "$wav_dir"' in script
    assert "git diff --quiet --no-ext-diff" in script
    assert "git diff --cached --quiet --no-ext-diff" in script
    assert "--item-error-policy \"$ITEM_ERROR_POLICY\"" in script
    assert "trap cleanup EXIT" in script
    assert "wait_group \"generation arm=$arm lang=$lang\"" in script
    assert "wait_group \"scoring arm=$arm lang=$lang\"" in script
    assert "CFG_GUIDANCE_PROBE_DONE" in script
    assert 'TIMING_ARGS+=(--measure-token-decode)' in script
    assert 'timing_scope=core token decode only' in script
    assert 'first_packet_latency=not measured' in script
    for forbidden in (
        "oms job submit",
        "oms pod console",
        "sleep infinity",
        "while true",
        "guardian",
    ):
        assert forbidden not in script.lower()


def test_cfg_guidance_runner_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(WORKLOAD_PATH)], check=True)
