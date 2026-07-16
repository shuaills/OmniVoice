import re
import shlex
import subprocess
from pathlib import Path


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

    assert array(script, "ARM_NAMES") == [
        "shared_g2",
        "drop_ref_g0p5",
        "drop_ref_g1",
        "drop_ref_g1p5",
        "drop_ref_g2",
    ]
    assert array(script, "ARM_CFG_POLICIES") == [
        "shared",
        "drop_ref",
        "drop_ref",
        "drop_ref",
        "drop_ref",
    ]
    assert array(script, "ARM_GUIDANCES") == ["2.0", "0.5", "1.0", "1.5", "2.0"]
    assert "BASELINE_ARM=shared_g2" in script
    assert "PROMPT_CONTRACT=current" in script
    assert "LANG_POLICY=dataset" in script
    assert "--lang None" not in script
    assert "--prompt-contract \"$PROMPT_CONTRACT\"" in script
    assert "--cfg-unconditional-seed-policy \"$cfg_policy\"" in script


def test_cfg_guidance_runner_locks_subset_shards_seed_and_fixed_decode() -> None:
    script = WORKLOAD_PATH.read_text()
    canonical = CANONICAL_PATH.read_text()

    assert "EXPECTED_COUNT=${EXPECTED_COUNT:-100}" in script
    assert '100 | 300) ;;' in script
    assert "GPU_IDS=${GPU_IDS:-0,1,2}" in script
    assert '${#GPU_ARRAY[@]} == 3' in script
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
