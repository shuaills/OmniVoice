import ast
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
CAMPAIGN = ROOT / "training_contract_eos_cfg_calibration_probe.sh"
ENTRYPOINT = ROOT / "eos_cfg_calibration_oms_entrypoint.sh"
GENERATOR = ROOT / "tests" / "seedtts_blockwise_gen.py"
DUAL = ROOT / "omnivoice" / "blockdiff_dual.py"


def array(script: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\((.*?)\)$", script, flags=re.MULTILINE)
    assert match is not None
    return shlex.split(match.group(1))


def run_contract(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "C": str(tmp_path / "missing-repo"),
            "RUN_ID": "pytest-contract",
            "RESULT_ROOT": str(tmp_path / "results"),
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(CAMPAIGN)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generator_exposes_explicit_calibration_and_opt_in_trace() -> None:
    source = GENERATOR.read_text()
    ast.parse(source)

    assert "'--eos-cfg-calibration'" in source
    assert "default='legacy'" in source
    assert "choices=['legacy', 'renorm', 'mass_preserving']" in source
    assert "'--eos-cfg-trace'" in source
    assert "'--generation-seed-index-map'" in source
    assert "action='store_true'" in source
    assert "gen.eos_cfg_calibration = args.eos_cfg_calibration" in source
    assert "eos_cfg_trace = [] if args.eos_cfg_trace else None" in source
    assert "'eos_cfg_calibration': args.eos_cfg_calibration" in source
    assert "'seed_frames_mod_block': seed_frames_mod_block" in source
    assert "'first_target_block_frames': first_target_block_frames" in source
    assert "'generation_seed_index': generation_seed_index" in source
    assert "'generation_seed_value': generation_seed" in source
    assert "meta['eos_cfg_trace'] = eos_cfg_trace" in source
    experimental_contract = source.split("experimental_contract = (", 1)[1].split(
        ")", 1
    )[0]
    assert "args.eos_cfg_trace" not in experimental_contract


def test_decoder_trace_is_optional_compact_and_records_actual_selection() -> None:
    source = DUAL.read_text()
    ast.parse(source)

    assert "eos_cfg_trace: Optional[list] = None" in source
    assert "if eos_cfg_trace is not None:" in source
    for key in (
        '"guided_eos_mass"',
        '"conditional_eos_mass"',
        '"legacy_eos_mass"',
        '"post_eos_mass"',
        '"legacy_total_mass"',
        '"post_total_mass"',
        '"guided_margin"',
        '"legacy_margin"',
        '"post_margin"',
        '"queue_cutoff"',
        '"queue_rank"',
        '"legacy_queue_cutoff"',
        '"legacy_queue_rank"',
        '"selected_eos_cols"',
    ):
        assert key in source
    assert source.index("_, topk_idx = torch.topk") < source.index(
        "if eos_cfg_trace is not None:"
    )


def test_campaign_is_a_locked_three_arm_paired_probe() -> None:
    script = CAMPAIGN.read_text()

    assert array(script, "ARM_NAMES") == ["legacy", "renorm", "mass_preserving"]
    assert array(script, "ARM_EOS_CFG_CALIBRATIONS") == [
        "legacy",
        "renorm",
        "mass_preserving",
    ]
    assert "BASELINE_ARM=legacy" in script
    for fixed in (
        "GUIDANCE_SCALE=2.0",
        "CFG_UNCONDITIONAL_SEED_POLICY=shared",
        "STEPS_PER_BLOCK=16",
        "BLOCK_SIZE=32",
        "MAX_BLOCKS=24",
        "PROMPT_CONTRACT=current",
        "EXPECTED_COUNT=${EXPECTED_COUNT:-100}",
        "EOS_CFG_TRACE=${EOS_CFG_TRACE:-0}",
        "GPU_IDS=${GPU_IDS:-0,1}",
    ):
        assert fixed in script
    assert '--eos-cfg-calibration "$eos_calibration"' in script
    assert '"${TRACE_ARGS[@]}"' in script
    assert '${#GPU_ARRAY[@]} >= 2' in script
    assert '--shard "$shard/${#GPU_ARRAY[@]}"' in script
    assert '--num-shards "${#GPU_ARRAY[@]}"' in script


def test_campaign_supports_fail_closed_language_and_utterance_filters() -> None:
    script = CAMPAIGN.read_text()

    assert "LANGS=${LANGS:-zh,en}" in script
    assert "UTT_IDS=${UTT_IDS:-}" in script
    assert "UTT_REGEX=${UTT_REGEX:-}" in script
    assert "UTT_IDS and UTT_REGEX are mutually exclusive" in script
    assert "filter selected {len(selected)} rows" in script
    assert '"source_index": index' in script
    assert '"canonical_source_tsv_sha256"' in script
    assert '"canonical_source_jsonl_sha256"' in script
    assert 'seed_index_map.json' in script
    assert '--generation-seed-index-map "$seed_index_map"' in script
    assert '--seed-index-map "$seed_index_map"' in script


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"EXPECTED_COUNT": "1", "UTT_IDS": "a", "UTT_REGEX": "a"},
            "UTT_IDS and UTT_REGEX are mutually exclusive",
        ),
        (
            {"EXPECTED_COUNT": "2", "UTT_IDS": "a"},
            "UTT_IDS count must equal EXPECTED_COUNT",
        ),
        ({"LANGS": "fr"}, "LANGS supports only zh,en"),
        ({"LANGS": "en,en"}, "duplicate language in LANGS"),
        ({"EOS_CFG_TRACE": "yes"}, "EOS_CFG_TRACE must be 0 or 1"),
        ({"GPU_IDS": "0"}, "GPU_IDS must provide at least two GPUs"),
    ],
)
def test_campaign_rejects_invalid_microprobe_contract_before_io(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    result = run_contract(tmp_path, **overrides)

    assert result.returncode != 0
    assert message in result.stderr


def test_campaign_reuses_canonical_report_and_self_terminates() -> None:
    script = CAMPAIGN.read_text()

    for command in ("prepare-inputs", "validate-generation", "summarize-arm", "aggregate"):
        assert f'python "$REPORTER" {command}' in script
    assert "omnivoice/eval/wer/seedtts.py" in script
    assert "omnivoice/eval/speaker_similarity/sim.py" in script
    assert "trap cleanup EXIT" in script
    assert "wait_group \"generation arm=$arm lang=$lang\"" in script
    assert "wait_group \"scoring arm=$arm lang=$lang\"" in script
    assert "EOS_CFG_CALIBRATION_PROBE_DONE" in script
    for forbidden in (
        "oms job submit",
        "oms pod console",
        "sleep infinity",
        "while true",
        "guardian",
    ):
        assert forbidden not in script.lower()


def test_oms_entrypoint_runs_preflight_then_execs_campaign_without_guardian() -> None:
    script = ENTRYPOINT.read_text()

    assert "set -Eeuo pipefail" in script
    assert "RUN_EOS_CFG_TESTS=${RUN_EOS_CFG_TESTS:-1}" in script
    assert "python scripts/eos_cfg_calibration_preflight.py" in script
    assert "python tests/test_block_dual_cpu.py" not in script
    assert 'exec bash "$C/training_contract_eos_cfg_calibration_probe.sh"' in script
    for forbidden in ("sleep infinity", "while true", "oms pod console"):
        assert forbidden not in script.lower()

    subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)


def test_campaign_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(CAMPAIGN)], check=True)
