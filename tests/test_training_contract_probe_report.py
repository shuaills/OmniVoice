import argparse
import csv
import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "training_contract_probe_report.py"
)
SPEC = importlib.util.spec_from_file_location("training_contract_probe_report", MODULE_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)
WORKLOAD_PATH = Path(__file__).parents[1] / "training_contract_probe_first100.sh"


def test_workload_locks_the_seven_arm_self_terminating_contract() -> None:
    script = WORKLOAD_PATH.read_text()

    assert "EXPECTED_COUNT=${EXPECTED_COUNT:-100}" in script
    assert "GPU_IDS=${GPU_IDS:-0,1,2}" in script
    assert (
        "ARM_NAMES=(baseline lang_none prompt_official eval_parity "
        "cfg_drop_ref combined gs0)"
    ) in script
    assert "PROMPT_CONTRACT_FLAG=${PROMPT_CONTRACT_FLAG:---prompt-contract}" in script
    assert "---cfg-unconditional-seed-policy" in script
    assert "ARM_GUIDANCES=(2.0 2.0 2.0 2.0 2.0 2.0 0)" in script
    assert '[[ $lang_policy == dataset ]] || LANG_ARGS+=(--lang "$LANG_NONE_VALUE")' in script
    assert 'mkdir "$wav_dir"' in script
    assert "--item-error-policy \"$ITEM_ERROR_POLICY\"" in script
    assert "oms job submit" not in script
    assert "oms pod console" not in script
    assert "sleep infinity" not in script
    assert "while true" not in script


def test_workload_seven_arm_argv_contracts_are_unique() -> None:
    script = WORKLOAD_PATH.read_text()

    def array(name: str) -> list[str]:
        match = re.search(rf"{name}=\((.*?)\)", script, flags=re.DOTALL)
        assert match is not None
        return shlex.split(match.group(1))

    names = array("ARM_NAMES")
    contracts = list(
        zip(
            array("ARM_LANG_POLICIES"),
            array("ARM_PROMPTS"),
            array("ARM_CFG_POLICIES"),
            array("ARM_GUIDANCES"),
            strict=True,
        )
    )
    assert len(names) == len(contracts) == 7
    assert len(set(names)) == len(names)
    assert len(set(contracts)) == len(contracts)


def make_input_fixture(tmp_path: Path, count: int = 3) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    prompts = source / "prompt_wavs"
    refs = tmp_path / "refs"
    anchors = tmp_path / "anchors"
    prompts.mkdir(parents=True)
    refs.mkdir()
    anchors.mkdir()

    tsv = source / "first300.tsv"
    jsonl = source / "first300.jsonl"
    with tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index in range(count):
            utt_id = f"utt-{index}"
            prompt = prompts / f"prompt-{index}.wav"
            ref = refs / f"ref-{index}.wav"
            anchor = anchors / f"{utt_id}.wav"
            prompt.write_bytes(f"prompt-{index}".encode())
            ref.write_bytes(f"ref-{index}".encode())
            anchor.write_bytes(f"anchor-{index}".encode())
            writer.writerow([utt_id, "prompt", str(prompt), "target"])
    with jsonl.open("w") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "id": f"utt-{index}",
                        "text": "target",
                        "ref_audio": str(refs / f"ref-{index}.wav"),
                    }
                )
                + "\n"
            )
    return tsv, jsonl, anchors


def test_prepare_inputs_is_fresh_ordered_and_content_fingerprinted(tmp_path: Path) -> None:
    tsv, jsonl, anchors = make_input_fixture(tmp_path)
    output = tmp_path / "prepared"
    args = argparse.Namespace(
        source_tsv=str(tsv),
        source_jsonl=str(jsonl),
        count=2,
        output_dir=str(output),
        anchor_dir=str(anchors),
        check_ref_audio=True,
    )

    REPORT.prepare_inputs(args)

    assert [row[0] for row in csv.reader((output / "test.tsv").open(), delimiter="\t")] == [
        "utt-0",
        "utt-1",
    ]
    assert (output / "prompt_wavs").is_symlink()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["count"] == 2
    assert len(manifest["prompt_wavs"]) == 2
    assert len(manifest["reference_audio"]) == 2
    assert len(manifest["duration_anchors"]) == 2
    assert all(len(row["sha256"]) == 64 for row in manifest["prompt_wavs"])
    with pytest.raises(ValueError, match="refusing to reuse"):
        REPORT.prepare_inputs(args)


def write_generation_contract(
    wav_dir: Path,
    *,
    prompt_contract: str,
    language: str | None,
    cfg_policy: str,
    guidance: float,
) -> None:
    wav_dir.mkdir()
    for shard in range(2):
        (wav_dir / f"failures_shard{shard}.jsonl").write_text("")
        rows = []
        for index in range(shard, 2, 2):
            utt_id = f"utt-{index}"
            (wav_dir / f"{utt_id}.wav").write_bytes(b"wav")
            rows.append(
                {
                    "utt_id": utt_id,
                    "frames": 20,
                    "eos": True,
                    "silence_stop": False,
                    "silence_run_frames": 0,
                    "stop_reason": "eos",
                    "n_blocks": 1,
                    "prompt_contract": prompt_contract,
                    "language": language,
                    "ref_text_punctuation": (
                        "preserve" if prompt_contract == "official-emilia" else "add"
                    ),
                    "cfg_unconditional_seed_policy": cfg_policy,
                    "guidance_scale": guidance,
                    "generation_seed": 20260707 + index,
                    **(
                        {
                            "ref_rms": 0.2,
                            "ref_truncated_samples": 3,
                            "output_ref_rms_restored": False,
                        }
                        if prompt_contract == "official-emilia"
                        else {}
                    ),
                }
            )
        (wav_dir / f"gen_meta_shard{shard}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_validate_generation_audits_nondefault_contract_and_seed(tmp_path: Path) -> None:
    tsv, jsonl, _ = make_input_fixture(tmp_path, count=2)
    wav_dir = tmp_path / "wavs"
    write_generation_contract(
        wav_dir,
        prompt_contract="official-emilia",
        language=None,
        cfg_policy="drop_ref",
        guidance=2.0,
    )
    output = tmp_path / "audit.json"
    args = argparse.Namespace(
        tsv=str(tsv),
        jsonl=str(jsonl),
        expected_count=2,
        num_shards=2,
        arm="combined",
        lang="zh",
        lang_policy="none",
        prompt_contract="official-emilia",
        cfg_unconditional_seed_policy="drop_ref",
        guidance_scale=2.0,
        seed_base=20260707,
        is_baseline=False,
        wav_dir=str(wav_dir),
        output=str(output),
    )

    REPORT.validate_generation(args)

    payload = json.loads(output.read_text())
    assert payload["arm"] == "combined"
    assert payload["stop_reasons"] == {"eos": 2}
    meta_path = wav_dir / "gen_meta_shard1.jsonl"
    row = json.loads(meta_path.read_text())
    row["generation_seed"] += 1
    meta_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="metadata contract mismatch"):
        REPORT.validate_generation(args)


def test_validate_generation_preserves_filtered_source_seed_indices(
    tmp_path: Path,
) -> None:
    tsv, jsonl, _ = make_input_fixture(tmp_path, count=2)
    wav_dir = tmp_path / "filtered-wavs"
    write_generation_contract(
        wav_dir,
        prompt_contract="current",
        language="en",
        cfg_policy="shared",
        guidance=2.0,
    )
    seed_indices = {"utt-0": 76, "utt-1": 143}
    seed_map = tmp_path / "seed-index-map.json"
    seed_map.write_text(json.dumps(seed_indices))
    for shard in range(2):
        meta_path = wav_dir / f"gen_meta_shard{shard}.jsonl"
        row = json.loads(meta_path.read_text())
        source_index = seed_indices[row["utt_id"]]
        row["generation_seed"] = 20260707 + source_index
        row["generation_seed_index"] = source_index
        row["generation_seed_value"] = 20260707 + source_index
        meta_path.write_text(json.dumps(row) + "\n")

    args = argparse.Namespace(
        tsv=str(tsv),
        jsonl=str(jsonl),
        expected_count=2,
        num_shards=2,
        arm="renorm",
        lang="en",
        lang_policy="dataset",
        prompt_contract="current",
        cfg_unconditional_seed_policy="shared",
        guidance_scale=2.0,
        seed_base=20260707,
        seed_index_map=str(seed_map),
        is_baseline=False,
        wav_dir=str(wav_dir),
        output=str(tmp_path / "filtered-audit.json"),
    )

    REPORT.validate_generation(args)

    meta_path = wav_dir / "gen_meta_shard0.jsonl"
    row = json.loads(meta_path.read_text())
    row["generation_seed_value"] += 1
    meta_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="canonical generation seed mismatch"):
        REPORT.validate_generation(args)


def test_validate_generation_locks_legacy_baseline_meta_shape(tmp_path: Path) -> None:
    tsv, jsonl, _ = make_input_fixture(tmp_path, count=2)
    wav_dir = tmp_path / "baseline-wavs"
    wav_dir.mkdir()
    for shard in range(2):
        (wav_dir / f"failures_shard{shard}.jsonl").write_text("")
        index = shard
        utt_id = f"utt-{index}"
        (wav_dir / f"{utt_id}.wav").write_bytes(b"wav")
        (wav_dir / f"gen_meta_shard{shard}.jsonl").write_text(
            json.dumps(
                {
                    "utt_id": utt_id,
                    "frames": 20,
                    "eos": True,
                    "silence_stop": False,
                    "silence_run_frames": 0,
                    "stop_reason": "eos",
                    "n_blocks": 1,
                }
            )
            + "\n"
        )
    args = argparse.Namespace(
        tsv=str(tsv),
        jsonl=str(jsonl),
        expected_count=2,
        num_shards=2,
        arm="baseline",
        lang="zh",
        lang_policy="dataset",
        prompt_contract="current",
        cfg_unconditional_seed_policy="shared",
        guidance_scale=2.0,
        seed_base=20260707,
        is_baseline=True,
        wav_dir=str(wav_dir),
        output=str(tmp_path / "baseline-audit.json"),
    )

    REPORT.validate_generation(args)

    meta_path = wav_dir / "gen_meta_shard0.jsonl"
    row = json.loads(meta_path.read_text())
    row["guidance_scale"] = 2.0
    meta_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="baseline metadata shape changed"):
        REPORT.validate_generation(args)


def summary_row(
    tmp_path: Path, lang: str, arm: str, sims: tuple[float, float]
) -> Path:
    per_utt = tmp_path / f"{lang}-{arm}.tsv"
    with per_utt.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["id", "sim"])
        writer.writeheader()
        writer.writerows(
            [{"id": "utt-0", "sim": sims[0]}, {"id": "utt-1", "sim": sims[1]}]
        )
    summary = {
        "lang": lang,
        "arm": arm,
        "lang_policy": "dataset",
        "prompt_contract": "current",
        "cfg_unconditional_seed_policy": "shared",
        "guidance_scale": 2.0,
        "steps_per_block": 16,
        "count": 2,
        "wer_percent": 1.0,
        "sim": sum(sims) / 2,
        "duration_ratio_mean": 1.1,
        "duration_ratio_median": 1.0,
        "duration_ratio_p95": 1.4,
        "runaway": 0,
        "lt_0_6": 0,
        "gt_2": 0,
        "eos_count": 2,
        "max_blocks_count": 0,
        "per_utt_tsv": str(per_utt),
    }
    path = tmp_path / f"{lang}-{arm}.json"
    path.write_text(json.dumps(summary))
    return path


def test_aggregate_reports_paired_sim_delta_against_baseline(tmp_path: Path) -> None:
    arms = [
        "baseline",
        "lang_none",
        "prompt_official",
        "eval_parity",
        "cfg_drop_ref",
        "combined",
        "gs0",
    ]
    summaries = []
    for lang in ("zh", "en"):
        for index, arm in enumerate(arms):
            sims = (0.60, 0.70) if arm == "baseline" else (0.61 + index / 100, 0.69)
            summaries.append(str(summary_row(tmp_path, lang, arm, sims)))

    output_tsv = tmp_path / "SUMMARY.tsv"
    output_json = tmp_path / "SUMMARY.json"
    output_md = tmp_path / "SUMMARY.md"
    REPORT.aggregate(
        argparse.Namespace(
            summaries=summaries,
            expected_arms=",".join(arms),
            expected_languages="zh,en",
            baseline_arm="baseline",
            expected_count=2,
            output_tsv=str(output_tsv),
            output_json=str(output_json),
            output_md=str(output_md),
        )
    )

    rows = json.loads(output_json.read_text())["rows"]
    zh_baseline = next(
        row for row in rows if row["lang"] == "zh" and row["arm"] == "baseline"
    )
    zh_combined = next(
        row for row in rows if row["lang"] == "zh" and row["arm"] == "combined"
    )
    assert zh_baseline["paired_sim_delta_mean"] == 0
    assert zh_baseline["paired_sim_equal"] == 2
    assert zh_combined["paired_sim_delta_mean"] == pytest.approx(0.025)
    assert "paired SIM Δ mean" in output_md.read_text()
