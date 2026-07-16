import json
from pathlib import Path

import pytest

from tools.validate_silence_force_stop_run import AuditError, audit_run


IDS = ["utt0", "utt1", "utt2", "utt3"]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_run(tmp_path: Path):
    dataset = tmp_path / "test.tsv"
    refs = tmp_path / "refs"
    refs.mkdir()
    tsv_rows = []
    jsonl_rows = []
    for index, utt_id in enumerate(IDS):
        ref = refs / f"{utt_id}.wav"
        ref.touch()
        text = "target text " + ("x" * (index * 30))
        tsv_rows.append(f"{utt_id}\tprompt\t{ref}\t{text}\n")
        jsonl_rows.append(
            json.dumps({"id": utt_id, "text": text, "ref_audio": str(ref)}) + "\n"
        )
    _write(dataset, "".join(tsv_rows))
    test_jsonl = tmp_path / "test.jsonl"
    _write(test_jsonl, "".join(jsonl_rows))

    logs = tmp_path / "logs"
    controls = tmp_path / "control"
    forced = tmp_path / "forced"
    for arm, out in (("control", controls), ("forced", forced)):
        out.mkdir()
        for utt_id in IDS:
            (out / f"{utt_id}.wav").touch()
        for shard in range(2):
            rows = []
            for index in range(shard, len(IDS), 2):
                triggered = arm == "forced" and index in (1, 3)
                rows.append(
                    {
                        "utt_id": IDS[index],
                        "frames": 80 + index * 80 - (30 if triggered else 0),
                        "eos": not triggered,
                        "silence_stop": triggered,
                        "stop_reason": "silence" if triggered else "eos",
                        "silence_col": 40 if triggered else None,
                        "silence_trigger_col": 64 if triggered else None,
                        "silence_run_frames": 25 if arm == "forced" else 0,
                        "silence_match_codebooks": 2,
                        "min_gen_frames": 8 + index,
                        "seed_frames": 20 + index,
                        "n_blocks": 3,
                    }
                )
            _write(
                out / f"gen_meta_shard{shard}.jsonl",
                "".join(json.dumps(row) + "\n" for row in rows),
            )
            (out / f"failures_shard{shard}.jsonl").touch()
            count = len(rows)
            _write(
                logs / f"gen_en_{arm}_shard{shard}.log",
                f"[shard {shard}/2] FINISHED total={count} done={count} "
                "skip=0 fail=0\n",
            )

    scores = {}
    for arm, out in (("control", controls), ("forced", forced)):
        wer = tmp_path / f"wer_{arm}.tsv"
        _write(
            wer,
            "Name\tWER\tTruth\tHypothesis\tInsertions\tDeletions\tSubstitutions\n"
            + "".join(f"{out / (utt_id + '.wav')}\t0\tt\tt\t0\t0\t0\n" for utt_id in IDS)
            + "Seed-TTS WER (Avg of WERs): 0.0%\n",
        )
        sim = tmp_path / f"sim_{arm}.tsv"
        _write(
            sim,
            "Prompt-path\tEval-path\tSIM-o\n"
            + "".join(
                f"English\t{refs / (utt_id + '.wav')}\t"
                f"{out / (utt_id + '.wav')}\t0.75\n"
                for utt_id in IDS
            )
            + "\nAverage SIM-o: 0.750\n",
        )
        scores[f"wer_{arm}"] = wer
        scores[f"sim_{arm}"] = sim
    return dataset, test_jsonl, controls, forced, logs, scores


def _audit(tmp_path: Path):
    dataset, test_jsonl, control, forced, logs, scores = _build_run(tmp_path)
    return audit_run(
        lang="en",
        dataset_tsv=dataset,
        test_jsonl=test_jsonl,
        control_dir=control,
        forced_dir=forced,
        logs_dir=logs,
        num_shards=2,
        match_codebooks=2,
        score_paths=scores,
    )


def test_accepts_complete_paired_run_and_reports_trigger_strata(tmp_path):
    summary = _audit(tmp_path)
    assert summary["expected_ids"] == 4
    assert summary["scores"] == {
        "wer_control": 4,
        "wer_forced": 4,
        "sim_control": 4,
        "sim_forced": 4,
    }
    assert summary["trigger"]["triggered"] == 2
    assert summary["trigger"]["strata"]["control_frames"]["65_128"] == {
        "n": 1,
        "triggered": 0,
        "rate": 0.0,
    }


def test_rejects_nonempty_failure_jsonl(tmp_path):
    dataset, test_jsonl, control, forced, logs, scores = _build_run(tmp_path)
    _write(forced / "failures_shard1.jsonl", '{"utt_id":"utt1"}\n')
    with pytest.raises(AuditError, match="non-empty failure JSONL"):
        audit_run(
            lang="en",
            dataset_tsv=dataset,
            test_jsonl=test_jsonl,
            control_dir=control,
            forced_dir=forced,
            logs_dir=logs,
            num_shards=2,
            match_codebooks=2,
            score_paths=scores,
        )


def test_rejects_extra_stale_wav(tmp_path):
    dataset, test_jsonl, control, forced, logs, scores = _build_run(tmp_path)
    (forced / "stale_from_previous_run.wav").touch()
    with pytest.raises(AuditError, match="wav IDs differ from dataset"):
        audit_run(
            lang="en",
            dataset_tsv=dataset,
            test_jsonl=test_jsonl,
            control_dir=control,
            forced_dir=forced,
            logs_dir=logs,
            num_shards=2,
            match_codebooks=2,
            score_paths=scores,
        )


def test_rejects_stale_cross_arm_score_path(tmp_path):
    dataset, test_jsonl, control, forced, logs, scores = _build_run(tmp_path)
    sim_forced = scores["sim_forced"]
    sim_forced.write_text(
        sim_forced.read_text(encoding="utf-8").replace(
            str(forced / "utt3.wav"), str(control / "utt3.wav")
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuditError, match="another output directory"):
        audit_run(
            lang="en",
            dataset_tsv=dataset,
            test_jsonl=test_jsonl,
            control_dir=control,
            forced_dir=forced,
            logs_dir=logs,
            num_shards=2,
            match_codebooks=2,
            score_paths=scores,
        )
