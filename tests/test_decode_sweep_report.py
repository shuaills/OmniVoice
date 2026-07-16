import argparse
import csv
import importlib.util
import json
import wave
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "decode_sweep_report.py"
SPEC = importlib.util.spec_from_file_location("decode_sweep_report", MODULE_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)
WORKLOAD_PATH = Path(__file__).parents[1] / "decode_sweep_first300.sh"


def write_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    frames = round(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames)


def test_workload_locks_the_paired_decode_contract():
    script = WORKLOAD_PATH.read_text()
    assert "GUIDANCES=(2.0 1.5 3.0 2.0 1.5 3.0)" in script
    assert "STEPS=(16 16 16 32 32 32)" in script
    assert "GPU_IDS=${GPU_IDS:-0,1,2}" in script
    assert '--dtype bf16 \\' in script
    assert '--lang "$lang" \\' in script
    assert '--silence-stop-seconds 0 \\' in script
    assert 'mkdir "$wav_dir"' in script
    assert "--check-prompt-wavs" not in script
    assert "oms job submit" not in script
    assert "oms pod console" not in script
    assert "sleep infinity" not in script
    assert "while true" not in script


def test_summarize_arm_keeps_paired_metrics_and_tail_counts(tmp_path):
    ids = ["utt-a", "utt-b"]
    tsv = tmp_path / "first.tsv"
    tsv.write_text(
        "".join(f"{utt_id}\tprompt\tref.wav\ttarget\n" for utt_id in ids)
    )
    jsonl = tmp_path / "first.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps({"id": utt_id, "text": "target"}) + "\n" for utt_id in ids
        )
    )
    wav_dir = tmp_path / "wavs"
    anchor_dir = tmp_path / "anchors"
    wav_dir.mkdir()
    anchor_dir.mkdir()
    write_wav(wav_dir / "utt-a.wav", 1.0)
    write_wav(anchor_dir / "utt-a.wav", 2.0)
    write_wav(wav_dir / "utt-b.wav", 3.0)
    write_wav(anchor_dir / "utt-b.wav", 1.0)

    wer_tsv = tmp_path / "wer.tsv"
    wer_tsv.write_text(
        "Name\tWER\tTruth\tHypothesis\tInsertions\tDeletions\tSubstitutions\n"
        f"{wav_dir / 'utt-a.wav'}\t0.1\ttarget\ttarget\t0\t0\t0\n"
        f"{wav_dir / 'utt-b.wav'}\t0.6\ttarget\tbad\t1\t0\t0\n"
        "Seed-TTS WER (Avg of WERs): 35.0%\n"
    )
    sim_tsv = tmp_path / "sim.tsv"
    sim_tsv.write_text(
        "Prompt-path\tEval-path\tSIM-o\n"
        f"ref-a.wav\t{wav_dir / 'utt-a.wav'}\t0.70\n"
        f"ref-b.wav\t{wav_dir / 'utt-b.wav'}\t0.80\n"
        "\nAverage SIM-o: 0.750\n"
    )
    meta = wav_dir / "gen_meta_shard0.jsonl"
    meta.write_text(
        "".join(
            json.dumps(
                {
                    "utt_id": utt_id,
                    "frames": 10,
                    "eos": True,
                    "silence_stop": False,
                    "silence_run_frames": 0,
                    "stop_reason": "eos",
                    "n_blocks": 1,
                }
            )
            + "\n"
            for utt_id in ids
        )
    )

    per_utt = tmp_path / "per_utt.tsv"
    summary_path = tmp_path / "summary.json"
    summary = REPORT.summarize_arm(
        argparse.Namespace(
            arm="g2p0_s16",
            lang="en",
            guidance_scale=2.0,
            steps_per_block=16,
            expected_count=2,
            tsv=str(tsv),
            jsonl=str(jsonl),
            wav_dir=str(wav_dir),
            anchor_dir=str(anchor_dir),
            wer_tsv=str(wer_tsv),
            sim_tsv=str(sim_tsv),
            meta_glob=str(wav_dir / "gen_meta_shard*.jsonl"),
            failures_glob=str(wav_dir / "failures_shard*.jsonl"),
            per_utt_out=str(per_utt),
            summary_out=str(summary_path),
        )
    )

    assert summary["wer_percent"] == 35.0
    assert summary["sim"] == 0.75
    assert summary["runaway"] == 1
    assert summary["lt_0_6"] == 1
    assert summary["gt_2"] == 1
    assert summary["duration_ratio_median"] == 1.75
    with per_utt.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["id"] for row in rows] == ids


def test_aggregate_requires_and_ranks_the_full_matrix(tmp_path):
    summaries = []
    for lang in ("zh", "en"):
        for steps in (16, 32):
            for guidance in (1.5, 2.0, 3.0):
                arm = f"g{str(guidance).replace('.', 'p')}_s{steps}"
                path = tmp_path / f"{lang}-{arm}.json"
                path.write_text(
                    json.dumps(
                        {
                            "arm": arm,
                            "lang": lang,
                            "guidance_scale": guidance,
                            "steps_per_block": steps,
                            "count": 300,
                            "wer_percent": guidance,
                            "sim": guidance / 10 + steps / 1000,
                            "duration_ratio_mean": 1.0,
                            "duration_ratio_median": 1.0,
                            "duration_ratio_p95": 1.1,
                            "runaway": 0,
                            "lt_0_6": 0,
                            "gt_2": 0,
                            "eos_count": 300,
                            "max_blocks_count": 0,
                            "silence_stop_count": 0,
                            "per_utt_tsv": f"/{lang}/{arm}.tsv",
                        }
                    )
                )
                summaries.append(str(path))

    output_tsv = tmp_path / "SUMMARY.tsv"
    output_json = tmp_path / "SUMMARY.json"
    output_md = tmp_path / "SUMMARY.md"
    REPORT.aggregate(
        argparse.Namespace(
            summaries=summaries,
            expected_languages="zh,en",
            expected_guidance="1.5,2.0,3.0",
            expected_steps="16,32",
            expected_count=300,
            output_tsv=str(output_tsv),
            output_json=str(output_json),
            output_md=str(output_md),
        )
    )

    payload = json.loads(output_json.read_text())
    assert len(payload["rows"]) == 12
    assert payload["best_sim_by_language"]["zh"]["arm"] == "g3p0_s32"
    assert "| SIM rank |" in output_md.read_text()
