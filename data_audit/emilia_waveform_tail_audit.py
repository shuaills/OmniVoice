#!/usr/bin/env python3
"""Estimate speech-end-to-clip-end gaps on a spread Emilia waveform sample."""

import argparse
import io
import json
import math
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf


def estimate_tail_seconds(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = 0.02,
    sustain_seconds: float = 0.30,
    active_fraction: float = 0.40,
    margin_db: float = 30.0,
) -> dict:
    """Estimate trailing non-speech with sustained pre-emphasized activity.

    Pre-emphasis suppresses low-frequency hum.  The sustained-activity vote
    prevents an isolated click or bang in an otherwise quiet tail from moving
    the estimated speech end to the clip boundary.
    """
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected nonempty mono/stereo waveform, got {values.shape}")
    values = np.nan_to_num(values)
    emphasized = np.empty_like(values)
    emphasized[0] = values[0]
    emphasized[1:] = values[1:] - 0.97 * values[:-1]

    frame_size = max(1, int(round(frame_seconds * sample_rate)))
    frame_count = values.size // frame_size
    if frame_count < 2:
        return {
            "duration_s": values.size / sample_rate,
            "speech_end_s": values.size / sample_rate,
            "tail_s": 0.0,
            "threshold_db": None,
            "reference_db": None,
        }
    framed = emphasized[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1) + 1e-12)
    levels = 20.0 * np.log10(rms + 1e-12)
    reference_db = float(np.quantile(levels, 0.90))
    threshold_db = max(-65.0, reference_db - margin_db)
    active = levels >= threshold_db

    sustain_frames = max(1, int(round(sustain_seconds / frame_seconds)))
    required = max(1, int(math.ceil(sustain_frames * active_fraction)))
    votes = np.convolve(
        active.astype(np.int16),
        np.ones(sustain_frames, dtype=np.int16),
        mode="same",
    )
    supported_active = active & (votes >= required)
    hits = np.flatnonzero(supported_active)
    duration_s = values.size / sample_rate
    if hits.size == 0:
        speech_end_s = 0.0
    else:
        speech_end_frame = min(frame_count, int(hits[-1]) + 1)
        speech_end_s = min(duration_s, speech_end_frame * frame_seconds)
    return {
        "duration_s": duration_s,
        "speech_end_s": speech_end_s,
        "tail_s": max(0.0, duration_s - speech_end_s),
        "threshold_db": threshold_db,
        "reference_db": reference_db,
    }


def _selected_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    return [int(value) for value in np.linspace(0, total - 1, count + 2)[1:-1]]


def _read_selected_labels(path: Path, count: int) -> list[dict]:
    with path.open() as handle:
        lines = handle.readlines()
    return [json.loads(lines[index]) for index in _selected_indices(len(lines), count)]


def _audit_source(
    language: str,
    chunk: str,
    source_tar: Path,
    label_jsonl: Path,
    samples_per_source: int,
    margins_db: list[float],
) -> list[dict]:
    labels = _read_selected_labels(label_jsonl, samples_per_source)
    output = []
    with tarfile.open(source_tar, "r") as archive:
        for label in labels:
            utt_id = label["id"]
            member = archive.getmember(f"{utt_id}.opus")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"could not extract {member.name}")
            audio, sample_rate = sf.read(
                io.BytesIO(extracted.read()), dtype="float32", always_2d=False
            )
            estimates = {
                str(int(margin)): estimate_tail_seconds(
                    audio, sample_rate, margin_db=margin
                )
                for margin in margins_db
            }
            output.append(
                {
                    "id": utt_id,
                    "language": language,
                    "chunk": chunk,
                    "source_tar": str(source_tar),
                    "label_duration_s": float(label["duration"]),
                    "sample_rate": sample_rate,
                    "estimates": estimates,
                }
            )
    return output


def _summarize(rows: list[dict], margins_db: list[float]) -> dict:
    summary = {"samples": len(rows), "languages": {}, "margins_db": margins_db}
    languages = sorted({row["language"] for row in rows})
    for language in languages + ["all"]:
        selected = rows if language == "all" else [
            row for row in rows if row["language"] == language
        ]
        language_summary = {"samples": len(selected), "tail": {}}
        for margin in margins_db:
            key = str(int(margin))
            tails = sorted(row["estimates"][key]["tail_s"] for row in selected)
            values = np.asarray(tails, dtype=np.float64)
            language_summary["tail"][key] = {
                "mean_s": float(values.mean()),
                "median_s": float(np.quantile(values, 0.50)),
                "p90_s": float(np.quantile(values, 0.90)),
                "p95_s": float(np.quantile(values, 0.95)),
                "gt_0p2": int((values > 0.2).sum()),
                "gt_0p5": int((values > 0.5).sum()),
                "gt_1p0": int((values > 1.0).sum()),
            }
        summary["languages"][language] = language_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-root", type=Path, required=True)
    parser.add_argument("--languages", default="en,zh")
    parser.add_argument("--sources-per-chunk", type=int, default=1)
    parser.add_argument("--samples-per-source", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--margins-db", default="25,30,35")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    languages = args.languages.split(",")
    margins_db = [float(value) for value in args.margins_db.split(",")]
    if args.sources_per_chunk < 1 or args.samples_per_source < 1:
        raise ValueError("sample counts must be positive")

    sources = []
    for language in languages:
        lists = sorted(args.provenance_root.glob(f"{language}_chunk*.lst"))
        if len(lists) != 10:
            raise ValueError(f"expected 10 {language} chunk lists, found {len(lists)}")
        for list_path in lists:
            lines = [line.split() for line in list_path.read_text().splitlines()]
            for index in _selected_indices(len(lines), args.sources_per_chunk):
                source_tar, label_jsonl = lines[index][:2]
                sources.append(
                    (
                        language,
                        list_path.stem,
                        Path(source_tar),
                        Path(label_jsonl),
                        args.samples_per_source,
                        margins_db,
                    )
                )

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_audit_source, *source) for source in sources]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row["language"], row["chunk"], row["id"]))

    expected = len(sources) * args.samples_per_source
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} samples, audited {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = _summarize(rows, margins_db)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
