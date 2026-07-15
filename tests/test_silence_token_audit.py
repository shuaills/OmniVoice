import json

import numpy as np

from tools.silence_token_audit import SILENCE_FRAME_TOKENS, audit


def test_audit_respects_minimum_frames_and_subthreshold_tail(tmp_path):
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()

    before_min = np.zeros((8, 60), dtype=np.int16)
    before_min[:3, 0:25] = SILENCE_FRAME_TOKENS[:3, None]
    np.save(token_dir / "before_min.npy", before_min)

    ends_before_threshold = np.zeros((8, 60), dtype=np.int16)
    ends_before_threshold[:3, 36:60] = SILENCE_FRAME_TOKENS[:3, None]
    np.save(token_dir / "ends_before_threshold.npy", ends_before_threshold)

    silence_wins = np.zeros((8, 80), dtype=np.int16)
    silence_wins[:3, 20:45] = SILENCE_FRAME_TOKENS[:3, None]
    np.save(token_dir / "silence_wins.npy", silence_wins)

    meta = {
        "before_min": {"utt_id": "before_min", "frames": 60, "eos": False,
                       "min_gen_frames": 25},
        "ends_before_threshold": {
            "utt_id": "ends_before_threshold",
            "frames": 60,
            "eos": True,
            "min_gen_frames": 0,
        },
        "silence_wins": {"utt_id": "silence_wins", "frames": 80, "eos": False,
                         "min_gen_frames": 0},
    }
    summary = audit(token_dir, meta, [3], 25)
    width = summary["widths"]["3"]
    assert width["run_ge_threshold"] == 1
    assert width["would_force_stop"] == 1
    assert width["would_force_stop_ids"] == ["silence_wins"]
    assert width["trimmed_frames"] == [60]


def test_audit_output_is_json_serializable(tmp_path):
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    tokens = np.zeros((8, 30), dtype=np.int16)
    np.save(token_dir / "plain.npy", tokens)
    summary = audit(
        token_dir,
        {"plain": {"utt_id": "plain", "frames": 30, "eos": False}},
        [1, 2, 3],
        25,
    )
    json.dumps(summary)
