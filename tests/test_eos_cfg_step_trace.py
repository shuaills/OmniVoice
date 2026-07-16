from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from omnivoice.blockdiff import _calibrate_blockwise_eos_log_probs  # noqa: E402
from omnivoice.blockdiff_dual import _eos_cfg_step_trace  # noqa: E402


def test_mass_preserving_trace_reports_mass_margin_and_actual_selection() -> None:
    # vocab: acoustic 0..3, mask 4, EOS 5; EOS is legal only on codebook 0.
    conditional = torch.tensor(
        [
            [
                [[2.0, 1.0, 0.0, -1.0, -8.0, -0.5],
                 [1.0, 0.0, -1.0, -2.0, -8.0, 2.5]],
                [[2.0, 1.0, 0.0, -1.0, -8.0, -8.0],
                 [1.0, 0.0, -1.0, -2.0, -8.0, -8.0]],
            ]
        ],
        dtype=torch.float32,
    )
    unconditional = conditional.clone()
    unconditional[:, 0, :, 0] += 0.75
    predicted = torch.zeros((1, 2, 2), dtype=torch.long)
    predicted[0, 0, 1] = 5

    trace = _eos_cfg_step_trace(
        conditional,
        unconditional,
        SimpleNamespace(
            guidance_scale=2.0,
            eos_cfg_calibration="mass_preserving",
        ),
        mask_id=4,
        eos_id=5,
        active_cb0=torch.tensor([True, True]),
        pred_tokens=predicted,
        queue_scores=torch.tensor([[[-0.2, 0.7], [-3.0, -4.0]]]),
        selected_flat_indices=torch.tensor([1, 2]),
        block_index=0,
        step_index=3,
        scheduled_positions=2,
        calibrate=_calibrate_blockwise_eos_log_probs,
    )

    assert trace["policy"] == "mass_preserving"
    assert trace["candidate_col"] == 1
    assert trace["class_eos"] == 1
    assert trace["selected_eos_cols"] == [1]
    assert trace["candidate_is_class_eos"] is True
    assert trace["selected"] is True
    assert trace["queue_rank"] == 1
    assert trace["queue_cutoff"] == pytest.approx(-3.0)
    assert trace["post_total_mass"] == pytest.approx(1.0, abs=1e-6)
    assert trace["post_eos_mass"] == pytest.approx(
        trace["conditional_eos_mass"], abs=1e-7
    )
    assert trace["guided_margin"] != trace["post_margin"]
    assert trace["legacy_margin"] != trace["guided_margin"]
    assert trace["legacy_queue_score"] != trace["queue_score"]


def test_trace_with_no_active_cb0_column_uses_json_safe_null_metrics() -> None:
    logits = torch.zeros((1, 2, 1, 6), dtype=torch.float32)
    trace = _eos_cfg_step_trace(
        logits,
        logits,
        SimpleNamespace(guidance_scale=0.0, eos_cfg_calibration="legacy"),
        mask_id=4,
        eos_id=5,
        active_cb0=torch.tensor([False]),
        pred_tokens=torch.zeros((1, 2, 1), dtype=torch.long),
        queue_scores=torch.tensor([[[-1.0], [-2.0]]]),
        selected_flat_indices=torch.tensor([1]),
        block_index=1,
        step_index=0,
        scheduled_positions=1,
        calibrate=_calibrate_blockwise_eos_log_probs,
    )

    assert trace["active_cb0"] == 0
    assert trace["candidate_col"] is None
    assert trace["post_total_mass"] is None
