#!/usr/bin/env bash
# Non-interactive OMS entrypoint for the EOS/CFG calibration campaign.
# The job launch command invokes this file directly, avoiding nested shell
# quoting; both the test preflight and campaign exit with the job.
set -Eeuo pipefail

C=${C:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
DL=${DL:-/opt/gpfs/users/yinfeng/work/OmniVoice}
RUN_EOS_CFG_TESTS=${RUN_EOS_CFG_TESTS:-1}

case "$RUN_EOS_CFG_TESTS" in
  0) ;;
  1)
    # shellcheck disable=SC1090
    source "$DL/.venv/bin/activate"
    export PYTHONPATH=$C
    cd "$C"
    pytest -q \
      tests/test_block_eos_cfg_calibration.py \
      tests/test_eos_cfg_step_trace.py \
      tests/test_legacy_fixed_canvas_guard.py \
      tests/test_eos_cfg_calibration_eval_surface.py \
      tests/test_training_contract_probe_report.py \
      tests/test_block_dual_cpu.py \
      tests/test_blockdiff_smoke.py \
      tests/test_cfg_unconditional_seed_policy.py
    ;;
  *)
    echo "ERROR: RUN_EOS_CFG_TESTS must be 0 or 1, got $RUN_EOS_CFG_TESTS" >&2
    exit 2
    ;;
esac

exec bash "$C/training_contract_eos_cfg_calibration_probe.sh"
