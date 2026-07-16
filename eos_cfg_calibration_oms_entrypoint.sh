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
    python scripts/eos_cfg_calibration_preflight.py
    ;;
  *)
    echo "ERROR: RUN_EOS_CFG_TESTS must be 0 or 1, got $RUN_EOS_CFG_TESTS" >&2
    exit 2
    ;;
esac

exec bash "$C/training_contract_eos_cfg_calibration_probe.sh"
