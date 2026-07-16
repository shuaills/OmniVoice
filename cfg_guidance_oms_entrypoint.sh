#!/usr/bin/env bash
# Non-interactive OMS entrypoint for the shared-reference CFG guidance sweep.
# The launch command invokes this file through `env ... bash`, so no nested
# shell quoting or long-lived guardian process is required.
set -Eeuo pipefail

C=${C:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
DL=${DL:-/opt/gpfs/users/yinfeng/work/OmniVoice}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
RUN_CFG_SWEEP_TESTS=${RUN_CFG_SWEEP_TESTS:-1}

cd "$C"
actual_commit=$(git rev-parse HEAD)
if [[ $actual_commit != "$EXPECTED_COMMIT" ]]; then
  echo "ERROR: commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi

case "$RUN_CFG_SWEEP_TESTS" in
  0) ;;
  1)
    # shellcheck disable=SC1090
    source "$DL/.venv/bin/activate"
    export PYTHONPATH=$C
    python tests/test_cfg_unconditional_seed_policy.py
    ;;
  *)
    echo "ERROR: RUN_CFG_SWEEP_TESTS must be 0 or 1, got $RUN_CFG_SWEEP_TESTS" >&2
    exit 2
    ;;
esac

exec bash "$C/training_contract_cfg_guidance_probe.sh"
