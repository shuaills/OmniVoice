import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/config/train_config_correct_case_eos_300.json"
RUNNER = ROOT / "correct_case_eos_finetune_300.sh"
EVAL_RUNNER = ROOT / "band_ft10k_eval_pair.sh"
CHECKER_PATH = ROOT / "scripts/check_correct_case_eos_config.py"
SPEC = importlib.util.spec_from_file_location("correct_case_checker", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


def test_config_is_plain_correct_case_eos_finetune():
    config = json.loads(CONFIG.read_text())

    assert config["init_from_checkpoint"] == (
        "/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block"
    )
    assert config["resume_from_checkpoint"] is None
    assert config["steps"] == 300
    assert config["save_steps"] == 300
    assert config["block_training"] is True
    assert config["block_scheme"] == "dual"
    assert config["eos_decouple_silence"] is False
    assert config["split_loss"] is False

    forbidden = (
        "generated_prefix",
        "teacher_kl",
        "terminal_recovery",
        "lambda_eos",
        "lambda_void",
        "reward",
    )
    assert not any(any(token in key for token in forbidden) for key in config)
    assert CHECKER.validate(config) == []


def test_runner_launches_one_training_job_on_ground_truth_data():
    source = RUNNER.read_text()

    assert source.count("accelerate.commands.accelerate_cli launch") == 1
    assert "train_config_correct_case_eos_300.json" in source
    assert "data_config_emilia_full_blockparity.json" in source
    assert "generated_prefix" not in source
    assert "teacher" not in source
    assert "EXPECTED_COMMIT" in source
    assert "git status --porcelain --untracked-files=all" in source
    assert "scripts/check_checkpoint_vocab.py" in source
    assert "scripts/check_correct_case_eos_config.py" in source
    assert "checkpoint-300" in source


def test_checker_rejects_unknown_or_complex_objective_keys():
    config = json.loads(CONFIG.read_text())
    config["generated_prefix_endpoint_training"] = True
    config["misspelled_steps"] = 300

    failures = CHECKER.validate(config)
    assert any("unknown config keys" in failure for failure in failures)
    assert any("forbidden objective keys" in failure for failure in failures)


def test_existing_pair_eval_can_run_at_deployment_guidance():
    source = EVAL_RUNNER.read_text()

    assert "GUIDANCE_SCALE=${GUIDANCE_SCALE:-2.0}" in source
    assert '--guidance-scale "$GUIDANCE_SCALE"' in source
    validation = source.split('python "$REPORTER" validate-generation', 1)[1].split(
        "    PIDS=()", 1
    )[0]
    assert "--is-baseline" not in validation
