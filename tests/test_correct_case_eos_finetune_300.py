import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/config/train_config_correct_case_eos_300.json"
RUNNER = ROOT / "correct_case_eos_finetune_300.sh"


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
    assert config["eos_decouple_silence"] is True
    assert config["eos_band_k"] == 1
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
    assert "checkpoint-300" in source
