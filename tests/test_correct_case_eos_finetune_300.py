import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/config/train_config_correct_case_eos_300.json"
CONFIG_600 = ROOT / "examples/config/train_config_correct_case_eos_600.json"
RUNNER = ROOT / "correct_case_eos_finetune_300.sh"
RUNNER_600 = ROOT / "correct_case_eos_finetune_600.sh"
EVAL_RUNNER = ROOT / "band_ft10k_eval_pair.sh"
EVAL_LAUNCHER = ROOT / "correct_case_eos_eval30.sh"
EVAL_300_VS_600 = ROOT / "correct_case_eos_eval30_300_vs_600.sh"
RESUME_MANIFEST = ROOT / "examples/config/ce300_resume_manifest.sha256"
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
    config["force_lr_from_config_on_resume"] = True

    failures = CHECKER.validate(config)
    assert any("unknown config keys" in failure for failure in failures)
    assert any("forbidden objective keys" in failure for failure in failures)
    assert any("force_lr_from_config_on_resume" in failure for failure in failures)


def test_checker_rejects_any_training_semantic_drift():
    config = json.loads(CONFIG_600.read_text())
    mutations = {
        "audio_codebook_weights": [1] * 8,
        "drop_cond_ratio": 0.2,
        "mask_ratio_range": [0.2, 0.8],
        "prompt_ratio_range": [0.1, 0.2],
        "weight_decay": 0.0,
        "batch_tokens": 1024,
        "seed": 7,
        "filter_edge_fillers": True,
    }

    for key, value in mutations.items():
        changed = dict(config)
        changed[key] = value
        assert CHECKER.validate(changed), key


def test_step_600_config_only_continues_the_plain_objective():
    config = json.loads(CONFIG_600.read_text())

    assert config["resume_from_checkpoint"] == (
        "/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/ce300-v1/train/checkpoint-300"
    )
    assert config["init_from_checkpoint"] == (
        "/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/ce300-v1/train/checkpoint-300"
    )
    assert config["force_lr_from_config_on_resume"] is False
    assert config["steps"] == 600
    assert config["save_steps"] == 300
    assert config["eos_decouple_silence"] is False
    assert config["split_loss"] is False
    assert CHECKER.validate(config) == []


def test_step_600_runner_restores_full_training_state():
    source = RUNNER_600.read_text()

    assert source.count("accelerate.commands.accelerate_cli launch") == 1
    assert "train_config_correct_case_eos_600.json" in source
    assert "data_config_emilia_full_blockparity.json" in source
    assert "optimizer.bin scheduler.bin random_states_0.pkl random_states_1.pkl" in source
    assert "ce300_resume_manifest.sha256" in source
    assert "sha256sum --check --strict" in source
    assert "CORRECT_CASE_EOS_FINETUNE_300_PASS" in source
    assert "checkpoint-600" in source
    assert "generated_prefix" not in source
    assert "teacher" not in source

    entries = {
        filename: digest
        for digest, filename in (
            line.split() for line in RESUME_MANIFEST.read_text().splitlines()
        )
    }
    assert set(entries) == {
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "random_states_0.pkl",
        "random_states_1.pkl",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "train_config.json",
        "chat_template.jinja",
    }
    assert all(len(digest) == 64 for digest in entries.values())


def test_existing_pair_eval_can_run_at_deployment_guidance():
    source = EVAL_RUNNER.read_text()

    assert "GUIDANCE_SCALE=${GUIDANCE_SCALE:-2.0}" in source
    assert '--guidance-scale "$GUIDANCE_SCALE"' in source
    validation = source.split('python "$REPORTER" validate-generation', 1)[1].split(
        "    PIDS=()", 1
    )[0]
    assert "--is-baseline" not in validation


def test_eval_launcher_compares_base_and_finetune_on_30_cases_per_language():
    source = EVAL_LAUNCHER.read_text()

    assert "OmniVoice-block" in source
    assert "ce300-v1/train/checkpoint-300" in source
    assert "EXPECTED_COUNT=30" in source
    assert "GUIDANCE_SCALE=1.0" in source
    assert "band_ft10k_eval_pair.sh" in source


def test_continuation_eval_compares_step_300_and_step_600():
    source = EVAL_300_VS_600.read_text()

    assert "ce300-v1/train/checkpoint-300" in source
    assert "ce600-v1/train/checkpoint-600" in source
    assert "EXPECTED_COUNT=30" in source
    assert "GUIDANCE_SCALE=1.0" in source
    assert "band_ft10k_eval_pair.sh" in source
