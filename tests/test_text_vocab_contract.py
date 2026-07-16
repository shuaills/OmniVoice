"""Regression gates for immutable checkpoint text-vocabulary repair."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaModel

import omnivoice.training.builder as builder
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from omnivoice.training.config import TrainingConfig


class _FixedTokenizer:
    def __init__(self, size: int, *, include_required_tokens: bool = True):
        self._size = size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self._vocab = {
            token: index
            for index, token in enumerate(builder._REQUIRED_TEXT_SPECIAL_TOKENS)
        }
        if not include_required_tokens:
            self._vocab.pop(builder._REQUIRED_TEXT_SPECIAL_TOKENS[-1])

    def __len__(self):
        return self._size

    def get_vocab(self):
        return self._vocab


def _llama_config(vocab_size: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def _write_stale_config_checkpoint(path, *, config_rows: int, weight_rows: int):
    model = OmniVoice(
        OmniVoiceConfig(
            audio_vocab_size=4,
            audio_mask_id=3,
            num_audio_codebook=2,
            audio_codebook_weights=[1, 1],
            llm_config=_llama_config(config_rows),
        ),
        llm=LlamaModel(_llama_config(weight_rows)),
    )
    with torch.no_grad():
        weights = model.llm.get_input_embeddings().weight
        weights.copy_(torch.arange(weights.numel()).view_as(weights))
    expected_weights = model.llm.get_input_embeddings().weight.detach().clone()
    model.save_pretrained(path)
    return expected_weights


def test_stale_nested_vocab_is_shimmed_without_resizing_and_survives_reload(
    tmp_path, monkeypatch
):
    broken_checkpoint = tmp_path / "broken"
    expected_weights = _write_stale_config_checkpoint(
        broken_checkpoint, config_rows=23, weight_rows=17
    )
    tokenizer = _FixedTokenizer(17)
    monkeypatch.setattr(
        builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    def _forbid_resize(*args, **kwargs):
        raise AssertionError("checkpoint embeddings must never be resized")

    monkeypatch.setattr(LlamaModel, "resize_token_embeddings", _forbid_resize)

    with (broken_checkpoint / "config.json").open() as f:
        disk_config_before = json.load(f)
    assert disk_config_before["llm_config"]["vocab_size"] == 23

    model, returned_tokenizer = builder.build_model_and_tokenizer(
        TrainingConfig(
            init_from_checkpoint=str(broken_checkpoint),
            attn_implementation="eager",
        )
    )

    assert returned_tokenizer is tokenizer
    builder.assert_model_text_vocab_contract(model, tokenizer)
    assert torch.equal(model.llm.get_input_embeddings().weight, expected_weights)

    # The source artifact is immutable; only the in-memory copy was corrected.
    with (broken_checkpoint / "config.json").open() as f:
        disk_config_after = json.load(f)
    assert disk_config_after == disk_config_before

    repaired_checkpoint = tmp_path / "repaired"
    model.save_pretrained(repaired_checkpoint)
    with (repaired_checkpoint / "config.json").open() as f:
        repaired_config = json.load(f)
    assert repaired_config["llm_config"]["vocab_size"] == 17

    reloaded = OmniVoice.from_pretrained(
        repaired_checkpoint,
        attn_implementation="eager",
        train=True,
    )
    builder.assert_model_text_vocab_contract(reloaded, tokenizer)
    assert torch.equal(reloaded.llm.get_input_embeddings().weight, expected_weights)


def test_preflight_accepts_only_stale_config_mismatch(tmp_path):
    checkpoint = tmp_path / "stale"
    _write_stale_config_checkpoint(checkpoint, config_rows=23, weight_rows=17)

    contract = builder.inspect_checkpoint_text_vocab_contract(
        checkpoint, tokenizer=_FixedTokenizer(17)
    )

    assert contract.tokenizer_size == 17
    assert contract.config_vocab_size == 23
    assert contract.embedding_rows == 17
    assert contract.lm_head_rows is None
    assert contract.requires_config_shim

    with pytest.raises(
        builder.TextVocabContractError, match="tokenizer/embedding mismatch"
    ):
        builder.inspect_checkpoint_text_vocab_contract(
            checkpoint, tokenizer=_FixedTokenizer(16)
        )


def test_matching_checkpoint_does_not_use_config_shim(tmp_path, monkeypatch):
    checkpoint = tmp_path / "matching"
    expected_weights = _write_stale_config_checkpoint(
        checkpoint, config_rows=17, weight_rows=17
    )
    tokenizer = _FixedTokenizer(17)
    monkeypatch.setattr(
        builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    def _forbid_shim(*args, **kwargs):
        raise AssertionError("matching checkpoint must not receive a config shim")

    monkeypatch.setattr(builder, "_checkpoint_config_shim", _forbid_shim)

    model, returned_tokenizer = builder.build_model_and_tokenizer(
        TrainingConfig(
            init_from_checkpoint=str(checkpoint),
            attn_implementation="eager",
        )
    )

    assert returned_tokenizer is tokenizer
    builder.assert_model_text_vocab_contract(model, tokenizer)
    assert torch.equal(model.llm.get_input_embeddings().weight, expected_weights)


def test_checkpoint_load_rejects_hf_loading_anomalies(tmp_path, monkeypatch):
    checkpoint = tmp_path / "matching"
    _write_stale_config_checkpoint(checkpoint, config_rows=17, weight_rows=17)
    tokenizer = _FixedTokenizer(17)
    monkeypatch.setattr(
        builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    real_from_pretrained = OmniVoice.from_pretrained

    def _inject_missing_embedding(*args, **kwargs):
        assert kwargs.get("output_loading_info") is True
        model, loading_info = real_from_pretrained(*args, **kwargs)
        loading_info["missing_keys"] = ["llm.embed_tokens.weight"]
        return model, loading_info

    monkeypatch.setattr(
        OmniVoice, "from_pretrained", staticmethod(_inject_missing_embedding)
    )
    with pytest.raises(
        builder.TextVocabContractError, match="did not load strictly"
    ):
        builder.build_model_and_tokenizer(
            TrainingConfig(
                init_from_checkpoint=str(checkpoint),
                attn_implementation="eager",
            )
        )


def test_fresh_base_init_may_resize_and_synchronizes_configs(monkeypatch):
    tokenizer = _FixedTokenizer(17)
    base_llm = LlamaModel(_llama_config(13))
    monkeypatch.setattr(builder, "_resolve_model_path", lambda path: path)
    monkeypatch.setattr(
        builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        builder.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _llama_config(13),
    )
    monkeypatch.setattr(
        builder.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: base_llm,
    )

    model, returned_tokenizer = builder.build_model_and_tokenizer(
        TrainingConfig(
            llm_name_or_path="unused-test-base",
            attn_implementation="eager",
        )
    )

    assert returned_tokenizer is tokenizer
    builder.assert_model_text_vocab_contract(model, tokenizer)
    assert model.llm.get_input_embeddings().weight.shape[0] == 17
    assert model.config.llm_config.vocab_size == 17
    assert model.llm.config.vocab_size == 17


def test_preflight_rejects_lm_head_row_mismatch(tmp_path):
    checkpoint = tmp_path / "bad-head"
    _write_stale_config_checkpoint(checkpoint, config_rows=17, weight_rows=17)
    save_file(
        {
            "llm.embed_tokens.weight": torch.zeros(17, 8),
            "llm.lm_head.weight": torch.zeros(16, 8),
        },
        checkpoint / "model.safetensors",
    )

    with pytest.raises(
        builder.TextVocabContractError, match="embedding/LM-head mismatch"
    ):
        builder.inspect_checkpoint_text_vocab_contract(
            checkpoint, tokenizer=_FixedTokenizer(17)
        )


def test_checkpoint_missing_special_tokens_is_not_grown(tmp_path):
    checkpoint = tmp_path / "missing-token"
    _write_stale_config_checkpoint(checkpoint, config_rows=17, weight_rows=17)

    with pytest.raises(
        builder.TextVocabContractError, match="missing required special tokens"
    ):
        builder.inspect_checkpoint_text_vocab_contract(
            checkpoint,
            tokenizer=_FixedTokenizer(17, include_required_tokens=False),
        )


def test_preflight_rejects_duplicate_text_tensor_keys_across_shards(tmp_path):
    checkpoint = tmp_path / "duplicate-shards"
    _write_stale_config_checkpoint(checkpoint, config_rows=17, weight_rows=17)
    os.remove(checkpoint / "model.safetensors")
    save_file(
        {
            "llm.embed_tokens.weight": torch.zeros(17, 8),
            "llm.layers.0.weight": torch.zeros(1),
        },
        checkpoint / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "llm.embed_tokens.weight": torch.ones(17, 8),
            "llm.layers.1.weight": torch.ones(1),
        },
        checkpoint / "model-00002-of-00002.safetensors",
    )
    with (checkpoint / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "weight_map": {
                    "llm.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "llm.layers.1.weight": "model-00002-of-00002.safetensors",
                }
            },
            f,
        )

    with pytest.raises(
        builder.TextVocabContractError, match="repeats a text-vocab tensor key"
    ):
        builder.inspect_checkpoint_text_vocab_contract(
            checkpoint, tokenizer=_FixedTokenizer(17)
        )
