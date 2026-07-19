import tempfile
from types import SimpleNamespace

import torch


def _tiny_omnivoice():
    from transformers import Qwen3Config

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

    llm_config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
    )
    config = OmniVoiceConfig(
        audio_vocab_size=7,
        audio_mask_id=5,
        num_audio_codebook=2,
        audio_codebook_weights=[1, 1],
        llm_config=llm_config,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(7)
    return OmniVoice(config).double().eval()


def _layout(length=4, boundary=(1, 2)):
    positions = torch.arange(length).view(1, 1, length)
    boundaries = torch.tensor(boundary).view(1, 1, 2)
    return positions, boundaries


def _max_trace_error(cached_trace, recomputed_trace):
    assert len(cached_trace) == len(recomputed_trace) > 0
    max_error = 0.0
    for cached, recomputed in zip(cached_trace, recomputed_trace):
        cached_finite = torch.isfinite(cached)
        recomputed_finite = torch.isfinite(recomputed)
        assert torch.equal(cached_finite, recomputed_finite)
        if cached_finite.any():
            error = (cached[cached_finite] - recomputed[recomputed_finite]).abs()
            max_error = max(max_error, error.max().item())
        assert torch.equal(cached[~cached_finite], recomputed[~recomputed_finite])
    return max_error


def test_anchor_attach_save_and_strict_reload_round_trip():
    from omnivoice.models.omnivoice import OmniVoice

    model = _tiny_omnivoice()
    hidden = torch.randn(1, 4, model.config.llm_config.hidden_size).double()
    ids = torch.tensor([[[1, 5, 2, 5], [2, 5, 3, 5]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, boundaries = _layout()
    baseline = model._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
    )

    model.enable_block_anchor_scan_head(
        6, proposal_dim=3, stride=2, mode="causal", seed=77
    )
    attached = model._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
        anchor_positions=positions,
        anchor_boundary_ids=boundaries,
    )
    assert torch.equal(attached, baseline)
    with torch.no_grad():
        model.block_anchor_scan_head.output.weight.normal_(std=0.01)
    expected = model._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
        anchor_positions=positions,
        anchor_boundary_ids=boundaries,
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir, safe_serialization=False)
        reloaded, loading_info = OmniVoice.from_pretrained(
            checkpoint_dir,
            train=True,
            output_loading_info=True,
            dtype=torch.float64,
        )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert reloaded.config.block_anchor_scan_dim == 6
    actual = reloaded._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
        anchor_positions=positions,
        anchor_boundary_ids=boundaries,
    )
    # The serialized tensors themselves must be bitwise exact.  A repeated
    # CPU einsum/softmax may differ in the final float64 bit across fresh
    # module allocations because of reduction scheduling, so compare the
    # resulting logits at a much tighter tolerance than inference uses.
    for name, tensor in model.block_anchor_scan_head.state_dict().items():
        assert torch.equal(
            tensor,
            reloaded.block_anchor_scan_head.state_dict()[name],
        )
    assert torch.allclose(actual, expected, atol=1e-15, rtol=0)


def test_anchor_training_contract_fails_closed_and_freezes_backbone_mode():
    model = _tiny_omnivoice()
    model.enable_block_anchor_scan_head(
        6, proposal_dim=3, stride=2, mode="causal", seed=77
    )
    ids = torch.tensor([[[1, 5, 2], [2, 5, 3]]])
    audio_mask = torch.ones(1, 3, dtype=torch.bool)
    try:
        model(input_ids=ids, audio_mask=audio_mask, labels=ids.clone())
    except ValueError as error:
        assert "requires explicit block-local" in str(error)
    else:
        raise AssertionError("anchor forward accepted a missing explicit layout")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.block_anchor_scan_head.parameters():
        parameter.requires_grad_(True)
    model._block_anchor_freeze_base = True
    model.train()

    assert model.training
    assert not model.llm.training
    assert not model.audio_embeddings.training
    assert not model.audio_heads.training
    assert model.block_anchor_scan_head.training
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("block_anchor_scan_head.") for name in trainable)


def test_builder_rejects_anchor_contracts_generation_cannot_serve():
    from omnivoice.training.builder import build_model_and_tokenizer
    from omnivoice.training.config import TrainingConfig

    common = dict(
        llm_name_or_path="unused",
        block_training=True,
        block_scheme="dual",
        split_loss=True,
        attn_implementation="flex_attention",
        block_anchor_scan_dim=64,
        block_anchor_freeze_base=True,
    )
    invalid = (
        ({"block_anchor_stride": 1}, "greater than one"),
        ({"block_anchor_stride": 8, "block_size": 64}, "block_size == 32"),
        (
            {"block_anchor_stride": 8, "block_markov_rank": 4},
            "mutually exclusive",
        ),
    )
    for overrides, expected in invalid:
        config = TrainingConfig(**common, **overrides)
        try:
            build_model_and_tokenizer(config)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"builder accepted invalid contract: {overrides}")


def test_optimizer_filtering_is_opt_in_to_anchor_frozen_base():
    from omnivoice.training.trainer import OmniTrainer

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    list(model.parameters())[0].requires_grad_(False)
    config = SimpleNamespace(
        learning_rate=3e-5,
        weight_decay=0.01,
        perf_fused_adamw=False,
        warmup_type="steps",
        warmup_ratio=0.0,
        warmup_steps=0,
        lr_scheduler_type="constant",
        steps=10,
    )
    trainer = object.__new__(OmniTrainer)
    trainer.model = model
    trainer.config = config
    optimizer, _ = trainer.create_optimizer_and_scheduler()
    assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters()))

    model._block_anchor_freeze_base = True
    optimizer, _ = trainer.create_optimizer_and_scheduler()
    assert len(optimizer.param_groups[0]["params"]) == sum(
        parameter.requires_grad for parameter in model.parameters()
    )


def test_nonzero_anchor_head_preserves_cache_recompute_parity_for_cfg_policies():
    from omnivoice.blockdiff_dual import _decode_block_causal
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = _tiny_omnivoice()
    model.enable_block_anchor_scan_head(
        6, proposal_dim=3, stride=2, mode="causal", seed=77
    )
    with torch.no_grad():
        model.block_anchor_scan_head.output.weight.normal_(std=0.01)

    generation = OmniVoiceGenerationConfig(
        guidance_scale=2.0,
        class_temperature=0.0,
        position_temperature=0.0,
    )
    prefix = torch.randint(1, 100, (2, 4))
    seed_audio = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    for policy in ("shared", "drop_ref"):
        outputs = {}
        traces = {}
        for use_cache in (True, False):
            trace = []
            output, stats = _decode_block_causal(
                model,
                prefix,
                generation,
                block_size=3,
                max_blocks=3,
                num_step_per_block=2,
                use_kv_cache=use_cache,
                logit_trace=trace,
                seed_audio=seed_audio,
                min_gen_frames=20,
                cfg_unconditional_seed_policy=policy,
            )
            assert stats["n_blocks"] == 3
            outputs[use_cache] = output
            traces[use_cache] = trace
        assert torch.equal(outputs[True], outputs[False]), policy
        assert _max_trace_error(traces[True], traces[False]) < 1e-9


def test_production_stride_ragged_drop_ref_cache_recompute_parity():
    from omnivoice.blockdiff_dual import _decode_block_causal
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = _tiny_omnivoice()
    model.enable_block_anchor_scan_head(
        6, proposal_dim=3, stride=8, mode="causal", seed=77
    )
    with torch.no_grad():
        model.block_anchor_scan_head.output.weight.normal_(std=0.01)
    generation = OmniVoiceGenerationConfig(
        guidance_scale=2.0,
        class_temperature=0.0,
        position_temperature=0.0,
    )
    prefix = torch.randint(1, 100, (2, 4))
    for ragged_unconditional_length in (1, 4, 7, 8, 9, 31, 32):
        seed_length = (
            32
            if ragged_unconditional_length == 32
            else 64 - ragged_unconditional_length
        )
        seed_audio = (
            torch.arange(2 * seed_length).view(2, seed_length) % 5
        ).long()
        outputs = {}
        traces = {}
        for use_cache in (True, False):
            trace = []
            output, stats = _decode_block_causal(
                model,
                prefix,
                generation,
                block_size=32,
                max_blocks=2,
                num_step_per_block=1,
                use_kv_cache=use_cache,
                logit_trace=trace,
                seed_audio=seed_audio,
                min_gen_frames=1000,
                cfg_unconditional_seed_policy="drop_ref",
            )
            assert stats["n_blocks"] == 2
            outputs[use_cache] = output
            traces[use_cache] = trace
        assert torch.equal(outputs[True], outputs[False]), (
            ragged_unconditional_length
        )
        assert _max_trace_error(traces[True], traces[False]) < 1e-9
