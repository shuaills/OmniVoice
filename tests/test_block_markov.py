import json
import tempfile

import torch

from omnivoice.blockdiff_dual import build_block_markov_prev_ids
from omnivoice.models.block_markov import (
    RevealedNeighborMarkovHead,
    infer_adjacent_audio_prev_ids,
)


def _head(rank=3):
    return RevealedNeighborMarkovHead(
        num_codebooks=2,
        vocab_size=7,
        mask_id=5,
        rank=rank,
    )


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


def test_zero_output_attachment_is_exact_and_structural_logits_stay_frozen():
    torch.manual_seed(0)
    head = _head()
    logits = torch.randn(1, 2, 4, 7)
    prev = torch.tensor([[[1, 2, 5, 3], [4, 5, 5, 2]]])
    audio_mask = torch.tensor([[True, True, True, False]])

    attached = head(logits, prev, audio_mask)
    assert torch.equal(attached, logits)

    with torch.no_grad():
        head.output.weight.fill_(0.1)
    corrected = head(logits, prev, audio_mask)
    assert not torch.equal(corrected[:, :, :2, :5], logits[:, :, :2, :5])
    # Both previous codebooks are unknown at frame 2; frame 3 is non-audio.
    assert torch.equal(corrected[:, :, 2:, :], logits[:, :, 2:, :])
    # mask/eos and any later structural classes are outside this experiment.
    assert torch.equal(corrected[..., 5:], logits[..., 5:])
    assert torch.allclose(
        torch.logsumexp(corrected[..., :5], dim=-1),
        torch.logsumexp(logits[..., :5], dim=-1),
        atol=1e-6,
        rtol=0,
    )


def test_bf16_eos_and_void_loss_bypass_acoustic_markov_head():
    torch.manual_seed(2)
    model = _tiny_omnivoice().to(dtype=torch.bfloat16)
    model.enable_block_markov_head(4)
    head = model.block_markov_head
    with torch.no_grad():
        head.output.weight.normal_(std=0.05)
    logits = torch.randn(1, 2, 3, 7, dtype=torch.bfloat16)
    prev = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
    audio_mask = torch.ones(1, 3, dtype=torch.bool)
    corrected = head(logits, prev, audio_mask)
    targets = torch.tensor([[[6, 2, 3], [2, 3, 4]]], dtype=torch.long)
    loss_kind = torch.tensor([[[2, 3, 1], [3, 3, 1]]], dtype=torch.uint8)
    loss_logits = model._audio_logits_for_loss(
        logits,
        corrected,
        targets,
        loss_kind,
    )
    assert torch.equal(loss_logits[:, :, :2], logits[:, :, :2])
    assert torch.equal(loss_logits[:, :, 2:], corrected[:, :, 2:])
    loss = torch.nn.functional.cross_entropy(
        loss_logits.float().permute(0, 3, 1, 2),
        targets,
    )
    loss.backward()
    assert head.output.weight.grad is not None
    assert head.output.weight.grad.abs().max() > 0
    assert head.prev_embeddings.weight.grad is not None
    # The only Markov gradient comes from the content-acoustic last frame.
    assert head.prev_embeddings.weight.grad.abs().max() > 0


def test_markov_training_refuses_missing_loss_kind():
    model = _tiny_omnivoice()
    model.enable_block_markov_head(4)
    logits = torch.randn(1, 2, 2, 7, dtype=torch.float64)
    labels = torch.ones(1, 2, 2, dtype=torch.long)
    try:
        model._audio_logits_for_loss(logits, logits, labels, None)
    except ValueError as error:
        assert "requires loss_kind" in str(error)
    else:
        raise AssertionError("missing loss_kind did not fail closed")


def test_markov_head_learns_after_zero_initialized_attachment():
    torch.manual_seed(1)
    head = _head(rank=4)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.2)
    logits = torch.zeros(2, 2, 3, 7)
    prev = torch.tensor(
        [
            [[1, 2, 3], [2, 3, 4]],
            [[4, 3, 2], [3, 2, 1]],
        ]
    )
    audio_mask = torch.ones(2, 3, dtype=torch.bool)
    target = torch.ones_like(logits[..., :5])

    for _ in range(2):
        optimizer.zero_grad()
        output = head(logits, prev, audio_mask)
        loss = torch.nn.functional.mse_loss(output[..., :5], target)
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()

    assert head.prev_embeddings.weight.grad is not None
    assert torch.isfinite(head.prev_embeddings.weight.grad).all()
    assert head.prev_embeddings.weight.grad.abs().sum() > 0
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_inference_prev_ids_respect_audio_boundary_and_explicit_anchor():
    mask_id = 9
    ids = torch.tensor([[[100, 101, 3, 4], [100, 101, 5, 6]]])
    audio_mask = torch.tensor([[False, False, True, True]])
    prev = infer_adjacent_audio_prev_ids(ids, audio_mask, mask_id=mask_id)
    assert torch.equal(
        prev,
        torch.tensor([[[9, 9, 9, 3], [9, 9, 9, 5]]]),
    )

    block = torch.tensor([[[3, 4], [5, 6]]])
    block_audio = torch.ones(1, 2, dtype=torch.bool)
    anchored = infer_adjacent_audio_prev_ids(
        block,
        block_audio,
        mask_id=mask_id,
        first_prev_ids=torch.tensor([[7, 8]]),
    )
    assert torch.equal(anchored, torch.tensor([[[7, 3], [8, 5]]]))


def test_training_prev_ids_use_corrupted_neighbor_and_committed_anchor():
    mask_id = 99
    clean = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    noisy = torch.tensor(
        [[10, 99, 12, 99, 99, 99], [20, 21, 99, 23, 99, 99]]
    )
    block_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)

    prev = build_block_markov_prev_ids(
        noisy,
        clean,
        block_ids,
        mask_id=mask_id,
    )

    expected = torch.tensor(
        [
            [99, 10, 11, 12, 13, 99],
            [99, 20, 21, 99, 23, 99],
        ]
    )
    assert torch.equal(prev, expected)
    # In particular, frame 3 sees the corrupted frame 2 (mask on cb1), not
    # clean target 22.  This is the no-teacher-forcing/no-leakage contract.
    assert prev[1, 3].item() == mask_id


def test_processor_pack_and_model_keep_documents_isolated():
    import random

    from omnivoice.blockdiff_dual import (
        OmniVoiceBlockDualSampleProcessor,
        build_block_causal_attn_mask,
    )
    from omnivoice.data.collator import PackingDataCollator

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, return_tensors=None):
            del return_tensors

            class Result:
                input_ids = torch.tensor([[1, 2, 3]])

            return Result()

    processor = OmniVoiceBlockDualSampleProcessor(
        text_tokenizer=FakeTokenizer(),
        num_channels=2,
        audio_mask_id=5,
        prompt_ratio_range=(0.0, 0.0),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=1.0,
        language_ratio=0.0,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
        block_size=4,
        block_markov_prev_ids=True,
    )
    samples = []
    for seed in (10, 20):
        random.seed(seed)
        torch.manual_seed(seed)
        samples.append(
            processor(
                {
                    "audio_tokens": torch.randint(0, 5, (2, 5)),
                    "label": {"text": "x", "language_id": "en"},
                }
            )
        )

    total_length = sum(sample["length"] for sample in samples)
    batch = PackingDataCollator(processor, total_length + 2)(samples)
    assert batch["markov_prev_ids"].shape == batch["input_ids"].shape
    expected_prev_ids = torch.nn.functional.pad(
        torch.cat([sample["markov_prev_ids"] for sample in samples], dim=1),
        (0, 2),
        value=5,
    ).unsqueeze(0)
    assert torch.equal(batch["markov_prev_ids"], expected_prev_ids)
    for document_id in (0, 1):
        in_document = batch["document_ids"][0].eq(document_id)
        noisy = in_document & batch["copy_tags"][0].eq(2)
        first_noisy = noisy.nonzero(as_tuple=True)[0][0]
        assert batch["markov_prev_ids"][0, :, first_noisy].eq(5).all()

    model = _tiny_omnivoice()
    model.enable_block_markov_head(4)
    with torch.no_grad():
        model.block_markov_head.output.weight.normal_(std=0.01)
    batch["attention_mask"] = build_block_causal_attn_mask(
        batch["document_ids"][0],
        batch["copy_tags"][0],
        batch["block_ids"][0],
    )
    output = model(**batch)
    assert torch.isfinite(output.loss)
    masked_batch = dict(batch)
    masked_batch["markov_prev_ids"] = torch.full_like(
        batch["markov_prev_ids"], 5
    )
    masked_output = model(**masked_batch)
    known_prev = batch["markov_prev_ids"].ne(5).any(dim=1)
    assert known_prev.any()
    assert not torch.equal(
        output.logits[:, :, known_prev[0], :5],
        masked_output.logits[:, :, known_prev[0], :5],
    )


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


def test_legacy_attach_save_and_strict_reload_round_trip():
    from omnivoice.models.omnivoice import OmniVoice

    legacy = _tiny_omnivoice()
    hidden = torch.randn(1, 3, legacy.config.llm_config.hidden_size).double()
    ids = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
    audio_mask = torch.ones(1, 3, dtype=torch.bool)
    baseline = legacy._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
    )

    legacy.enable_block_markov_head(4)
    attached = legacy._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
    )
    assert torch.equal(attached, baseline)
    with torch.no_grad():
        legacy.block_markov_head.output.weight.normal_(std=0.01)
    expected = legacy._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        legacy.save_pretrained(checkpoint_dir, safe_serialization=False)
        reloaded, loading_info = OmniVoice.from_pretrained(
            checkpoint_dir,
            train=True,
            output_loading_info=True,
            dtype=torch.float64,
        )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert reloaded.config.block_markov_rank == 4
    actual = reloaded._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
    )
    assert torch.equal(actual, expected)


def test_attachment_is_deterministic_and_does_not_advance_global_rng():
    first = _tiny_omnivoice()
    second = _tiny_omnivoice()

    torch.manual_seed(1234)
    expected_random = torch.rand(5)
    torch.manual_seed(1234)
    first.enable_block_markov_head(4, seed=77)
    actual_random = torch.rand(5)
    second.enable_block_markov_head(4, seed=77)

    assert torch.equal(actual_random, expected_random)
    assert torch.equal(
        first.block_markov_head.prev_embeddings.weight,
        second.block_markov_head.prev_embeddings.weight,
    )


def test_builder_strictly_loads_legacy_checkpoint_before_attachment():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
    from omnivoice.training.builder import (
        _REQUIRED_TEXT_SPECIAL_TOKENS,
        build_model_and_tokenizer,
    )
    from omnivoice.training.config import TrainingConfig

    tokens = ["<pad>", "<eos>", "<unk>", "x", *_REQUIRED_TEXT_SPECIAL_TOKENS]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=list(_REQUIRED_TEXT_SPECIAL_TOKENS),
    )
    llm_config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=128,
    )
    legacy_config = OmniVoiceConfig(
        audio_vocab_size=7,
        audio_mask_id=5,
        num_audio_codebook=2,
        audio_codebook_weights=[1, 1],
        llm_config=llm_config,
    )
    legacy_config._attn_implementation = "sdpa"
    torch.manual_seed(99)
    legacy_model = OmniVoice(legacy_config).eval()

    hidden = torch.randn(1, 3, 32)
    ids = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
    audio_mask = torch.ones(1, 3, dtype=torch.bool)
    expected = legacy_model._compute_audio_logits(
        hidden,
        input_ids=ids,
        audio_mask=audio_mask,
    )

    with tempfile.TemporaryDirectory() as root:
        legacy_dir = f"{root}/legacy"
        trained_dir = f"{root}/trained"
        legacy_model.save_pretrained(legacy_dir)
        tokenizer.save_pretrained(legacy_dir)
        config_path = f"{legacy_dir}/config.json"
        with open(config_path) as config_file:
            serialized = json.load(config_file)
        serialized.pop("block_markov_rank", None)
        serialized.pop("block_markov_seed", None)
        with open(config_path, "w") as config_file:
            json.dump(serialized, config_file)
        with open(config_path, "rb") as config_file:
            legacy_config_bytes = config_file.read()

        training_config = TrainingConfig(
            init_from_checkpoint=legacy_dir,
            block_training=True,
            block_scheme="dual",
            block_markov_rank=4,
            seed=77,
            attn_implementation="flex_attention",
            eos_decouple_silence=True,
            split_loss=True,
        )
        attached, _ = build_model_and_tokenizer(training_config)
        with open(config_path, "rb") as config_file:
            assert config_file.read() == legacy_config_bytes
        actual = attached._compute_audio_logits(
            hidden,
            input_ids=ids,
            audio_mask=audio_mask,
        )
        assert torch.equal(actual, expected)

        with torch.no_grad():
            attached.block_markov_head.output.weight.normal_(std=0.01)
        attached.save_pretrained(trained_dir)
        tokenizer.save_pretrained(trained_dir)
        matched, _ = build_model_and_tokenizer(
            TrainingConfig(
                init_from_checkpoint=trained_dir,
                block_training=True,
                block_scheme="dual",
                block_markov_rank=4,
                seed=77,
                attn_implementation="flex_attention",
                eos_decouple_silence=True,
                split_loss=True,
            )
        )
        assert matched.config.block_markov_rank == 4
        for mismatched_rank in (0, 3):
            try:
                build_model_and_tokenizer(
                    TrainingConfig(
                        init_from_checkpoint=trained_dir,
                        block_training=True,
                        block_scheme="dual",
                        block_markov_rank=mismatched_rank,
                        seed=77,
                        attn_implementation=(
                            "flex_attention" if mismatched_rank > 0 else "sdpa"
                        ),
                        eos_decouple_silence=mismatched_rank > 0,
                        split_loss=mismatched_rank > 0,
                    )
                )
            except ValueError as error:
                assert "block_markov_rank does not match checkpoint" in str(
                    error
                )
            else:
                raise AssertionError(
                    "resume silently accepted block_markov_rank="
                    f"{mismatched_rank}"
                )
        reloaded, loading_info = OmniVoice.from_pretrained(
            trained_dir,
            train=True,
            output_loading_info=True,
        )
        assert not loading_info["missing_keys"]
        assert not loading_info["unexpected_keys"]
        assert not loading_info["mismatched_keys"]
        assert not loading_info["error_msgs"]
        for name, tensor in attached.block_markov_head.state_dict().items():
            assert torch.equal(tensor, reloaded.block_markov_head.state_dict()[name])


def test_training_forward_without_explicit_prev_ids_fails_fast():
    model = _tiny_omnivoice()
    model.enable_block_markov_head(4)
    ids = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
    audio_mask = torch.ones(1, 3, dtype=torch.bool)
    labels = ids.clone()
    for packed_metadata in (
        {"labels": labels},
        {"document_ids": torch.zeros(1, 3, dtype=torch.long)},
    ):
        try:
            model(
                input_ids=ids,
                audio_mask=audio_mask,
                **packed_metadata,
            )
        except ValueError as error:
            assert "requires explicit markov_prev_ids" in str(error)
        else:
            raise AssertionError("missing markov_prev_ids did not fail fast")


def test_nonzero_head_preserves_dual_cache_recompute_equivalence():
    from omnivoice.blockdiff_dual import _decode_block_causal
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = _tiny_omnivoice()
    model.enable_block_markov_head(4)
    with torch.no_grad():
        model.block_markov_head.output.weight.normal_(std=0.01)

    gen = OmniVoiceGenerationConfig(
        guidance_scale=2.0,
        class_temperature=0.0,
        position_temperature=0.0,
    )
    prefix = torch.randint(1, 100, (2, 4))
    tokens = {}
    traces = {}
    for use_cache in (True, False):
        trace = []
        output, stats = _decode_block_causal(
            model,
            prefix,
            gen,
            block_size=3,
            max_blocks=2,
            num_step_per_block=2,
            use_kv_cache=use_cache,
            logit_trace=trace,
            min_gen_frames=20,
        )
        assert stats["n_blocks"] == 2
        tokens[use_cache] = output
        traces[use_cache] = trace

    assert torch.equal(tokens[True], tokens[False])
    max_logit_error = _max_trace_error(traces[True], traces[False])
    assert max_logit_error < 1e-9, max_logit_error


def test_partial_seed_shared_and_drop_ref_cache_recompute_equivalence():
    from omnivoice.blockdiff_dual import _decode_block_causal
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = _tiny_omnivoice()
    model.enable_block_markov_head(4)
    with torch.no_grad():
        model.block_markov_head.output.weight.normal_(std=0.01)

    gen = OmniVoiceGenerationConfig(
        guidance_scale=2.0,
        class_temperature=0.0,
        position_temperature=0.0,
    )
    prefix = torch.randint(1, 100, (2, 4))
    # One full seed block plus one partial frame exercises both the committed
    # reference anchor and the generated-history anchor on the next block.
    seed_audio = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    for policy in ("shared", "drop_ref"):
        results = {}
        traces = {}
        for use_cache in (True, False):
            trace = []
            output, stats = _decode_block_causal(
                model,
                prefix,
                gen,
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
            results[use_cache] = output
            traces[use_cache] = trace
        assert torch.equal(results[True], results[False]), policy
        max_logit_error = _max_trace_error(traces[True], traces[False])
        assert max_logit_error < 1e-9, (policy, max_logit_error)
