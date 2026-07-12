"""Gloo reference tests for DDP x gradient-accumulation split loss."""

import contextlib
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_IGNORE,
    KIND_VOID,
)
from omnivoice.training.split_loss import (
    SplitLossCounts,
    SplitLossNumerators,
    WindowCoordinator,
    backward_scalar,
    category_counts,
    split_loss_numerators,
)


FEATURES = 3
VOCAB = 4
WEIGHTS = torch.tensor([0.75, 0.25], dtype=torch.float64)
GAMMA = 0.8
LAMBDA_EOS = 0.3
LAMBDA_VOID = 0.6


class _ReplayLoader:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = self
        self.epoch = 0

    def __iter__(self):
        return iter(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch


def _make_batch(rank, microbatch, num_docs, with_void):
    cells_per_doc = 5
    length = num_docs * cells_per_doc
    kinds = torch.full((1, 2, length), KIND_IGNORE, dtype=torch.uint8)
    targets = torch.zeros((1, 2, length), dtype=torch.long)
    docs = torch.empty((1, length), dtype=torch.long)
    features = torch.empty((1, 2, length, FEATURES), dtype=torch.float64)

    for doc in range(num_docs):
        start = doc * cells_per_doc
        docs[0, start : start + cells_per_doc] = doc
        kinds[0, 0, start] = KIND_ACOUSTIC
        kinds[0, 1, start] = KIND_ACOUSTIC
        if (rank + microbatch + doc) % 2 == 0:
            kinds[0, 0, start + 1] = KIND_ACOUSTIC
        kinds[0, 0, start + 2] = KIND_EOS
        if with_void:
            kinds[0, :, start + 3 : start + 5] = KIND_VOID

        for codebook in range(2):
            for offset in range(cells_per_doc):
                position = start + offset
                base = 1 + rank * 17 + microbatch * 11 + doc * 5 + codebook * 3 + offset
                features[0, codebook, position] = (
                    torch.tensor([base, base % 5 - 2, 1.0], dtype=torch.float64) / 10.0
                )
                targets[0, codebook, position] = base % VOCAB

    return {
        "features": features,
        "targets": targets,
        "loss_kind": kinds,
        "document_ids": docs,
    }


def _case_batches(case, rank, gradient_accumulation_steps):
    batches = []
    for microbatch in range(gradient_accumulation_steps):
        if case == "global_zero_void":
            with_void = False
        elif case == "rank_zero_void":
            with_void = rank != 0
        else:
            with_void = not (rank == 0 and microbatch == 0)
        num_docs = 1 + ((rank + microbatch) % 2)
        batches.append(_make_batch(rank, microbatch, num_docs, with_void))
    return batches


def _losses(model, batch):
    logits = model(batch["features"])
    return F.cross_entropy(
        logits.reshape(-1, VOCAB),
        batch["targets"].reshape(-1),
        reduction="none",
    ).reshape_as(batch["targets"])


def _aggregate_counts(all_batches):
    stacked = None
    for batch in all_batches:
        current = category_counts(batch["loss_kind"], batch["document_ids"]).stacked()
        stacked = current.clone() if stacked is None else stacked + current
    return SplitLossCounts.from_stacked(stacked, num_codebooks=2)


def _reference_gradient(world_size, gradient_accumulation_steps, case):
    torch.manual_seed(20260712)
    model = torch.nn.Linear(FEATURES, VOCAB, bias=False, dtype=torch.float64)
    all_batches = [
        batch
        for rank in range(world_size)
        for batch in _case_batches(case, rank, gradient_accumulation_steps)
    ]
    counts = _aggregate_counts(all_batches)
    audio_sum = None
    eos_sum = None
    void_sum = None
    for batch in all_batches:
        numerators = split_loss_numerators(
            _losses(model, batch),
            batch["loss_kind"],
            batch["document_ids"],
            WEIGHTS,
        )
        audio_sum = (
            numerators.audio_sum
            if audio_sum is None
            else audio_sum + numerators.audio_sum
        )
        eos_sum = (
            numerators.eos_sum if eos_sum is None else eos_sum + numerators.eos_sum
        )
        void_sum = (
            numerators.void_event_sum
            if void_sum is None
            else void_sum + numerators.void_event_sum
        )
    loss = backward_scalar(
        SplitLossNumerators(audio_sum, eos_sum, void_sum),
        counts,
        WEIGHTS,
        gamma=GAMMA,
        lambda_eos=LAMBDA_EOS,
        lambda_void=LAMBDA_VOID,
        world_size=1,
        gradient_accumulation_steps=1,
    )
    loss.backward()
    return model.weight.grad.detach()


def _distributed_worker(
    rank, world_size, gradient_accumulation_steps, case, init_file, result_file
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(20260712)
        model = torch.nn.Linear(FEATURES, VOCAB, bias=False, dtype=torch.float64)
        ddp = DistributedDataParallel(model)
        loader = _ReplayLoader(_case_batches(case, rank, gradient_accumulation_steps))
        window = WindowCoordinator(
            loader,
            gradient_accumulation_steps,
            collective_device=torch.device("cpu"),
        ).next_window()

        for microbatch, batch in enumerate(window.batches):
            sync_context = (
                contextlib.nullcontext()
                if microbatch == gradient_accumulation_steps - 1
                else ddp.no_sync()
            )
            with sync_context:
                numerators = split_loss_numerators(
                    _losses(ddp, batch),
                    batch["loss_kind"],
                    batch["document_ids"],
                    WEIGHTS,
                )
                scalar = backward_scalar(
                    numerators,
                    window.global_counts,
                    WEIGHTS,
                    gamma=GAMMA,
                    lambda_eos=LAMBDA_EOS,
                    lambda_void=LAMBDA_VOID,
                    world_size=world_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                # Accelerator.backward applies this division before DDP's /W.
                (scalar / gradient_accumulation_steps).backward()

        dist.barrier()
        if rank == 0:
            torch.save(model.weight.grad.detach(), result_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,gradient_accumulation_steps,case",
    [
        (1, 1, "ordinary"),
        (2, 1, "rank_zero_void"),
        (2, 1, "global_zero_void"),
        (2, 3, "ordinary"),
    ],
)
def test_gloo_matches_concatenated_reference(
    world_size, gradient_accumulation_steps, case
):
    expected = _reference_gradient(world_size, gradient_accumulation_steps, case)
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "gloo_init")
        result_file = os.path.join(tmpdir, "gradient.pt")
        mp.spawn(
            _distributed_worker,
            args=(
                world_size,
                gradient_accumulation_steps,
                case,
                init_file,
                result_file,
            ),
            nprocs=world_size,
            join=True,
        )
        actual = torch.load(result_file, weights_only=True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_window_coordinator_rolls_epochs_without_short_windows():
    batch = _make_batch(rank=0, microbatch=0, num_docs=1, with_void=True)
    loader = _ReplayLoader([batch])
    window = WindowCoordinator(loader, 3).next_window()
    assert len(window.batches) == 3
    assert window.epoch == 2
    assert loader.epoch == 2


def test_legacy_loss_bit_identity_with_loss_kind_present():
    from transformers import PretrainedConfig

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

    class TinyLLM(torch.nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

        def get_input_embeddings(self):
            return self.embedding

        def set_input_embeddings(self, value):
            self.embedding = value

        def forward(self, inputs_embeds, **kwargs):
            return (inputs_embeds,)

    torch.manual_seed(7)
    llm_config = PretrainedConfig(hidden_size=4, vocab_size=16)
    config = OmniVoiceConfig(
        audio_vocab_size=8,
        audio_mask_id=7,
        num_audio_codebook=2,
        audio_codebook_weights=[3, 1],
        llm_config=llm_config,
    )
    model = OmniVoice(config, llm=TinyLLM(hidden_size=4, vocab_size=16))
    model._split_loss = False
    input_ids = torch.tensor([[[1, 2, 3], [1, 2, 3]]])
    audio_mask = torch.ones((1, 3), dtype=torch.bool)
    labels = torch.tensor([[[1, 2, 3], [3, -100, 4]]])
    loss_kind = torch.tensor(
        [
            [
                [KIND_ACOUSTIC, KIND_EOS, KIND_VOID],
                [KIND_ACOUSTIC, KIND_IGNORE, KIND_VOID],
            ]
        ],
        dtype=torch.uint8,
    )
    absent = model(input_ids=input_ids, audio_mask=audio_mask, labels=labels).loss
    present = model(
        input_ids=input_ids,
        audio_mask=audio_mask,
        labels=labels,
        loss_kind=loss_kind,
    ).loss
    assert torch.equal(absent, present)

    model._split_loss = True
    split = model(
        input_ids=input_ids,
        audio_mask=audio_mask,
        labels=labels,
        loss_kind=loss_kind,
        document_ids=torch.tensor([[0, 0, 0]]),
    )
    assert split.audio_sum.requires_grad
    assert split.eos_sum.requires_grad
    assert split.void_event_sum.requires_grad
    assert torch.equal(split.audio_count, torch.tensor([1, 1]))
    assert split.eos_count.item() == 1
    assert torch.equal(split.void_count, torch.tensor([1, 1]))
    assert split.void_events.item() == 1
    assert split.loss is not None and not split.loss.requires_grad
    assert split.legacy_loss is not None and not split.legacy_loss.requires_grad
