"""Hand-computed tests for the contract-v2 split objective."""

import torch

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_IGNORE,
    KIND_VOID,
)
from omnivoice.training.split_loss import (
    SplitLossCounts,
    backward_scalar,
    category_counts,
    objective_from_global_sums,
    split_loss_numerators,
)


def test_three_terms_and_per_document_void_weighting():
    # Two documents.  The second has two void cells per codebook, so its void
    # contribution must use the within-document mean rather than a cell mean
    # over the whole pack.
    losses = torch.tensor(
        [
            [
                [1.0, 0.0, 4.0, 3.0, 0.0, 8.0, 12.0],
                [10.0, 0.0, 20.0, 30.0, 0.0, 40.0, 60.0],
            ]
        ],
        requires_grad=True,
    )
    kinds = torch.tensor(
        [
            [
                [
                    KIND_ACOUSTIC,
                    KIND_EOS,
                    KIND_VOID,
                    KIND_ACOUSTIC,
                    KIND_EOS,
                    KIND_VOID,
                    KIND_VOID,
                ],
                [
                    KIND_ACOUSTIC,
                    KIND_IGNORE,
                    KIND_VOID,
                    KIND_ACOUSTIC,
                    KIND_IGNORE,
                    KIND_VOID,
                    KIND_VOID,
                ],
            ]
        ],
        dtype=torch.uint8,
    )
    docs = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    weights = torch.tensor([0.75, 0.25])

    counts = category_counts(kinds, docs)
    numerators = split_loss_numerators(losses, kinds, docs, weights)
    assert torch.equal(counts.audio_count, torch.tensor([2, 2]))
    assert counts.eos_count.item() == 2
    assert torch.equal(counts.void_count, torch.tensor([3, 3]))
    assert counts.void_events.item() == 2
    assert torch.equal(numerators.audio_sum, torch.tensor([4.0, 40.0]))
    assert numerators.eos_sum.item() == 0.0  # EOS cell losses were set to zero above.
    # doc0: .75*4 + .25*20 = 8; doc1: .75*10 + .25*50 = 20.
    assert numerators.void_event_sum.item() == 28.0

    # Give the two EOS cells explicit losses and check the complete objective.
    losses_with_eos = losses.detach().clone().requires_grad_(True)
    losses_with_eos.data[0, 0, 1] = 2.0
    losses_with_eos.data[0, 0, 4] = 6.0
    numerators = split_loss_numerators(losses_with_eos, kinds, docs, weights)
    total, audio, eos, void = objective_from_global_sums(
        numerators,
        counts,
        weights,
        gamma=0.5,
        lambda_eos=2.0,
        lambda_void=3.0,
    )
    assert audio.item() == 6.5
    assert eos.item() == 4.0
    assert void.item() == 14.0
    assert total.item() == 0.5 * (6.5 + 2.0 * 4.0 + 3.0 * 14.0)


def test_zero_void_is_graph_connected():
    losses = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]], requires_grad=True)
    kinds = torch.tensor(
        [[[KIND_ACOUSTIC, KIND_EOS], [KIND_IGNORE, KIND_IGNORE]]],
        dtype=torch.uint8,
    )
    docs = torch.tensor([[0, 0]])
    weights = torch.tensor([0.75, 0.25])
    counts = category_counts(kinds, docs)
    numerators = split_loss_numerators(losses, kinds, docs, weights)
    assert counts.void_events.item() == 0
    assert numerators.void_event_sum.requires_grad
    scalar = backward_scalar(
        numerators,
        counts,
        weights,
        gamma=1.0,
        lambda_eos=1.0,
        lambda_void=1.0,
        world_size=1,
        gradient_accumulation_steps=1,
    )
    scalar.backward()
    assert losses.grad is not None
    assert torch.isfinite(losses.grad).all()


def test_single_category_pack_and_zero_audio_codebook():
    losses = torch.tensor([[[2.0, 4.0], [100.0, 200.0]]], requires_grad=True)
    kinds = torch.tensor(
        [[[KIND_ACOUSTIC, KIND_EOS], [KIND_IGNORE, KIND_IGNORE]]],
        dtype=torch.uint8,
    )
    docs = torch.tensor([[0, 0]])
    weights = torch.tensor([0.75, 0.25])
    counts = category_counts(kinds, docs)
    assert torch.equal(counts.audio_count, torch.tensor([1, 0]))
    numerators = split_loss_numerators(losses, kinds, docs, weights)
    scalar = backward_scalar(
        numerators,
        counts,
        weights,
        gamma=1.0,
        lambda_eos=0.5,
        lambda_void=9.0,
        world_size=1,
        gradient_accumulation_steps=1,
    )
    assert scalar.item() == 0.75 * 2.0 + 0.5 * 4.0


def test_zero_eos_fails_fast():
    zero = torch.tensor(0.0, requires_grad=True)
    numerators = split_loss_numerators(
        zero.reshape(1, 1, 1),
        torch.tensor([[[KIND_ACOUSTIC]]], dtype=torch.uint8),
        torch.tensor([[0]]),
        torch.tensor([1.0]),
    )
    counts = SplitLossCounts(
        audio_count=torch.tensor([1]),
        eos_count=torch.tensor(0),
        void_count=torch.tensor([0]),
        void_events=torch.tensor(0),
        invariant_errors=torch.tensor(0),
    )
    try:
        backward_scalar(
            numerators,
            counts,
            torch.tensor([1.0]),
            gamma=1.0,
            lambda_eos=1.0,
            lambda_void=1.0,
            world_size=1,
            gradient_accumulation_steps=1,
        )
    except RuntimeError as exc:
        assert "EOS" in str(exc)
    else:
        raise AssertionError("zero global EOS did not fail fast")
