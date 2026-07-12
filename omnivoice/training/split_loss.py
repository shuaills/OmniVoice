#!/usr/bin/env python3
"""Distributed split-loss primitives.

The split objective has three independently normalized terms:

* masked acoustic cells, normalized per codebook;
* one EOS event per document;
* void, averaged within each document before averaging across documents.

Counts are collected over a complete gradient-accumulation window before its
first forward pass.  Only counts are reduced at that point; differentiable
numerators remain local and are reduced by DDP through their gradients.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_IGNORE,
    KIND_VOID,
)


@dataclass(frozen=True)
class SplitLossCounts:
    """Fixed-order split-loss counts.

    ``void_count`` is retained for diagnostics and invariant checking.  The
    void loss denominator is ``void_events``, not the number of void cells.
    """

    audio_count: torch.Tensor
    eos_count: torch.Tensor
    void_count: torch.Tensor
    void_events: torch.Tensor
    invariant_errors: torch.Tensor

    def stacked(self) -> torch.Tensor:
        return torch.cat(
            (
                self.audio_count.reshape(-1),
                self.eos_count.reshape(1),
                self.void_count.reshape(-1),
                self.void_events.reshape(1),
                self.invariant_errors.reshape(1),
            )
        )

    @classmethod
    def from_stacked(cls, value: torch.Tensor, num_codebooks: int):
        expected = 2 * num_codebooks + 3
        if value.ndim != 1 or value.numel() != expected:
            raise ValueError(
                f"expected a [{expected}] count vector, got {tuple(value.shape)}"
            )
        return cls(
            audio_count=value[:num_codebooks],
            eos_count=value[num_codebooks],
            void_count=value[num_codebooks + 1 : 2 * num_codebooks + 1],
            void_events=value[-2],
            invariant_errors=value[-1],
        )


@dataclass(frozen=True)
class SplitLossNumerators:
    audio_sum: torch.Tensor
    eos_sum: torch.Tensor
    void_event_sum: torch.Tensor


@dataclass(frozen=True)
class SplitLossWindow:
    batches: list[dict[str, Any]]
    global_counts: SplitLossCounts
    epoch: int


def _batched_loss_kind(
    loss_kind: torch.Tensor, document_ids: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if loss_kind.ndim == 2:
        loss_kind = loss_kind.unsqueeze(0)
    if loss_kind.ndim != 3:
        raise ValueError(
            f"loss_kind must have shape [C,L] or [B,C,L], got {loss_kind.shape}"
        )
    if document_ids is not None:
        if document_ids.ndim == 1:
            document_ids = document_ids.unsqueeze(0)
        if document_ids.ndim != 2:
            raise ValueError(
                f"document_ids must have shape [L] or [B,L], got {document_ids.shape}"
            )
        if document_ids.shape != (loss_kind.shape[0], loss_kind.shape[2]):
            raise ValueError(
                "document_ids shape does not match loss_kind: "
                f"{document_ids.shape} vs {loss_kind.shape}"
            )
    return loss_kind, document_ids


def _void_event_count(
    void_mask: torch.Tensor, document_ids: Optional[torch.Tensor]
) -> torch.Tensor:
    if document_ids is None:
        raise ValueError("document_ids are required when void supervision is present")

    event_count = torch.zeros((), dtype=torch.int64, device=void_mask.device)
    for batch_index in range(void_mask.shape[0]):
        supervised_positions = void_mask[batch_index].any(dim=0)
        doc_ids = torch.unique(document_ids[batch_index, supervised_positions])
        event_count = event_count + (doc_ids >= 0).sum(dtype=torch.int64)
    return event_count


def category_counts(
    loss_kind: torch.Tensor, document_ids: Optional[torch.Tensor] = None
) -> SplitLossCounts:
    """Count split-loss categories without touching RNG or autograd state."""

    loss_kind, document_ids = _batched_loss_kind(loss_kind, document_ids)
    if document_ids is None:
        raise ValueError("split loss requires document_ids")
    invariant_errors = (loss_kind[:, 1:] == KIND_EOS).sum(dtype=torch.int64)
    for batch_index in range(loss_kind.shape[0]):
        supervised_positions = (loss_kind[batch_index] != KIND_IGNORE).any(dim=0)
        invariant_errors = invariant_errors + (
            supervised_positions & (document_ids[batch_index] < 0)
        ).sum(dtype=torch.int64)
        doc_ids = torch.unique(document_ids[batch_index, supervised_positions])
        doc_ids = doc_ids[doc_ids >= 0]
        if doc_ids.numel() == 0:
            continue
        membership = document_ids[batch_index].unsqueeze(0) == doc_ids.unsqueeze(1)
        eos_by_doc = (
            membership & (loss_kind[batch_index, 0] == KIND_EOS).unsqueeze(0)
        ).sum(dim=1)
        invariant_errors = invariant_errors + (eos_by_doc != 1).sum(dtype=torch.int64)
        void_by_doc_codebook = (
            membership.unsqueeze(1) & (loss_kind[batch_index] == KIND_VOID).unsqueeze(0)
        ).sum(dim=2)
        docs_with_void = (void_by_doc_codebook > 0).any(dim=1)
        invariant_errors = invariant_errors + (
            (void_by_doc_codebook == 0) & docs_with_void.unsqueeze(1)
        ).sum(dtype=torch.int64)
    audio_count = (loss_kind == KIND_ACOUSTIC).sum(dim=(0, 2), dtype=torch.int64)
    eos_count = (loss_kind == KIND_EOS).sum(dtype=torch.int64)
    void_mask = loss_kind == KIND_VOID
    void_count = void_mask.sum(dim=(0, 2), dtype=torch.int64)
    void_events = _void_event_count(void_mask, document_ids)
    return SplitLossCounts(
        audio_count=audio_count.detach(),
        eos_count=eos_count.detach(),
        void_count=void_count.detach(),
        void_events=void_events.detach(),
        invariant_errors=invariant_errors.detach(),
    )


def split_loss_numerators(
    per_token_loss: torch.Tensor,
    loss_kind: torch.Tensor,
    document_ids: Optional[torch.Tensor],
    normalized_codebook_weights: torch.Tensor,
) -> SplitLossNumerators:
    """Build differentiable local numerators for the three loss terms."""

    loss_kind, document_ids = _batched_loss_kind(loss_kind, document_ids)
    if per_token_loss.shape != loss_kind.shape:
        raise ValueError(
            "per_token_loss and loss_kind must have identical shapes: "
            f"{per_token_loss.shape} vs {loss_kind.shape}"
        )
    num_codebooks = loss_kind.shape[1]
    weights = normalized_codebook_weights.to(
        device=per_token_loss.device, dtype=per_token_loss.dtype
    ).reshape(-1)
    if weights.numel() != num_codebooks:
        raise ValueError(
            f"expected {num_codebooks} codebook weights, got {weights.numel()}"
        )

    audio_mask = loss_kind == KIND_ACOUSTIC
    eos_mask = loss_kind == KIND_EOS
    void_mask = loss_kind == KIND_VOID
    audio_sum = (per_token_loss * audio_mask).sum(dim=(0, 2))
    eos_sum = (per_token_loss * eos_mask).sum()

    # Graph-connected zero for the globally-zero-void case.
    void_event_sum = per_token_loss.sum() * 0.0
    if document_ids is None:
        raise ValueError("split loss requires document_ids")
    for batch_index in range(loss_kind.shape[0]):
        supervised_positions = void_mask[batch_index].any(dim=0)
        doc_ids = torch.unique(document_ids[batch_index, supervised_positions])
        doc_ids = doc_ids[doc_ids >= 0]
        if doc_ids.numel() == 0:
            continue
        membership = document_ids[batch_index].unsqueeze(0) == doc_ids.unsqueeze(1)
        doc_void = membership.unsqueeze(1) & void_mask[batch_index].unsqueeze(0)
        counts = doc_void.sum(dim=2)
        sums = (per_token_loss[batch_index].unsqueeze(0) * doc_void).sum(dim=2)
        codebook_means = sums / counts.clamp(min=1).to(per_token_loss.dtype)
        void_event_sum = void_event_sum + (codebook_means * weights.unsqueeze(0)).sum()

    return SplitLossNumerators(
        audio_sum=audio_sum,
        eos_sum=eos_sum,
        void_event_sum=void_event_sum,
    )


def backward_scalar(
    numerators: SplitLossNumerators,
    global_counts: SplitLossCounts,
    normalized_codebook_weights: torch.Tensor,
    *,
    gamma: float,
    lambda_eos: float,
    lambda_void: float,
    world_size: int,
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    """Return the W*G-compensated scalar passed to Accelerator.backward."""

    device = numerators.audio_sum.device
    dtype = numerators.audio_sum.dtype
    weights = normalized_codebook_weights.to(device=device, dtype=dtype).reshape(-1)
    audio_count = global_counts.audio_count.to(device=device)
    eos_count = global_counts.eos_count.to(device=device)
    void_events = global_counts.void_events.to(device=device)
    if weights.numel() != numerators.audio_sum.numel():
        raise ValueError("audio numerator and codebook-weight sizes differ")
    if audio_count.numel() != numerators.audio_sum.numel():
        raise ValueError("audio numerator and global-count sizes differ")
    if int(eos_count.item()) == 0:
        raise RuntimeError("global EOS count is zero; split-loss invariant violated")
    if int(global_counts.invariant_errors.item()) != 0:
        raise RuntimeError("global split-loss structural invariant is violated")

    connected_zero = (
        numerators.audio_sum.sum() + numerators.eos_sum + numerators.void_event_sum
    ) * 0.0
    nonzero_audio = audio_count > 0
    safe_audio_count = audio_count.clamp(min=1).to(dtype=dtype)
    audio_term = (
        weights
        * numerators.audio_sum
        / safe_audio_count
        * nonzero_audio.to(dtype=dtype)
    ).sum() + connected_zero
    eos_term = numerators.eos_sum / eos_count.to(dtype=dtype)
    has_void = void_events > 0
    void_term = torch.where(
        has_void,
        numerators.void_event_sum / void_events.clamp(min=1).to(dtype=dtype),
        connected_zero,
    )

    scale = float(world_size * gradient_accumulation_steps) * float(gamma)
    return scale * (
        audio_term + float(lambda_eos) * eos_term + float(lambda_void) * void_term
    )


def objective_from_global_sums(
    numerators: SplitLossNumerators,
    global_counts: SplitLossCounts,
    normalized_codebook_weights: torch.Tensor,
    *,
    gamma: float,
    lambda_eos: float,
    lambda_void: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute detached user-facing objective terms from global window sums."""

    scaled = backward_scalar(
        numerators,
        global_counts,
        normalized_codebook_weights,
        gamma=gamma,
        lambda_eos=lambda_eos,
        lambda_void=lambda_void,
        world_size=1,
        gradient_accumulation_steps=1,
    )
    device = numerators.audio_sum.device
    dtype = numerators.audio_sum.dtype
    weights = normalized_codebook_weights.to(device=device, dtype=dtype)
    counts = global_counts.audio_count.to(device=device)
    zero = scaled * 0.0
    nonzero_audio = counts > 0
    audio = (
        weights
        * numerators.audio_sum
        / counts.clamp(min=1).to(dtype=dtype)
        * nonzero_audio.to(dtype=dtype)
    ).sum() + zero
    eos = numerators.eos_sum / global_counts.eos_count.to(device=device, dtype=dtype)
    void_events = global_counts.void_events.to(device=device)
    void = torch.where(
        void_events > 0,
        numerators.void_event_sum / void_events.clamp(min=1).to(dtype=dtype),
        zero,
    )
    return scaled, audio, eos, void


class WindowCoordinator:
    """Prefetch exact GA windows and reduce their category counts once."""

    def __init__(
        self,
        dataloader: Iterable[dict[str, Any]],
        gradient_accumulation_steps: int,
        *,
        epoch: int = 0,
        collective_device: Optional[torch.device] = None,
    ):
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        self.dataloader = dataloader
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epoch = epoch
        self.collective_device = collective_device
        self._iterator = iter(dataloader)

    def _restart_epoch(self) -> None:
        self.epoch += 1
        dataset = getattr(self.dataloader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)

    def _next_batch(self) -> dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            self._restart_epoch()
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "training dataloader is empty after epoch rollover"
                ) from exc

    def _backend_device(self) -> torch.device:
        if not dist.is_available() or not dist.is_initialized():
            return torch.device("cpu")
        backend = str(dist.get_backend()).lower()
        if "nccl" in backend:
            if self.collective_device is None:
                raise RuntimeError("NCCL count reduction requires a collective device")
            return torch.device(self.collective_device)
        return torch.device("cpu")

    def next_window(self) -> SplitLossWindow:
        batches = [self._next_batch() for _ in range(self.gradient_accumulation_steps)]
        local_stack = None
        num_codebooks = None
        for batch in batches:
            if "loss_kind" not in batch:
                raise KeyError("split-loss batch is missing loss_kind")
            counts = category_counts(batch["loss_kind"], batch.get("document_ids"))
            stacked = counts.stacked().to(dtype=torch.int64, device="cpu")
            if local_stack is None:
                local_stack = torch.zeros_like(stacked)
                num_codebooks = counts.audio_count.numel()
            elif stacked.shape != local_stack.shape:
                raise ValueError("codebook count changed within a GA window")
            local_stack += stacked

        global_stack = local_stack.to(self._backend_device())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_stack, op=dist.ReduceOp.SUM)
        global_stack = global_stack.to("cpu")
        global_counts = SplitLossCounts.from_stacked(global_stack, num_codebooks)
        if int(global_counts.invariant_errors.item()) != 0:
            raise RuntimeError(
                "split-loss structural invariant failed on one or more ranks"
            )
        if int(global_counts.eos_count.item()) == 0:
            raise RuntimeError(
                "global EOS count is zero; split-loss invariant violated"
            )
        return SplitLossWindow(
            batches=batches,
            global_counts=global_counts,
            epoch=self.epoch,
        )
