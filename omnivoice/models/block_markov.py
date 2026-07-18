"""Low-rank revealed-neighbour correction for block audio logits.

The block-diffusion backbone predicts every frame in parallel.  This module
adds a small residual bias from the *revealed* previous audio frame without
turning the block back into autoregressive decoding.  Unknown previous tokens
(``audio_mask_id``) contribute exactly zero.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RevealedNeighborMarkovHead(nn.Module):
    """Project the previous multi-codebook frame into a logit residual.

    The same low-rank state can couple all input and output codebooks.  All
    positions are evaluated by two dense kernels, so inference remains
    parallel over the block.
    """

    def __init__(
        self,
        *,
        num_codebooks: int,
        vocab_size: int,
        mask_id: int,
        rank: int,
    ) -> None:
        super().__init__()
        if num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not 0 <= mask_id < vocab_size:
            raise ValueError(
                f"mask_id must be in [0, {vocab_size}), got {mask_id}"
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)
        self.mask_id = int(mask_id)
        self.rank = int(rank)

        width = self.num_codebooks * self.vocab_size
        self.prev_embeddings = nn.Embedding(width, self.rank)
        self.output = nn.Linear(self.rank, width, bias=False)
        self.register_buffer(
            "codebook_offsets",
            torch.arange(self.num_codebooks) * self.vocab_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prev_embeddings.weight, mean=0.0, std=0.02)
        # Exact baseline at attachment time.  The projection learns first;
        # embeddings receive gradients once the projection leaves zero.
        nn.init.zeros_(self.output.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        logits: torch.Tensor,
        prev_token_ids: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add the transition residual.

        Args:
            logits: ``[B, C, S, V]`` backbone audio logits.
            prev_token_ids: ``[B, C, S]`` previous-frame token ids.  Unknown
                or unavailable neighbours must be represented by ``mask_id``.
            audio_mask: ``[B, S]`` identifying acoustic query positions.
        """
        expected = (
            logits.size(0),
            self.num_codebooks,
            logits.size(2),
            self.vocab_size,
        )
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"logits must have shape {expected}, got {tuple(logits.shape)}"
            )
        if tuple(prev_token_ids.shape) != expected[:-1]:
            raise ValueError(
                "prev_token_ids must have shape "
                f"{expected[:-1]}, got {tuple(prev_token_ids.shape)}"
            )
        if tuple(audio_mask.shape) != (expected[0], expected[2]):
            raise ValueError(
                "audio_mask must have shape "
                f"{(expected[0], expected[2])}, got {tuple(audio_mask.shape)}"
            )
        known = prev_token_ids.ne(self.mask_id)
        shifted = prev_token_ids + self.codebook_offsets.view(1, -1, 1)
        embedded = self.prev_embeddings(shifted.long())
        embedded = embedded * known.unsqueeze(-1).to(embedded.dtype)

        known_count = known.sum(dim=1).clamp(min=1).to(embedded.dtype)
        state = embedded.sum(dim=1) / known_count.sqrt().unsqueeze(-1)
        state = state * audio_mask.unsqueeze(-1).to(state.dtype)

        residual = self.output(state).view(
            expected[0], expected[2], self.num_codebooks, self.vocab_size
        )
        residual = residual.permute(0, 2, 1, 3).to(dtype=logits.dtype)
        # Keep stopping and structural-token behavior outside this experiment.
        # Preserve the acoustic log-partition in fp32 before casting back to
        # the model dtype.  The training loss additionally routes EOS/mask
        # target rows through the untouched backbone logits, so bf16 rounding
        # cannot turn structural supervision into a head-training signal.
        base_acoustic = logits[..., : self.mask_id]
        corrected_acoustic = base_acoustic + residual[..., : self.mask_id]
        partition_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        partition_shift = torch.logsumexp(
            base_acoustic.to(partition_dtype), dim=-1, keepdim=True
        ) - torch.logsumexp(
            corrected_acoustic.to(partition_dtype), dim=-1, keepdim=True
        )
        corrected_acoustic = corrected_acoustic + partition_shift.to(logits.dtype)
        return torch.cat(
            (corrected_acoustic, logits[..., self.mask_id :]),
            dim=-1,
        )


def infer_adjacent_audio_prev_ids(
    input_ids: torch.Tensor,
    audio_mask: torch.Tensor,
    *,
    mask_id: int,
    first_prev_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Infer previous-frame ids for ordinary contiguous inference inputs."""
    if input_ids.ndim != 3:
        raise ValueError(f"input_ids must be [B, C, S], got {tuple(input_ids.shape)}")
    if tuple(audio_mask.shape) != (input_ids.size(0), input_ids.size(2)):
        raise ValueError(
            "audio_mask must match input_ids batch/sequence dimensions, got "
            f"{tuple(audio_mask.shape)} for {tuple(input_ids.shape)}"
        )

    prev = torch.full_like(input_ids, int(mask_id))
    if first_prev_ids is not None:
        expected = (input_ids.size(0), input_ids.size(1))
        if tuple(first_prev_ids.shape) != expected:
            raise ValueError(
                f"first_prev_ids must have shape {expected}, got "
                f"{tuple(first_prev_ids.shape)}"
            )
        if input_ids.size(-1) > 0:
            prev[:, :, 0] = torch.where(
                audio_mask[:, :1],
                first_prev_ids,
                torch.full_like(first_prev_ids, int(mask_id)),
            )
    if input_ids.size(-1) <= 1:
        return prev
    contiguous_audio = audio_mask[:, 1:] & audio_mask[:, :-1]
    prev[:, :, 1:] = torch.where(
        contiguous_audio.unsqueeze(1),
        input_ids[:, :, :-1],
        torch.full_like(input_ids[:, :, :-1], int(mask_id)),
    )
    return prev


__all__ = ["RevealedNeighborMarkovHead", "infer_adjacent_audio_prev_ids"]
