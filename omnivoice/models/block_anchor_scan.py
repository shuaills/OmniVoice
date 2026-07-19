"""Soft anchor proposals with a tiny block-local causal scan.

The block backbone still produces every frame in parallel.  This head reads a
small set of anchor positions, turns either revealed acoustic tokens or a
detached soft backbone proposal into a compact feature, and carries only a
64-dimensional state across the four anchors of a block.  There is no state
stored across forward calls.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


_VALID_MODES = frozenset({"causal", "stateless"})


class SoftAnchorScanHead(nn.Module):
    """Correct acoustic logits with a short, block-internal anchor scan.

    ``causal`` carries the post-anchor state into the next anchor update.
    ``stateless`` uses the same parameters and executes the same recurrence,
    but resets its recurrent input to zero for every anchor.  The latter is a
    diagnostic override for measuring whether state propagation is useful.

    A proposal at anchor ``a`` never corrects anchor ``a`` itself.  It corrects
    offsets ``a + 1`` through ``a + stride - 1``; the next anchor is corrected
    by the previous anchor's post-state.  The first anchor is corrected only by
    the actual committed boundary token supplied for that block.
    """

    def __init__(
        self,
        *,
        num_codebooks: int = 8,
        vocab_size: int = 1026,
        mask_id: int = 1024,
        scan_dim: int = 64,
        proposal_dim: int = 32,
        stride: int = 8,
        mode: str = "causal",
    ) -> None:
        super().__init__()
        if num_codebooks <= 0:
            raise ValueError(
                f"num_codebooks must be positive, got {num_codebooks}"
            )
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not 0 < mask_id < vocab_size:
            raise ValueError(
                f"mask_id must be in [1, {vocab_size}), got {mask_id}"
            )
        if scan_dim <= 0:
            raise ValueError(f"scan_dim must be positive, got {scan_dim}")
        if proposal_dim <= 0:
            raise ValueError(
                f"proposal_dim must be positive, got {proposal_dim}"
            )
        if stride <= 1:
            raise ValueError(f"stride must be greater than one, got {stride}")
        self._validate_mode(mode)

        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)
        self.mask_id = int(mask_id)
        self.scan_dim = int(scan_dim)
        self.proposal_dim = int(proposal_dim)
        self.stride = int(stride)
        self.mode = mode

        # Codebook-specific acoustic embeddings.  Unknown anchors use the
        # expectation under a detached backbone proposal; revealed ids
        # replace that expectation exactly.
        self.proposal_embeddings = nn.Embedding(
            self.num_codebooks * self.mask_id,
            self.proposal_dim,
        )
        self.proposal_projection = nn.Linear(
            self.proposal_dim + 2,
            self.scan_dim,
        )
        self.recurrence = nn.Linear(
            2 * self.scan_dim,
            2 * self.scan_dim,
        )
        self.output = nn.Linear(
            self.scan_dim,
            self.num_codebooks * self.mask_id,
            bias=False,
        )
        # Distance-specific scale: index zero is the preceding state at an
        # anchor (including frame zero's boundary); indices 1..stride-1 are
        # the current anchor's post-state at those within-group offsets.
        self.rho = nn.Parameter(torch.ones(self.stride, self.scan_dim))
        self.register_buffer(
            "codebook_offsets",
            torch.arange(self.num_codebooks) * self.mask_id,
        )

        self.reset_parameters()

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proposal_embeddings.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.proposal_projection.weight)
        nn.init.zeros_(self.proposal_projection.bias)
        nn.init.xavier_uniform_(self.recurrence.weight)
        nn.init.zeros_(self.recurrence.bias)
        nn.init.ones_(self.rho)
        # Exact attachment to the frozen/backbone baseline.
        nn.init.zeros_(self.output.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _embed_revealed(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return codebook embeddings and a known-token mask."""
        known = token_ids.ge(0) & token_ids.lt(self.mask_id)
        safe_ids = token_ids.clamp(min=0, max=self.mask_id - 1).long()
        offsets = self.codebook_offsets.view(
            *((1,) * (token_ids.ndim - 1)), self.num_codebooks
        )
        embedded = self.proposal_embeddings(safe_ids + offsets)
        return embedded, known

    def _revealed_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode an actual boundary frame (padded rows become zero)."""
        embedded, known = self._embed_revealed(token_ids)
        embedded = embedded * known.unsqueeze(-1).to(embedded.dtype)
        pooled = embedded.sum(dim=-2) / math.sqrt(self.num_codebooks)
        known_fraction = known.to(embedded.dtype).mean(dim=-1, keepdim=True)
        entropy = torch.zeros_like(known_fraction)
        return torch.tanh(
            self.proposal_projection(
                torch.cat((pooled, known_fraction, entropy), dim=-1)
            )
        )

    def _proposal_features(
        self,
        token_ids: torch.Tensor,
        acoustic_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Build anchor features without a gradient path into base logits.

        Args:
            token_ids: ``[..., C]`` corrupted/revealed input ids.
            acoustic_logits: ``[..., C, mask_id]`` *head-off* logits.
        """
        expected = (*token_ids.shape, self.mask_id)
        if tuple(acoustic_logits.shape) != expected:
            raise ValueError(
                "acoustic_logits must have shape "
                f"{expected}, got {tuple(acoustic_logits.shape)}"
            )
        if token_ids.size(-1) != self.num_codebooks:
            raise ValueError(
                f"token_ids last dimension must be {self.num_codebooks}, got "
                f"{token_ids.size(-1)}"
            )

        # Rejection/selection decisions from the backbone are observations,
        # not a second training path into the backbone.
        weights = torch.softmax(
            acoustic_logits.detach().to(torch.float32),
            dim=-1,
        )
        embedding_table = self.proposal_embeddings.weight.view(
            self.num_codebooks,
            self.mask_id,
            self.proposal_dim,
        )
        proposed = torch.einsum(
            "...cv,cvd->...cd",
            weights.to(embedding_table.dtype),
            embedding_table,
        )

        revealed, known = self._embed_revealed(token_ids)
        selected = torch.where(known.unsqueeze(-1), revealed, proposed)
        pooled = selected.sum(dim=-2) / math.sqrt(self.num_codebooks)
        known_fraction = known.to(selected.dtype).mean(dim=-1, keepdim=True)
        proposal_entropy = -(
            weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
        ).sum(dim=-1)
        proposal_entropy = torch.where(
            known,
            torch.zeros_like(proposal_entropy),
            proposal_entropy,
        ).mean(dim=-1, keepdim=True) / math.log(max(self.mask_id, 2))
        return torch.tanh(
            self.proposal_projection(
                torch.cat(
                    (
                        pooled,
                        known_fraction,
                        proposal_entropy.to(pooled.dtype),
                    ),
                    dim=-1,
                )
            )
        )

    def _scan_step(
        self,
        previous_state: torch.Tensor,
        anchor_feature: torch.Tensor,
    ) -> torch.Tensor:
        gate_logits, candidate = self.recurrence(
            torch.cat((previous_state, anchor_feature), dim=-1)
        ).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_logits)
        return gate * previous_state + (1.0 - gate) * torch.tanh(candidate)

    def _gather_anchor_inputs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, blocks, anchors = anchor_positions.shape
        flat_positions = anchor_positions.clamp(
            min=0,
            max=input_ids.size(-1) - 1,
        ).reshape(batch, -1).long()
        gathered_ids = input_ids.gather(
            2,
            flat_positions.unsqueeze(1).expand(-1, self.num_codebooks, -1),
        )
        gathered_ids = gathered_ids.permute(0, 2, 1).reshape(
            batch, blocks, anchors, self.num_codebooks
        )

        acoustic = logits[..., : self.mask_id]
        gathered_logits = acoustic.gather(
            2,
            flat_positions[:, None, :, None].expand(
                -1,
                self.num_codebooks,
                -1,
                self.mask_id,
            ),
        )
        gathered_logits = gathered_logits.permute(0, 2, 1, 3).reshape(
            batch,
            blocks,
            anchors,
            self.num_codebooks,
            self.mask_id,
        )
        return gathered_ids, gathered_logits

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_boundary_ids: torch.Tensor,
        *,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply the block-local scan residual.

        Args:
            logits: Base/head-off logits ``[B, C, S, V]``.
            input_ids: Corrupted/revealed model input ``[B, C, S]``.
            audio_mask: Acoustic query mask ``[B, S]``.
            anchor_positions: Dense *full-block* layout ``[B, N, K]`` with
                ``-1`` padding.  The head selects columns ``0::stride`` as
                anchors, so normal 32-frame blocks use offsets 0, 8, 16, 24.
            anchor_boundary_ids: Actual committed boundary frame per block,
                shaped ``[B, N, C]``.
            mode: Optional diagnostic override (``causal`` or ``stateless``).
        """
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B,C,S,V], got {tuple(logits.shape)}")
        batch, codebooks, sequence_length, vocab = logits.shape
        expected_logits = (
            batch,
            self.num_codebooks,
            sequence_length,
            self.vocab_size,
        )
        if tuple(logits.shape) != expected_logits:
            raise ValueError(
                f"logits must have shape {expected_logits}, got {tuple(logits.shape)}"
            )
        if tuple(input_ids.shape) != expected_logits[:-1]:
            raise ValueError(
                f"input_ids must have shape {expected_logits[:-1]}, got "
                f"{tuple(input_ids.shape)}"
            )
        if tuple(audio_mask.shape) != (batch, sequence_length):
            raise ValueError(
                f"audio_mask must have shape {(batch, sequence_length)}, got "
                f"{tuple(audio_mask.shape)}"
            )
        if anchor_positions.ndim != 3:
            raise ValueError(
                "anchor_positions must be [B,N,K], got "
                f"{tuple(anchor_positions.shape)}"
            )
        if anchor_positions.size(0) != batch:
            raise ValueError("anchor_positions batch dimension must match logits")
        blocks = anchor_positions.size(1)
        expected_boundaries = (batch, blocks, codebooks)
        if tuple(anchor_boundary_ids.shape) != expected_boundaries:
            raise ValueError(
                "anchor_boundary_ids must have shape "
                f"{expected_boundaries}, got {tuple(anchor_boundary_ids.shape)}"
            )

        active_mode = self.mode if mode is None else mode
        self._validate_mode(active_mode)

        layout_valid = anchor_positions.ge(0) & anchor_positions.lt(sequence_length)
        scan_positions = anchor_positions[..., :: self.stride]
        anchor_valid = layout_valid[..., :: self.stride]
        block_valid = anchor_valid.any(dim=-1)
        gathered_ids, gathered_logits = self._gather_anchor_inputs(
            logits,
            input_ids,
            scan_positions,
        )
        anchor_features = self._proposal_features(gathered_ids, gathered_logits)
        boundary_features = self._revealed_features(anchor_boundary_ids)

        zeros = torch.zeros_like(boundary_features)
        boundary_known = anchor_boundary_ids.ge(0) & anchor_boundary_ids.lt(
            self.mask_id
        )
        boundary_state = boundary_features * (
            block_valid & boundary_known.any(dim=-1)
        ).unsqueeze(-1).to(boundary_features.dtype)

        pre_states = []
        post_states = []
        carried = boundary_state
        for anchor_index in range(scan_positions.size(2)):
            pre_states.append(carried)
            recurrent_input = carried if active_mode == "causal" else zeros
            updated = self._scan_step(
                recurrent_input,
                anchor_features[:, :, anchor_index],
            )
            valid = anchor_valid[:, :, anchor_index].unsqueeze(-1)
            carried = torch.where(valid, updated, carried)
            post_states.append(carried)
        pre_state = torch.stack(pre_states, dim=2)
        post_state = torch.stack(post_states, dim=2)

        # Keep scan states in the explicit block layout.  Packed sequences can
        # contain long text/reference regions that the head must not project;
        # materializing ``[B,S,D]`` here would make the tiny head scale with
        # the whole packed sequence instead of just the current query blocks.
        state_by_layout = boundary_state.new_zeros(
            batch,
            blocks,
            anchor_positions.size(2),
            self.scan_dim,
        )

        # An anchor position sees only the preceding post-state.  For anchor
        # zero, that preceding state is derived solely from the boundary ids.
        state_by_layout[..., :: self.stride, :] = (
            pre_state
            * self.rho[0]
            * anchor_valid.unsqueeze(-1).to(pre_state.dtype)
        )
        for relative_offset in range(1, self.stride):
            valid = layout_valid[..., relative_offset :: self.stride]
            count = valid.size(-1)
            state_by_layout[..., relative_offset :: self.stride, :] = (
                post_state[..., :count, :]
                * self.rho[relative_offset]
                * valid.unsqueeze(-1).to(post_state.dtype)
            )

        correction_dtype = (
            torch.float64 if logits.dtype == torch.float64 else torch.float32
        )
        safe_layout_positions = anchor_positions.clamp(
            min=0,
            max=sequence_length - 1,
        ).long()
        layout_audio = audio_mask.gather(
            1,
            safe_layout_positions.reshape(batch, -1),
        ).reshape_as(anchor_positions)
        correction_valid = layout_valid & layout_audio

        # This is the only full-logit allocation.  Structural logits and all
        # rows outside the explicit query layout are copied unchanged.  The
        # selected acoustic rows below are the only ``[..., M]`` tensors, so
        # projection/recentering memory is O(valid query frames), not O(S).
        corrected_logits = logits.to(dtype=correction_dtype, copy=True)
        valid_indices = correction_valid.nonzero(as_tuple=False)
        if valid_indices.numel() == 0:
            return corrected_logits

        batch_indices = valid_indices[:, 0]
        block_indices = valid_indices[:, 1]
        layout_indices = valid_indices[:, 2]
        sequence_positions = anchor_positions[
            batch_indices,
            block_indices,
            layout_indices,
        ].long()
        selected_states = state_by_layout[
            batch_indices,
            block_indices,
            layout_indices,
        ]
        residual = self.output(selected_states).reshape(
            -1,
            self.num_codebooks,
            self.mask_id,
        ).to(correction_dtype)

        logits_by_position = logits.permute(0, 2, 1, 3)
        base_acoustic = logits_by_position[
            batch_indices,
            sequence_positions,
            :,
            : self.mask_id,
        ].to(correction_dtype)
        corrected_acoustic = base_acoustic + residual
        corrected_acoustic = corrected_acoustic + (
            torch.logsumexp(base_acoustic, dim=-1, keepdim=True)
            - torch.logsumexp(corrected_acoustic, dim=-1, keepdim=True)
        )

        # Keep corrected rows in fp32.  Casting the recentered logits back to
        # bf16 introduces ~1e-3 log-partition drift.  Advanced assignment on
        # this promoted clone preserves autograd paths into both the scan head
        # and (when unfrozen) the base acoustic logits.
        corrected_by_position = corrected_logits.permute(0, 2, 1, 3)
        corrected_by_position[
            batch_indices,
            sequence_positions,
            :,
            : self.mask_id,
        ] = corrected_acoustic
        return corrected_logits


__all__ = ["SoftAnchorScanHead"]
