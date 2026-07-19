import torch

from omnivoice.models.block_anchor_scan import SoftAnchorScanHead


def _small_head(*, mode="causal", stride=2):
    return SoftAnchorScanHead(
        num_codebooks=2,
        vocab_size=7,
        mask_id=5,
        scan_dim=6,
        proposal_dim=3,
        stride=stride,
        mode=mode,
    )


def _layout(batch=1):
    positions = torch.tensor([[[0, 1, 2, 3]]]).expand(batch, -1, -1).clone()
    boundaries = torch.tensor([[[1, 2]]]).expand(batch, -1, -1).clone()
    return positions, boundaries


def _activate(head, seed=123):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(mean=0.0, std=0.15, generator=generator)
        head.rho.fill_(1.0)


def _dense_reference(
    head,
    logits,
    input_ids,
    audio_mask,
    anchor_positions,
    anchor_boundary_ids,
    *,
    mode=None,
):
    """Reference for the former full-sequence projection implementation."""
    batch, _, sequence_length, _ = logits.shape
    layout_valid = anchor_positions.ge(0) & anchor_positions.lt(sequence_length)
    scan_positions = anchor_positions[..., :: head.stride]
    anchor_valid = layout_valid[..., :: head.stride]
    block_valid = anchor_valid.any(dim=-1)
    gathered_ids, gathered_logits = head._gather_anchor_inputs(
        logits,
        input_ids,
        scan_positions,
    )
    anchor_features = head._proposal_features(gathered_ids, gathered_logits)
    boundary_features = head._revealed_features(anchor_boundary_ids)
    zeros = torch.zeros_like(boundary_features)
    boundary_known = anchor_boundary_ids.ge(0) & anchor_boundary_ids.lt(
        head.mask_id
    )
    boundary_state = boundary_features * (
        block_valid & boundary_known.any(dim=-1)
    ).unsqueeze(-1).to(boundary_features.dtype)

    active_mode = head.mode if mode is None else mode
    pre_states = []
    post_states = []
    carried = boundary_state
    for anchor_index in range(scan_positions.size(2)):
        pre_states.append(carried)
        recurrent_input = carried if active_mode == "causal" else zeros
        updated = head._scan_step(
            recurrent_input,
            anchor_features[:, :, anchor_index],
        )
        valid = anchor_valid[:, :, anchor_index].unsqueeze(-1)
        carried = torch.where(valid, updated, carried)
        post_states.append(carried)
    pre_state = torch.stack(pre_states, dim=2)
    post_state = torch.stack(post_states, dim=2)

    state_by_position = boundary_state.new_zeros(
        batch,
        sequence_length,
        head.scan_dim,
    )

    def scatter_state(positions, states, valid):
        indices = positions.clamp(min=0, max=sequence_length - 1).long()
        source = states * valid.unsqueeze(-1).to(states.dtype)
        state_by_position.scatter_add_(
            1,
            indices.reshape(batch, -1, 1).expand(-1, -1, head.scan_dim),
            source.reshape(batch, -1, head.scan_dim),
        )

    scatter_state(scan_positions, pre_state * head.rho[0], anchor_valid)
    for relative_offset in range(1, head.stride):
        positions = anchor_positions[..., relative_offset :: head.stride]
        valid = layout_valid[..., relative_offset :: head.stride]
        missing = scan_positions.size(-1) - positions.size(-1)
        if missing > 0:
            positions = torch.nn.functional.pad(positions, (0, missing), value=-1)
            valid = torch.nn.functional.pad(valid, (0, missing), value=False)
        scatter_state(
            positions,
            post_state * head.rho[relative_offset],
            valid,
        )

    correction_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    residual = head.output(state_by_position).view(
        batch,
        sequence_length,
        head.num_codebooks,
        head.mask_id,
    )
    residual = residual.permute(0, 2, 1, 3).to(correction_dtype)
    residual = residual * audio_mask[:, None, :, None].to(residual.dtype)
    base_acoustic = logits[..., : head.mask_id].to(correction_dtype)
    corrected_acoustic = base_acoustic + residual
    corrected_acoustic = corrected_acoustic + (
        torch.logsumexp(base_acoustic, dim=-1, keepdim=True)
        - torch.logsumexp(corrected_acoustic, dim=-1, keepdim=True)
    )
    return torch.cat(
        (
            corrected_acoustic,
            logits[..., head.mask_id :].to(correction_dtype),
        ),
        dim=-1,
    )


def test_default_parameter_contract_and_zero_attachment_are_exact():
    default = SoftAnchorScanHead()
    assert default.parameter_count == 805_696

    torch.manual_seed(0)
    head = _small_head()
    logits = torch.randn(2, 2, 4, 7)
    input_ids = torch.tensor(
        [
            [[1, 5, 2, 5], [2, 5, 3, 5]],
            [[4, 5, 1, 5], [3, 5, 2, 5]],
        ]
    )
    audio_mask = torch.ones(2, 4, dtype=torch.bool)
    positions, boundaries = _layout(batch=2)

    output = head(logits, input_ids, audio_mask, positions, boundaries)
    assert output.shape == logits.shape
    assert torch.equal(output, logits)


def test_active_head_preserves_structural_logits_and_acoustic_partition():
    torch.manual_seed(1)
    head = _small_head()
    _activate(head)
    logits = torch.randn(1, 2, 4, 7)
    input_ids = torch.tensor([[[1, 5, 2, 5], [2, 5, 3, 5]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, boundaries = _layout()

    output = head(logits, input_ids, audio_mask, positions, boundaries)
    assert not torch.equal(output[..., :5], logits[..., :5])
    assert torch.equal(output[..., 5:], logits[..., 5:])
    assert torch.allclose(
        torch.logsumexp(output[..., :5], dim=-1),
        torch.logsumexp(logits[..., :5], dim=-1),
        atol=1e-6,
        rtol=0,
    )


def test_sparse_layout_projection_matches_dense_reference():
    torch.manual_seed(13)
    head = _small_head()
    _activate(head, seed=13)
    logits = torch.randn(2, 2, 11, 7)
    input_ids = torch.randint(0, 6, (2, 2, 11))
    audio_mask = torch.tensor(
        [
            [True, True, True, False, False, False, True, True, True, True, False],
            [False, True, True, True, True, False, True, False, False, False, False],
        ]
    )
    positions = torch.tensor(
        [
            [[0, 1, 2, 3], [6, 7, 8, 9]],
            [[1, 2, 3, 4], [6, -1, -1, -1]],
        ]
    )
    boundaries = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[2, 1], [5, 5]],
        ]
    )

    sparse = head(logits, input_ids, audio_mask, positions, boundaries)
    dense = _dense_reference(
        head,
        logits,
        input_ids,
        audio_mask,
        positions,
        boundaries,
    )
    torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=0)


def test_float64_correction_dtype_and_dense_parity_are_preserved():
    torch.manual_seed(16)
    head = _small_head().to(dtype=torch.float64)
    _activate(head, seed=16)
    logits = torch.randn(1, 2, 4, 7, dtype=torch.float64)
    input_ids = torch.tensor([[[5, 1, 5, 2], [5, 3, 5, 4]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, boundaries = _layout()

    sparse = head(logits, input_ids, audio_mask, positions, boundaries)
    dense = _dense_reference(
        head,
        logits,
        input_ids,
        audio_mask,
        positions,
        boundaries,
    )
    assert sparse.dtype == torch.float64
    torch.testing.assert_close(sparse, dense, atol=1e-12, rtol=0)


def test_output_projection_only_receives_valid_audio_layout_rows():
    torch.manual_seed(14)
    head = _small_head()
    _activate(head, seed=14)
    logits = torch.randn(1, 2, 20, 7)
    input_ids = torch.randint(0, 6, (1, 2, 20))
    audio_mask = torch.ones(1, 20, dtype=torch.bool)
    audio_mask[0, 2] = False
    positions = torch.tensor([[[0, 1, 2, -1]]])
    boundaries = torch.tensor([[[1, 2]]])
    projected_shapes = []
    handle = head.output.register_forward_pre_hook(
        lambda _module, args: projected_shapes.append(tuple(args[0].shape))
    )
    try:
        head(logits, input_ids, audio_mask, positions, boundaries)
    finally:
        handle.remove()

    # S=20, but only layout positions 0 and 1 are both valid and acoustic.
    assert projected_shapes == [(2, head.scan_dim)]


def test_sparse_correction_backpropagates_to_head_and_base_logits():
    torch.manual_seed(15)
    head = _small_head()
    _activate(head, seed=15)
    logits = torch.randn(1, 2, 4, 7, requires_grad=True)
    input_ids = torch.full((1, 2, 4), 5, dtype=torch.long)
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, boundaries = _layout()

    output = head(logits, input_ids, audio_mask, positions, boundaries)
    output[..., :5].square().mean().backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    for parameter in (
        head.proposal_embeddings.weight,
        head.proposal_projection.weight,
        head.recurrence.weight,
        head.output.weight,
        head.rho,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_bf16_input_is_recentered_and_returned_in_fp32():
    torch.manual_seed(11)
    head = _small_head().to(dtype=torch.bfloat16)
    _activate(head, seed=11)
    logits = torch.randn(1, 2, 4, 7, dtype=torch.bfloat16)
    input_ids = torch.tensor([[[5, 5, 2, 5], [5, 5, 3, 5]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, boundaries = _layout()

    output = head(logits, input_ids, audio_mask, positions, boundaries)
    assert output.dtype == torch.float32
    assert torch.equal(output[..., 5:], logits[..., 5:])
    partition_error = (
        torch.logsumexp(output[..., :5], dim=-1)
        - torch.logsumexp(logits[..., :5].float(), dim=-1)
    ).abs().max()
    assert partition_error <= 5e-6


def test_missing_boundary_is_strict_zero_even_with_active_projection_bias():
    torch.manual_seed(11)
    head = _small_head()
    _activate(head, seed=11)
    logits = torch.randn(1, 2, 4, 7)
    input_ids = torch.tensor([[[1, 5, 2, 5], [2, 5, 3, 5]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, _ = _layout()
    missing_boundary = torch.full((1, 1, 2), 5, dtype=torch.long)

    output = head(
        logits,
        input_ids,
        audio_mask,
        positions,
        missing_boundary,
    )
    assert torch.equal(output[:, :, 0], logits[:, :, 0])
    assert not torch.equal(output[:, :, 1], logits[:, :, 1])


def test_revealed_tokens_override_soft_proposals_and_unknowns_use_them():
    torch.manual_seed(2)
    head = _small_head()
    revealed = torch.tensor([[[1, 3]]])
    first_logits = torch.randn(1, 1, 2, 5)
    second_logits = torch.randn(1, 1, 2, 5) * 9.0

    first = head._proposal_features(revealed, first_logits)
    second = head._proposal_features(revealed, second_logits)
    assert torch.equal(first, second)

    partly_unknown = torch.tensor([[[5, 3]]])
    first = head._proposal_features(partly_unknown, first_logits)
    second = head._proposal_features(partly_unknown, second_logits)
    assert not torch.equal(first, second)


def test_unknown_proposal_uses_full_acoustic_distribution():
    torch.manual_seed(12)
    head = SoftAnchorScanHead(
        num_codebooks=1,
        vocab_size=7,
        mask_id=5,
        scan_dim=4,
        proposal_dim=3,
        stride=2,
    )
    token_ids = torch.tensor([[[5]]])
    first_logits = torch.tensor([[[[4.0, 3.0, 2.0, 1.0, 0.0]]]])
    second_logits = first_logits.clone()
    # Class four remains outside a top-4 set in both cases.  A full soft
    # expectation still observes its changed probability mass.
    second_logits[..., 4] = 0.5
    first = head._proposal_features(token_ids, first_logits)
    second = head._proposal_features(token_ids, second_logits)
    assert not torch.equal(first, second)


def test_soft_proposal_is_stop_gradient_but_embedding_still_learns():
    torch.manual_seed(3)
    head = _small_head()
    token_ids = torch.full((1, 2, 2), 5, dtype=torch.long)
    acoustic_logits = torch.randn(1, 2, 2, 5, requires_grad=True)

    features = head._proposal_features(token_ids, acoustic_logits)
    features.square().sum().backward()

    assert acoustic_logits.grad is None
    assert head.proposal_embeddings.weight.grad is not None
    assert head.proposal_embeddings.weight.grad.abs().sum() > 0


def test_anchor_never_affects_itself_and_stateless_breaks_long_dependency():
    torch.manual_seed(4)
    head = SoftAnchorScanHead(
        num_codebooks=1,
        vocab_size=7,
        mask_id=5,
        scan_dim=5,
        proposal_dim=3,
        stride=2,
    )
    _activate(head, seed=4)
    logits = torch.randn(1, 1, 4, 7)
    first_ids = torch.tensor([[[1, 5, 3, 5]]])
    changed_first_anchor = torch.tensor([[[2, 5, 3, 5]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.tensor([[[0, 1, 2, 3]]])
    boundaries = torch.tensor([[[4]]])

    causal_first = head(
        logits, first_ids, audio_mask, positions, boundaries, mode="causal"
    )
    causal_changed = head(
        logits,
        changed_first_anchor,
        audio_mask,
        positions,
        boundaries,
        mode="causal",
    )
    # Frame zero is seeded only by the committed boundary.  The first anchor
    # changes the following group, including the next anchor.
    assert torch.equal(causal_first[:, :, 0], causal_changed[:, :, 0])
    assert not torch.equal(causal_first[:, :, 1], causal_changed[:, :, 1])
    assert not torch.equal(causal_first[:, :, 2], causal_changed[:, :, 2])
    # At frame three, causal q_1 still contains q_0.  Stateless q_1 is updated
    # from zero, so changing q_0 can no longer affect that frame.
    assert not torch.equal(causal_first[:, :, 3], causal_changed[:, :, 3])

    stateless_first = head(
        logits, first_ids, audio_mask, positions, boundaries, mode="stateless"
    )
    stateless_changed = head(
        logits,
        changed_first_anchor,
        audio_mask,
        positions,
        boundaries,
        mode="stateless",
    )
    assert torch.equal(stateless_first[:, :, 3], stateless_changed[:, :, 3])


def test_non_anchor_input_tokens_cannot_leak_into_the_scan():
    torch.manual_seed(5)
    head = _small_head()
    _activate(head, seed=5)
    logits = torch.randn(1, 2, 4, 7)
    first_ids = torch.tensor([[[5, 1, 2, 3], [5, 4, 3, 1]]])
    changed_non_anchors = torch.tensor([[[5, 4, 2, 0], [5, 1, 3, 2]]])
    audio_mask = torch.ones(1, 4, dtype=torch.bool)
    positions, boundaries = _layout()

    first = head(logits, first_ids, audio_mask, positions, boundaries)
    changed = head(
        logits,
        changed_non_anchors,
        audio_mask,
        positions,
        boundaries,
    )
    assert torch.equal(first, changed)


def test_ragged_layout_respects_audio_mask_and_block_boundaries():
    torch.manual_seed(6)
    head = _small_head()
    _activate(head, seed=6)
    logits = torch.randn(2, 2, 9, 7)
    input_ids = torch.randint(0, 5, (2, 2, 9))
    input_ids[:, :, 1::2] = 5
    audio_mask = torch.tensor(
        [
            [True, True, True, True, False, True, True, True, True],
            [False, True, True, True, True, True, True, False, False],
        ]
    )
    positions = torch.tensor(
        [
            [
                [0, 1, 2, 3, -1, -1],
                [5, 6, 7, 8, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            [
                [1, 2, 3, 4, 5, 6],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
        ]
    )
    boundaries = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 5]],
            [[2, 1], [5, 5], [5, 5]],
        ]
    )

    output = head(logits, input_ids, audio_mask, positions, boundaries)
    non_audio = ~audio_mask
    assert torch.equal(
        output.permute(0, 2, 1, 3)[non_audio],
        logits.permute(0, 2, 1, 3)[non_audio],
    )
    assert torch.isfinite(output).all()

    changed_boundaries = boundaries.clone()
    changed_boundaries[0, 0] = torch.tensor([4, 0])
    changed = head(
        logits,
        input_ids,
        audio_mask,
        positions,
        changed_boundaries,
    )
    # A boundary change in block zero is clamped before block one's start.
    assert torch.equal(output[0, :, 5:], changed[0, :, 5:])
