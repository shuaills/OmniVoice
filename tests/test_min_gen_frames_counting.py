"""Regression test for the min_gen_frames seed_rem miscount (Codex-reported).

Covers the exact failure geometry: seed_total % block_size != 0, second
generated block. Old formula (committed - seed_blocks*bs + jr - n_pre)
counted the seed_rem prompt frames prefilled into gen block 1 as generated
frames from block 2 onward, expiring the EOS floor seed_rem frames early.
"""
import torch
from omnivoice.blockdiff_dual import gen_frame_positions

BS = 32


def _ban(committed_len, seed_total, n_pre, floor):
    gen_pos, jr = gen_frame_positions(committed_len, seed_total, BS, "cpu")
    return (jr >= n_pre) & (gen_pos < floor), gen_pos, jr


def test_aligned_seed_first_block():
    # seed_total=96 aligned: block 1 has no prefill, positions 0..31
    ban, gen_pos, _ = _ban(96, 96, 0, 30)
    assert gen_pos[0] == 0 and gen_pos[31] == 31
    assert ban[:30].all() and not ban[30:].any()


def test_nonaligned_first_block():
    # seed_total=116 -> seed_blocks*bs=96, seed_rem=n_pre=20
    # generated columns jr=20..31 are true positions 0..11
    ban, gen_pos, jr = _ban(96, 116, 20, 30)
    assert gen_pos[20] == 0 and gen_pos[31] == 11
    assert not ban[:20].any()          # prefilled prompt columns never banned
    assert ban[20:].all()              # all 12 true positions < 30


def test_nonaligned_second_block_regression():
    # THE bug geometry: committed=128 (block 1 incl. prefill), true generated=12
    ban, gen_pos, _ = _ban(128, 116, 0, 30)
    assert gen_pos[0] == 12            # old formula said 32 here
    assert ban[:18].all()              # true positions 12..29 still banned
    assert not ban[18:].any()          # 30.. free
    # old-formula reproduction: at jr=2 it computed 34 >= 30 and allowed the
    # T=14 stop observed in the padprobe ban=30 anomaly
    old_gen_pos = 128 - 3 * BS + torch.arange(BS)
    assert old_gen_pos[2] == 34 and gen_pos[2] == 14


def test_no_seed():
    ban, gen_pos, _ = _ban(0, 0, 0, 8)
    assert gen_pos[0] == 0 and ban[:8].all() and not ban[8:].any()


def test_floor_expires_at_true_count():
    # sweep: floor holds until true generated frames reach it, for any rem
    for seed_total in (64, 65, 80, 95):
        n_pre = seed_total % BS
        floor = 40
        committed = (seed_total // BS) * BS
        banned_true_positions = 0
        for blk in range(4):
            npre_b = n_pre if blk == 0 else 0
            ban, gen_pos, jr = _ban(committed, seed_total, npre_b, floor)
            banned_true_positions += int(ban.sum())
            committed += BS
        assert banned_true_positions == floor, seed_total
