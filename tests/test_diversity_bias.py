"""Tests for the walk-jump exploration bias (src/boltz/model/wrappers/diversity.py)."""

import math

import numpy as np
import torch

from boltz.model.wrappers.diversity import RMSDHills, _aligned_sq_dev, build_bias


def random_rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float64)


def kabsch_rmsd_numpy(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def coords(n=30, seed=0):
    return torch.tensor(np.random.default_rng(seed).normal(scale=5.0, size=(n, 3)), dtype=torch.float64)


def test_aligned_deviation_matches_reference_kabsch():
    x, c = coords(seed=1), coords(seed=2)
    mask = torch.ones(len(x), dtype=bool)
    ours = math.sqrt(float(_aligned_sq_dev(x, c, mask)))
    assert ours == np.float64(kabsch_rmsd_numpy(c.numpy(), x.numpy())).item() or \
        abs(ours - kabsch_rmsd_numpy(c.numpy(), x.numpy())) < 1e-8


def test_aligned_deviation_is_rotation_and_translation_invariant():
    x = coords(seed=3)
    moved = x @ random_rotation(4).T + torch.tensor([3.0, -2.0, 7.0], dtype=torch.float64)
    mask = torch.ones(len(x), dtype=bool)
    assert float(_aligned_sq_dev(moved, x, mask)) < 1e-18


def test_mask_excludes_padding():
    x, c = coords(seed=5), coords(seed=6)
    mask = torch.ones(len(x), dtype=bool)
    mask[-5:] = False
    garbage = x.clone()
    garbage[-5:] += 1000.0  # padded atoms must not matter
    assert abs(float(_aligned_sq_dev(garbage, c, mask)) - float(_aligned_sq_dev(x, c, mask))) < 1e-9


def test_seed_repulsion_gradient_points_away_from_seed():
    """Following -dU/dx must increase the RMSD to the seed."""
    seed = coords(seed=7)
    x = seed + 0.3 * coords(seed=8) / 5.0
    mask = torch.ones(len(x), dtype=bool)
    bias = RMSDHills(mode="seed", height=1.0, width=2.0, centres=[seed], heights=[1.0])
    before = float(_aligned_sq_dev(x, seed, mask))
    after = float(_aligned_sq_dev(x - 0.01 * bias.gradient(x, mask), seed, mask))
    assert after > before


def test_gradient_matches_finite_differences():
    """Detaching the rotation is exact at the optimum (envelope theorem)."""
    seed, x = coords(seed=9), coords(seed=10) * 0.2 + coords(seed=9)
    mask = torch.ones(len(x), dtype=bool)
    bias = RMSDHills(mode="seed", height=1.0, width=3.0, centres=[seed], heights=[1.0])
    g = bias.gradient(x, mask)
    eps = 1e-6
    for (i, j) in [(0, 0), (5, 1), (17, 2)]:
        xp, xm = x.clone(), x.clone()
        xp[i, j] += eps
        xm[i, j] -= eps
        fd = (float(bias.energy(xp, mask)) - float(bias.energy(xm, mask))) / (2 * eps)
        assert abs(fd - float(g[i, j])) < 1e-6


def test_metadynamics_deposits_on_schedule():
    bias = RMSDHills(mode="metad", height=1.0, width=2.0, deposit_every=5)
    x, mask = coords(seed=11), torch.ones(30, dtype=bool)
    for _ in range(12):
        bias.maybe_deposit(x, mask)
    assert len(bias.centres) == 2


def test_well_tempered_heights_shrink_where_bias_is_high():
    """Repeated deposits at the same place must get smaller."""
    bias = RMSDHills(mode="metad", height=1.0, width=2.0, deposit_every=1, bias_factor=5.0)
    x, mask = coords(seed=12), torch.ones(30, dtype=bool)
    for _ in range(4):
        bias.maybe_deposit(x, mask)
    assert all(a > b for a, b in zip(bias.heights, bias.heights[1:]))


def test_build_bias_none_and_seed():
    x = coords(seed=13)
    assert build_bias({"diversity_mode": "none"}, x) is None
    b = build_bias({"diversity_mode": "seed", "diversity_height": 2.0, "diversity_width": 1.5}, x)
    assert len(b.centres) == 1 and b.heights == [2.0]
