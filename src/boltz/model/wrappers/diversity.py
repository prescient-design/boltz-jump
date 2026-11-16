"""History-dependent biases that push a walk-jump chain to explore.

The bias is an energy in units of kT over the denoised structure x of N atoms,

    U(x) = N sum_k h_k exp(-RMSD(x, c_k)^2 / (2 w^2)),

so h is a height per atom. This makes one setting behave the same for small and
large proteins: the model's log-density, whose gradient is the score, is a sum
over atoms, so a bias of a few kT on the whole structure would be negligible.
Near a centre the bias acts on each atom like a repulsive spring of stiffness
about h / w^2, to be compared with the walk's restoring force of about 1 / sigma^2.

a sum of Gaussian "hills" in RMSD to a set of centres c_k. Its negative gradient
is added to the walk's score, so the chain is pushed away from the centres.

- ``seed``: one fixed hill at the starting structure (repulsion from the seed).
- ``metad``: a hill is deposited at the current structure every
  ``deposit_every`` steps, so the chain is pushed away from everywhere it has
  been. With ``bias_factor`` > 1 this is well-tempered metadynamics: a new hill's
  height is scaled by exp(-U(x) / (bias_factor - 1)), so the bias converges
  instead of growing without bound.

RMSD is computed after optimal superposition over the atoms in ``atom_mask``.
The rotation is treated as a constant when differentiating; this is exact,
because at the optimal rotation the derivative of RMSD with respect to the
rotation is zero.

The gradient is taken with respect to the denoised structure x and applied to
the noisy walk coordinate y as if dx/dy were the identity. At the small noise
levels used for ensembles (sigma = 2 A, where the denoiser's skip weight is
about 0.985) this is a good approximation.
"""

import math
from dataclasses import dataclass, field

import torch


def _aligned_sq_dev(x, c, mask):
    """Mean squared deviation of x from c after superposing c onto x (masked)."""
    w = mask.to(x.dtype)
    n = w.sum()
    xc = (x * w[..., None]).sum(-2, keepdim=True) / n
    cc = (c * w[..., None]).sum(-2, keepdim=True) / n
    x0, c0 = x - xc, c - cc
    with torch.no_grad():
        h = torch.einsum("...ni,...nj->...ij", c0 * w[..., None], x0)
        u, _, vt = torch.linalg.svd(h)
        d = torch.sign(torch.linalg.det(torch.einsum("...ij,...jk->...ik", u, vt)))
        u = u.clone()
        u[..., :, -1] = u[..., :, -1] * d[..., None]
        rot = torch.einsum("...ij,...jk->...ik", u, vt)
    c_aligned = torch.einsum("...ni,...ij->...nj", c0, rot)
    return (((x0 - c_aligned) ** 2).sum(-1) * w).sum(-1) / n


@dataclass
class RMSDHills:
    mode: str  # "seed" or "metad"
    height: float
    width: float
    deposit_every: int = 20
    bias_factor: float = 0.0  # <= 1: plain metadynamics
    centres: list = field(default_factory=list)
    heights: list = field(default_factory=list)
    step: int = 0

    def energy(self, x, mask):
        if not self.centres:
            return x.new_zeros(x.shape[:-2])
        u = x.new_zeros(x.shape[:-2])
        n = mask.sum().to(x.dtype)
        for c, h in zip(self.centres, self.heights):
            u = u + n * h * torch.exp(-_aligned_sq_dev(x, c.to(x.dtype), mask) / (2 * self.width**2))
        return u

    def gradient(self, x, mask):
        """dU/dx for the current structure, in units of kT per Angstrom."""
        if not self.centres:
            return torch.zeros_like(x)
        with torch.enable_grad():
            # at least fp32: the walk may run in reduced precision
            xg = x.detach().to(torch.promote_types(x.dtype, torch.float32)).requires_grad_(True)
            (g,) = torch.autograd.grad(self.energy(xg, mask).sum(), xg)
        return g.to(x.dtype)

    def maybe_deposit(self, x, mask):
        """Record one walk step; deposit a hill at x if it is time to."""
        self.step += 1
        if self.mode != "metad" or self.step % self.deposit_every:
            return
        h = self.height
        if self.bias_factor > 1:
            with torch.no_grad():
                h = h * math.exp(-float(self.energy(x.to(torch.promote_types(x.dtype, torch.float32)), mask).max())
                                 / (self.bias_factor - 1))
        self.centres.append(x.detach().float().clone())
        self.heights.append(h)


def build_bias(walk_jump_args, x_init):
    """Construct the bias described by walk_jump_args, or None."""
    mode = walk_jump_args.get("diversity_mode", "none")
    if mode in (None, "none"):
        return None
    if mode not in ("seed", "metad"):
        raise ValueError(f"unknown diversity_mode {mode!r}")
    bias = RMSDHills(mode=mode, height=walk_jump_args["diversity_height"],
                     width=walk_jump_args["diversity_width"],
                     deposit_every=walk_jump_args.get("diversity_deposit_every", 20),
                     bias_factor=walk_jump_args.get("diversity_bias_factor", 0.0))
    if mode == "seed":
        bias.centres.append(x_init.detach().float().clone())
        bias.heights.append(bias.height)
    return bias
