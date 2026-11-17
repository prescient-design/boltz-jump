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

- ``q_ratchet``: ratchet-and-pawl folding bias (as in ratchet-and-pawl MD) on the
  fraction of native contacts Q of a target structure (the YAML's
  ``final_structure``). It remembers the highest Q reached; while the chain is at
  or above it there is no force, and below it a harmonic restoring force acts,
  U = (k N / 2) (Q_max - Q)^2 for N residues. Q is one number for the whole chain,
  so the bias does not prescribe which contacts form first -- unlike per-residue
  restraints, which pull every residue towards its own target position.

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


@dataclass
class NativeContactRatchet:
    """Ratchet on the fraction of native C-alpha contacts (see module docstring).

    Q(x) = mean_k sigmoid(-beta (r_k - lam r0_k)) over native contacts k: C-alpha
    pairs within ``cutoff`` A in the target and at least ``min_sep`` apart in sequence
    (Best, Hummer & Eaton, PNAS 2013, with C-alpha distances).
    """

    target_ca: torch.Tensor  # [n_res, 3]
    ca_index: torch.Tensor  # [n_res] atom indices of the C-alphas
    k: float
    beta: float = 5.0
    lam: float = 1.5
    cutoff: float = 8.0
    min_sep: int = 4
    q_max: float = float("-inf")

    def __post_init__(self):
        d = torch.cdist(self.target_ca, self.target_ca)
        n = len(self.target_ca)
        i, j = torch.triu_indices(n, n, offset=self.min_sep)
        keep = d[i, j] < self.cutoff
        self.i, self.j, self.r0 = i[keep], j[keep], d[i, j][keep]

    def q(self, x):
        ca = x[..., self.ca_index, :]
        r = torch.linalg.norm(ca[..., self.i, :] - ca[..., self.j, :], dim=-1)
        return torch.sigmoid(-self.beta * (r - self.lam * self.r0.to(x))).mean(-1)

    def energy(self, x, mask=None):
        gap = torch.clamp(self.q_max - self.q(x), min=0.0)
        return 0.5 * self.k * len(self.target_ca) * gap**2

    def gradient(self, x, mask=None):
        if not torch.isfinite(torch.tensor(self.q_max)):
            return torch.zeros_like(x)
        with torch.enable_grad():
            xg = x.detach().to(torch.promote_types(x.dtype, torch.float32)).requires_grad_(True)
            (g,) = torch.autograd.grad(self.energy(xg).sum(), xg)
        return g.to(x.dtype)

    def maybe_deposit(self, x, mask=None):
        """Advance the pawl: remember the highest Q reached."""
        with torch.no_grad():
            self.q_max = max(self.q_max, float(self.q(x.float()).max()))


def build_bias(walk_jump_args, x_init, feats=None):
    """Construct the bias described by walk_jump_args, or None."""
    mode = walk_jump_args.get("diversity_mode", "none")
    if mode in (None, "none"):
        return None
    if mode == "q_ratchet":
        if feats is None or not bool(feats.get("valid_final_structure", torch.tensor(False)).any()):
            raise ValueError("q_ratchet needs a final_structure in the input YAML")
        mask = feats["atom_pad_mask"][0].bool()
        target = torch.zeros(mask.shape[0], 3, device=mask.device)
        target[mask] = feats["final_structure_coords"].reshape(-1, 3).float()
        ca_index = feats["token_to_rep_atom"][0].float().argmax(-1)  # representative atom of each token
        ca_index = ca_index[feats["token_pad_mask"][0].bool()]
        return NativeContactRatchet(target_ca=target[ca_index], ca_index=ca_index,
                                    k=walk_jump_args["diversity_height"])
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
