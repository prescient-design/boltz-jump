#!/usr/bin/env python3
"""Build random-coil starting structures for folding runs.

Backbone dihedrals are drawn from broad basins of the Ramachandran plot
(beta / PPII and alpha-right, with a small left-handed share for glycine),
which gives a compact-to-expanded random coil rather than a fully extended,
straight chain. Starts with steric clashes between non-adjacent residues are
rejected and redrawn. Side chains are built by PeptideBuilder in a standard
rotamer.

Usage:
    python scripts/hdx/build_unfolded.py --sequence MQIFV... --n 3 --out_dir starts/ --prefix 1ubq
"""

import argparse
from pathlib import Path

import numpy as np
import PeptideBuilder
from Bio.PDB import PDBIO
from PeptideBuilder import Geometry

BASINS = [  # (phi, psi, weight)
    (-120.0, 130.0, 0.45),  # beta
    (-70.0, 145.0, 0.25),   # polyproline II
    (-65.0, -40.0, 0.25),   # alpha-right
    (60.0, 40.0, 0.05),     # alpha-left
]
SPREAD_DEG = 20.0
CLASH_A = 3.0  # minimum CA-CA distance between residues more than 2 apart


def draw(sequence, rng):
    w = np.array([b[2] for b in BASINS])
    angles = []
    for aa in sequence:
        weights = w.copy()
        if aa != "G":
            weights[3] = 0.0  # only glycine readily adopts alpha-left
        b = BASINS[rng.choice(len(BASINS), p=weights / weights.sum())]
        angles.append((b[0] + rng.normal(0, SPREAD_DEG), b[1] + rng.normal(0, SPREAD_DEG)))
    return angles


def build(sequence, angles):
    structure = PeptideBuilder.initialize_res(Geometry.geometry(sequence[0]))
    for k in range(1, len(sequence)):
        g = Geometry.geometry(sequence[k])
        # PeptideBuilder takes this residue's phi and the *previous* residue's psi.
        g.phi, g.psi_im1 = angles[k][0], angles[k - 1][1]
        PeptideBuilder.add_residue(structure, g)
    return structure


def clashes(structure):
    ca = np.array([r["CA"].coord for r in structure.get_residues()])
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    i, j = np.triu_indices(len(ca), k=3)
    return int((d[i, j] < CLASH_A).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    made = 0
    while made < args.n:
        s = build(args.sequence, draw(args.sequence, rng))
        if clashes(s):
            continue
        ca = np.array([r["CA"].coord for r in s.get_residues()])
        rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
        io = PDBIO()
        io.set_structure(s)
        path = out / f"{args.prefix}_unfolded{made}.pdb"
        io.save(str(path))
        print(f"{path.name}: Rg {rg:.1f} A, end-to-end {np.linalg.norm(ca[0] - ca[-1]):.1f} A")
        made += 1


if __name__ == "__main__":
    main()
