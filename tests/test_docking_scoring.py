"""Unit tests for the docking pose scorer (scripts/docking/common.py).

These are self-contained: they build small molecules with RDKit rather than
reading benchmark data, so they run anywhere.

The scorer was additionally checked against the implementation used for the
original pose-refinement campaign (`rnp_scoring_core.py` on the Roche GitLab
`vanib/*` branches) by re-scoring its published co-folded seed poses for the
Runs N' Poses benchmark: 60/60 agreed to within 0.005 A (median 0.002 A).
That comparison needs the original run outputs and so cannot be a unit test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "docking"))

from common import (  # noqa: E402
    Ideal,
    automorphisms,
    element_graph,
    kabsch_rt,
    mol_graph,
    rmsd,
)

Chem = pytest.importorskip("rdkit.Chem")

BENZOIC_ACID = "OC(=O)c1ccccc1"
PARA_XYLENE = "Cc1ccc(C)cc1"  # two-fold symmetric


def conformer(smiles, seed=0xC0FFEE):
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol, mol.GetConformer().GetPositions()


def random_rigid_motion(xyz, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(3, 3))
    rot, _ = np.linalg.qr(q)
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    return (rot @ xyz.T).T + rng.normal(scale=5.0, size=3)


def test_kabsch_recovers_rigid_motion():
    _, xyz = conformer(BENZOIC_ACID)
    moved = random_rigid_motion(xyz, seed=1)
    rot, t = kabsch_rt(moved, xyz)
    assert rmsd((rot @ moved.T).T + t, xyz) < 1e-6


def test_identical_pose_scores_zero():
    """A ligand scored against itself must give RMSD 0 under the graph matching."""
    mol, xyz = conformer(BENZOIC_ACID)
    graph = mol_graph(mol)
    autos = automorphisms(graph)
    best = min(rmsd(xyz[a], xyz) for a in autos)
    assert best == pytest.approx(0.0, abs=1e-9)


def test_symmetry_equivalent_atoms_give_zero_rmsd():
    """Relabelling a symmetric molecule by one of its automorphisms is not a real deviation."""
    mol, xyz = conformer(PARA_XYLENE)
    graph = mol_graph(mol)
    autos = automorphisms(graph)
    assert len(autos) > 1, "para-xylene should have a non-trivial automorphism"
    nontrivial = next(a for a in autos if not np.array_equal(a, np.arange(len(xyz))))
    permuted = xyz[nontrivial]
    naive = rmsd(permuted, xyz)
    symmetry_aware = min(rmsd(permuted[a], xyz) for a in autos)
    assert symmetry_aware == pytest.approx(0.0, abs=1e-9)
    assert naive > symmetry_aware, "the permutation should look like a deviation without symmetry handling"


def test_element_graph_connectivity_matches_molecule():
    mol, xyz = conformer(BENZOIC_ACID)
    from_coords = element_graph([a.GetSymbol() for a in mol.GetAtoms()], xyz)
    assert from_coords.GetNumBonds() == mol.GetNumBonds()


def test_ideal_bond_ratios_near_one_for_a_clean_conformer():
    """An undistorted conformer must pass the bond-geometry diagnostic."""
    ideal = Ideal(BENZOIC_ACID, n_confs=5)
    ratios = ideal.bond_ratios(ideal.mol.GetConformer().GetPositions())
    assert ratios.min() > 0.9 and ratios.max() < 1.1


def test_ideal_bond_ratios_detect_compression():
    """Shrinking the ligand must show up as short bonds."""
    ideal = Ideal(BENZOIC_ACID, n_confs=5)
    xyz = ideal.mol.GetConformer().GetPositions()
    shrunk = (xyz - xyz.mean(0)) * 0.75 + xyz.mean(0)
    assert ideal.bond_ratios(shrunk).max() < 0.8


def test_template_fit_recovers_a_rigidly_moved_conformer():
    """Idealization of an undistorted, displaced pose returns essentially that pose."""
    ideal = Ideal(BENZOIC_ACID, n_confs=20)
    target = random_rigid_motion(ideal.mol.GetConformer().GetPositions(), seed=2)
    fitted, fit_rmsd = ideal.fit(target)
    assert fit_rmsd < 0.3
    assert rmsd(fitted, target) == pytest.approx(fit_rmsd, abs=1e-9)
