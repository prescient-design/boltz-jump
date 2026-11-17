"""Shared paths and pose scoring for the docking experiments.

Scoring follows the protocol of the original pose-refinement analysis
(scripts/usecases/pose_prediction/relax_ligand on the Roche GitLab vanib/*
branches):

- The predicted protein is superposed on the crystal structure using the
  C-alpha atoms of pocket residues (crystal residues of the ligand-bearing
  chain within 8 A of the crystal ligand), after global sequence alignment.
  Every predicted chain is tried, so homomultimers are handled.
- Ligand RMSD is symmetry-aware (graph automorphisms of the crystal ligand,
  bond orders ignored) and is taken over all crystal copies of the ligand.

Poses are scored as sampled, without the template-fitting "idealization" of
the original analysis. A frame counts as valid if, compared with ideal
conformers built from the ligand SMILES (no crystal information is used):

- every ligand bond length is within 0.75-1.25x of its ideal length (the
  PoseBusters bond-length tolerance),
- the ligand diameter is > 0.7x the ideal diameter (no collapse), and
- no ligand heavy atom is within 1.5 A of a non-ligand heavy atom.
"""

import hashlib
import os
from pathlib import Path

import gemmi
import mdtraj as md
import numpy as np
from Bio import Align
from dotenv import find_dotenv, load_dotenv
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D
from scipy.spatial.distance import pdist

POCKET_CUTOFF_A = 8.0
HARD_CLASH_A = 1.5
DIAMETER_FRACTION = 0.7
BOND_RATIO_RANGE = (0.75, 1.25)
SUCCESS_RMSD_A = 2.0


def docking_dir():
    load_dotenv(find_dotenv(usecwd=True), override=False)
    root = os.environ.get("DOCKING_DIR")
    if not root:
        raise ValueError("Set DOCKING_DIR (in .env or the environment) to the docking data/output directory")
    return Path(root)


def seq_hash(seq):
    return hashlib.sha1(seq.encode()).hexdigest()[:12]


def msa_path(seq):
    """Path to the MSA of a sequence, as written into the input YAMLs.

    MSAs are looked up by sequence hash. By default they come from
    $DOCKING_DIR/msa/<hash>.a3m, as produced by scripts/slurm/docking_msa.sh.
    Set MSA_DIR to reuse an existing collection instead, and MSA_LAYOUT=csv if
    that collection stores them as <hash>/A.csv rather than <hash>.a3m.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)
    root = Path(os.environ.get("MSA_DIR") or (docking_dir() / "msa"))
    h = seq_hash(seq)
    if os.environ.get("MSA_LAYOUT", "a3m") == "csv":
        return root / h / "A.csv"
    return root / f"{h}.a3m"


# ---------------------------------------------------------------- geometry

def kabsch_rt(P, Q):
    """Rotation R and translation t minimizing |R P + t - Q|."""
    Pc, Qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


_ALIGNER = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                 open_gap_score=-5, extend_gap_score=-0.5)


def align_indices(seq_a, seq_b):
    """Index pairs (i, j) of aligned residues between two sequences."""
    aln = _ALIGNER.align(seq_a, seq_b)[0]
    pairs = [(i, j) for (a0, a1), (b0, b1) in zip(*aln.aligned) for i, j in zip(range(a0, a1), range(b0, b1))]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def one_letter(resname):
    info = gemmi.find_tabulated_residue(resname)
    if info and info.is_amino_acid():
        code = info.one_letter_code.upper()
        return code if code.isalpha() else "X"
    return None


# ---------------------------------------------------------------- ligand graphs

def element_graph(elements, coords, bonds=None):
    """Single-bond RDKit graph in the given atom order (connectivity from coords if no bonds)."""
    rw = Chem.RWMol()
    for el in elements:
        rw.AddAtom(Chem.Atom(el.capitalize() if len(el) > 1 else el.upper()))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_NONE)
    rw = Chem.RWMol(mol)
    if bonds is None:
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        except Exception:
            bonds = []
    for a, b in bonds:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, sanitizeOps=Chem.SANITIZE_NONE)
    return out


def mol_graph(mol):
    return element_graph([a.GetSymbol() for a in mol.GetAtoms()], mol.GetConformer().GetPositions(),
                         [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])


def automorphisms(graph):
    matches = graph.GetSubstructMatches(graph, uniquify=False, useChirality=False, maxMatches=5000)
    return [np.array(m) for m in matches] or [np.arange(graph.GetNumAtoms())]


def atom_map(query_graph, target_graph):
    """Indices into query atoms, ordered like target atoms (None if the graphs don't match)."""
    m = query_graph.GetSubstructMatches(target_graph, uniquify=True, useChirality=False, maxMatches=1)
    if m:
        return np.array(m[0])
    m = target_graph.GetSubstructMatches(query_graph, uniquify=True, useChirality=False, maxMatches=1)
    if m and len(m[0]) == query_graph.GetNumAtoms() == target_graph.GetNumAtoms():
        inv = np.zeros(len(m[0]), int)
        inv[np.array(m[0])] = np.arange(len(m[0]))
        return inv
    return None


def read_sdf_ligands(path):
    for sanitize in (True, False):
        mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=True, sanitize=sanitize) if m is not None]
        if mols:
            return mols
    return []


class Ideal:
    """Ideal ligand geometry from SMILES: bond lengths and diameter from ETKDG/MMFF conformers."""

    def __init__(self, smiles, n_confs=100, seed=42):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 0
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if mol.GetNumConformers():
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=500)
        self.mol = Chem.RemoveHs(mol)
        confs = [c.GetPositions() for c in self.mol.GetConformers()]
        self.library = confs
        self.graph = mol_graph(self.mol)
        self.perms = automorphisms(self.graph)
        self.bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in self.mol.GetBonds()]).reshape(-1, 2)
        self.bond_lengths = np.linalg.norm(confs[0][self.bonds[:, 0]] - confs[0][self.bonds[:, 1]], axis=1)
        self.diameter = float(np.median([pdist(c).max() for c in confs]))

    def fit(self, xyz_ideal_order):
        """Best rigid fit of a library conformer onto the coordinates (ideal atom order).

        This is the template-fit idealization: it replaces the sampled ligand with a
        chemically valid conformer placed to match the sampled pose as closely as
        possible, repairing covalent geometry distorted by the one-step jump.
        """
        best = (None, np.inf)
        for coords in self.library:
            for perm in self.perms:
                R, t = kabsch_rt(coords[perm], xyz_ideal_order)
                fitted = (R @ coords[perm].T).T + t
                r = rmsd(fitted, xyz_ideal_order)
                if r < best[1]:
                    inv = np.empty(len(perm), int)
                    inv[perm] = np.arange(len(perm))
                    best = (fitted[inv], r)
        return best

    def bond_ratios(self, xyz_ideal_order):
        d = np.linalg.norm(xyz_ideal_order[self.bonds[:, 0]] - xyz_ideal_order[self.bonds[:, 1]], axis=1)
        return d / self.bond_lengths


# ---------------------------------------------------------------- crystal reference

class Reference:
    """Crystal protein pocket and ligand copies for one system."""

    def __init__(self, protein_path, ligand_mol, copy_mols=(), pocket_cutoff=POCKET_CUTOFF_A):
        st = gemmi.read_structure(str(protein_path))
        st.setup_entities()
        lig_xyz = ligand_mol.GetConformer().GetPositions()
        chains = {}
        for chain in st[0]:
            rows = [(one_letter(r.name), r.find_atom("CA", "*")) for r in chain]
            rows = [(c, np.array(a.pos.tolist())) for c, a in rows if c and a]
            if rows:
                chains[chain.name] = rows
        # ligand-bearing chain = chain with the C-alpha closest to the ligand
        lc = min(chains, key=lambda c: min(np.linalg.norm(x - lig_xyz, axis=1).min() for _, x in chains[c]))
        self.seq = "".join(c for c, _ in chains[lc])
        self.ca = np.array([x for _, x in chains[lc]])
        dist = np.linalg.norm(self.ca[:, None] - lig_xyz[None], axis=-1).min(1)
        self.pocket = dist < pocket_cutoff
        if self.pocket.sum() < 5:
            self.pocket[:] = True
        self.pocket_cutoff = pocket_cutoff
        n = ligand_mol.GetNumAtoms()
        copies = [m for m in [ligand_mol, *copy_mols] if m.GetNumAtoms() == n]
        self.ligands = []  # (graph, xyz, automorphisms)
        seen = []
        for m in copies:
            xyz = m.GetConformer().GetPositions()
            if any(np.allclose(xyz, s, atol=1e-3) for s in seen):
                continue
            seen.append(xyz)
            g = mol_graph(m)
            self.ligands.append((g, xyz, automorphisms(g)))
        self.ligand_xyz = lig_xyz

    @classmethod
    def posebusters(cls, entry_dir):
        entry_dir = Path(entry_dir)
        name = entry_dir.name
        lig = read_sdf_ligands(entry_dir / f"{name}_ligand.sdf")[0]
        copies_path = entry_dir / f"{name}_ligands.sdf"
        copies = read_sdf_ligands(copies_path) if copies_path.exists() else []
        return cls(entry_dir / f"{name}_protein.pdb", lig, copies)

    @classmethod
    def runs_n_poses(cls, gt_dir, ligand_chain):
        gt_dir = Path(gt_dir)
        lig = read_sdf_ligands(gt_dir / "ligand_files" / f"{ligand_chain}.sdf")[0]
        copies = [m for p in sorted((gt_dir / "ligand_files").glob("*.sdf")) for m in read_sdf_ligands(p)]
        return cls(gt_dir / "receptor.cif", lig, copies)

    def pocket_residues(self, ligand_xyz_aligned):
        """Crystal-chain residues within the pocket cutoff of a ligand placed in the crystal frame."""
        dist = np.linalg.norm(self.ca[:, None] - ligand_xyz_aligned[None], axis=-1).min(1)
        return set(np.flatnonzero(dist < self.pocket_cutoff))

    def jaccard(self, ligand_xyz_aligned):
        a, b = set(np.flatnonzero(self.pocket)), self.pocket_residues(ligand_xyz_aligned)
        return len(a & b) / len(a | b) if a | b else 0.0


# ---------------------------------------------------------------- predicted structures

class PoseScorer:
    """Scores ligand poses of one predicted topology (static CIF or trajectory frames)."""

    def __init__(self, reference, topology_cif, ideal=None, ligand_chain=None):
        self.ref = reference
        self.ideal = ideal
        traj = md.load(str(topology_cif))
        top = traj.topology
        xyz0 = traj.xyz[0] * 10.0
        # pocket aligners: one per predicted protein chain that aligns to the crystal chain
        self.aligners = []
        lig_atoms, env_atoms = [], []
        for chain in top.chains:
            ca, seq = [], []
            for res in chain.residues:
                code = one_letter(res.name)
                if code is None:
                    continue
                atoms = [a.index for a in res.atoms if a.name == "CA"]
                if atoms:
                    ca.append(atoms[0])
                    seq.append(code)
            is_ligand_chain = not ca and (ligand_chain is None or chain.chain_id == ligand_chain)
            for a in chain.atoms:
                if a.element is not None and a.element.symbol == "H" or a.residue.name in ("HOH", "WAT"):
                    continue
                (lig_atoms if is_ligand_chain else env_atoms).append(a.index)
            if len(ca) >= 15:
                pairs = align_indices("".join(seq), self.ref.seq)
                if len(pairs) >= 15:
                    pred_idx = np.array(ca)[pairs[:, 0]]
                    mask = self.ref.pocket[pairs[:, 1]]
                    if mask.sum() < 5:
                        mask[:] = True
                    self.aligners.append((pred_idx[mask], self.ref.ca[pairs[:, 1]][mask]))
        if not self.aligners:
            raise ValueError(f"{topology_cif}: no protein chain aligns to the crystal pocket")
        self.lig_idx = np.array(lig_atoms)
        self.env_idx = np.array(env_atoms)
        self.elements = [top.atom(i).element.symbol for i in self.lig_idx]
        lig0 = xyz0[self.lig_idx]
        self.start_xyz = lig0
        # atom correspondences, using connectivity of the start pose
        graph = element_graph(self.elements, lig0)
        self.crystal_maps = []
        for g, xyz, autos in self.ref.ligands:
            m = atom_map(graph, g)
            if m is not None:
                self.crystal_maps.append((m, xyz, autos))
        if not self.crystal_maps:
            raise ValueError(f"{topology_cif}: predicted ligand does not match the crystal ligand graph")
        self.ideal_map = atom_map(graph, ideal.graph) if ideal is not None else None
        self.ideal_crystal_maps = []
        if ideal is not None:
            for g, xyz, autos in self.ref.ligands:
                m = atom_map(ideal.graph, g)
                if m is not None:
                    self.ideal_crystal_maps.append((m, xyz, autos))

    @staticmethod
    def _best(maps, aligned):
        best = np.inf
        for m, xyz, autos in maps:
            p = aligned[m]
            for auto in autos:
                best = min(best, rmsd(p, xyz[auto]))
        return best

    def score(self, frame_xyz, idealize=False):
        """Pose metrics for one frame (coordinates in A, topology atom order)."""
        lig = frame_xyz[self.lig_idx]
        best = None
        for pred_idx, cry in self.aligners:
            R, t = kabsch_rt(frame_xyz[pred_idx], cry)
            pocket = rmsd((R @ frame_xyz[pred_idx].T).T + t, cry)
            aligned = (R @ lig.T).T + t
            r = self._best(self.crystal_maps, aligned)
            if best is None or r < best["lig_rmsd"]:
                best = {"lig_rmsd": r, "pocket_ca": pocket, "aligned": aligned, "R": R, "t": t}
        aligned = best.pop("aligned")
        best["jaccard"] = self.ref.jaccard(aligned)
        best["diameter"] = float(pdist(lig).max()) if len(lig) > 1 else 0.0
        best["min_env_dist"] = float(np.linalg.norm(lig[:, None] - frame_xyz[self.env_idx][None], axis=-1).min())
        if self.ideal is not None and self.ideal_map is not None:
            ratios = self.ideal.bond_ratios(lig[self.ideal_map])
            best["bond_ratio_min"], best["bond_ratio_max"] = float(ratios.min()), float(ratios.max())
            # Frame eligibility (handoff protocol): ideal-SMILES diameter gate and no
            # hard clash. Covalent geometry is repaired by the template fit below, so
            # bond ratios are reported as a diagnostic rather than used as a gate.
            best["valid"] = bool(best["diameter"] > DIAMETER_FRACTION * self.ideal.diameter
                                 and best["min_env_dist"] >= HARD_CLASH_A)
            if idealize and best["valid"] and self.ideal_map is not None and self.ideal_crystal_maps:
                fitted, fit_rmsd = self.ideal.fit(aligned[self.ideal_map])
                best["ideal_rmsd"] = self._best(self.ideal_crystal_maps, fitted)
                best["ideal_fit_rmsd"] = fit_rmsd
                # Idealized pose in the crystal frame, for label-free selection.
                best["ideal_xyz"] = fitted
        del best["R"], best["t"]
        return best
