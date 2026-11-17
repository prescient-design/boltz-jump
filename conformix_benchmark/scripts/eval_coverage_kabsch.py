#!/usr/bin/env python3
"""ConforMix-style coverage with Kabsch C-alpha RMSD.

For each protein, the closest sample to each reference state is found by
C-alpha RMSD after optimal superposition (Kabsch), over the residues present in
the reference. Residues are matched to the target sequence by global sequence
alignment, so reference numbering and missing residues do not matter.
Coverage at X is the fraction of proteins whose closest sample is within X times
the reference-to-reference RMSD, for the best-matched and the alternate state.

--check_refs recomputes the reference-to-reference RMSD with this definition and
compares it to the RMSD column shipped with the ConforMix datasets.

Usage:
    python scripts/eval_coverage_kabsch.py --dataset domainmotion --check_refs
    python scripts/eval_coverage_kabsch.py --dataset domainmotion \
        --predictions_dir ../results/conformix_benchmark_1000/domainmotion \
        --prefix "boltz_jump_results_s=2.0_d=1.0_f=0.1_" --out_csv eval/domainmotion.csv
"""

import argparse
import glob
from pathlib import Path

import gemmi
import mdtraj as md
import numpy as np
import pandas as pd
from Bio import Align

HERE = Path(__file__).resolve().parent.parent


def load_dataset(dataset):
    df = pd.read_csv(HERE / f"{dataset}.csv").rename(columns={"pdbid_if": "pdb1", "pdbid_of": "pdb2", "test_case": "system_id"})
    seqs = pd.read_csv(HERE / f"{dataset}_sequences.csv").rename(columns={"test_case": "system_id"})
    return df.merge(seqs[["system_id", "sequence"]], on="system_id")


def chain_ca(cif_path, chain_id):
    """One-letter sequence and C-alpha coordinates (Angstrom) of a polymer chain."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    names = [c.name for c in st[0]]
    if chain_id in names:
        residues = st[0][chain_id].get_polymer()
    else:
        # Some dataset entries use the mmCIF label_asym_id rather than the author chain.
        residues = st[0].get_subchain(chain_id)
        if len(residues) == 0:
            raise ValueError(f"Chain {chain_id} not found (author chains {names})")
    seq, xyz = [], []
    for res in residues:
        ca = res.find_atom("CA", "*")
        if ca is None:
            continue
        seq.append(gemmi.find_tabulated_residue(res.name).one_letter_code.upper() if gemmi.find_tabulated_residue(res.name) else "X")
        xyz.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return "".join(seq), np.array(xyz)


_ALIGNER = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                 open_gap_score=-5, extend_gap_score=-0.5)


def map_to_target(chain_seq, target_seq):
    """Indices (chain_idx, target_idx) of aligned, identical residues."""
    aln = _ALIGNER.align(chain_seq, target_seq)[0]
    pairs = []
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for i, j in zip(range(a0, a1), range(b0, b1)):
            if chain_seq[i] == target_seq[j]:
                pairs.append((i, j))
    return np.array(pairs)


def kabsch_rmsd(P, Q):
    """RMSD between each P[k] (k, n, 3) and Q (n, 3) after optimal superposition."""
    P = P - P.mean(1, keepdims=True)
    Q = Q - Q.mean(0)
    H = np.einsum("kni,nj->kij", P, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("kij,kjl->kil", U, Vt)))
    S[:, -1] *= d
    e = (P**2).sum((1, 2)) + (Q**2).sum() - 2 * S.sum(1)
    return np.sqrt(np.maximum(e, 0) / Q.shape[0])


def reference(code_chain, target_seq, ref_dir):
    code, chain = code_chain.split("_")
    seq, xyz = chain_ca(Path(ref_dir) / f"{code.upper()}.cif", chain)
    pairs = map_to_target(seq, target_seq)
    coords = np.full((len(target_seq), 3), np.nan)
    coords[pairs[:, 1]] = xyz[pairs[:, 0]]
    return coords


def pair_rmsd(frames, ref):
    """Min RMSD over frames to ref, using target positions resolved in ref."""
    mask = ~np.isnan(ref[:, 0])
    return kabsch_rmsd(frames[:, mask], ref[mask])


def load_frames(pred_root, prefix, system_id):
    dirs = sorted(glob.glob(f"{pred_root}/{prefix}{system_id}/predictions/{system_id}"))
    if len(dirs) != 1:
        return None
    d = dirs[0]
    xtc = f"{d}/{system_id}_model_0.xtc"
    if Path(xtc).exists():  # walk-jump: one trajectory
        traj = md.load(xtc, top=f"{d}/{system_id}_model_0.cif")
    else:  # diffusion: one CIF per independent sample
        cifs = sorted(glob.glob(f"{d}/{system_id}_model_*.cif"),
                      key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]))
        if not cifs:
            return None
        traj = md.join([md.load(c) for c in cifs])
    traj = traj.atom_slice(traj.top.select("name CA"))
    return 10 * traj.xyz  # nm -> Angstrom


def coverage(fracs, thresholds):
    fracs = np.asarray(fracs)
    return {t: float(np.mean(fracs <= t)) for t in thresholds}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ref_dir", default=str(HERE.parent / "conformix_ref_pdbs"))
    parser.add_argument("--check_refs", action="store_true")
    parser.add_argument("--predictions_dir")
    parser.add_argument("--prefix", default="boltz_jump_results_s=2.0_d=1.0_f=0.1_")
    parser.add_argument("--max_frames", type=int, help="Use only the first N frames")
    parser.add_argument("--out_csv")
    args = parser.parse_args()

    systems = load_dataset(args.dataset)
    rows = []
    for s in systems.itertuples():
        try:
            r1 = reference(s.pdb1, s.sequence, args.ref_dir)
            r2 = reference(s.pdb2, s.sequence, args.ref_dir)
        except Exception as e:  # report and skip unusable reference files
            print(f"  skipping {s.system_id} ({s.pdb1}, {s.pdb2}): {e}")
            continue
        both = ~np.isnan(r1[:, 0]) & ~np.isnan(r2[:, 0])
        ref_ref = kabsch_rmsd(r1[None, both], r2[both])[0]
        row = {"system_id": s.system_id, "csv_ref_rmsd": s.RMSD, "kabsch_ref_rmsd": ref_ref,
               "n_res_ref1": int((~np.isnan(r1[:, 0])).sum()), "n_res_ref2": int((~np.isnan(r2[:, 0])).sum())}
        if not args.check_refs:
            frames = load_frames(args.predictions_dir, args.prefix, s.system_id)
            if frames is None:
                print(f"  missing predictions: {s.system_id}")
                continue
            if args.max_frames:
                frames = frames[: args.max_frames]
            if frames.shape[1] != len(s.sequence):
                print(f"  residue count mismatch: {s.system_id}")
                continue
            d1, d2 = pair_rmsd(frames, r1).min(), pair_rmsd(frames, r2).min()
            row.update(n_frames=len(frames), rmsd_ref1=d1, rmsd_ref2=d2,
                       best_frac=min(d1, d2) / s.RMSD, alt_frac=max(d1, d2) / s.RMSD,
                       ref2_frac=d2 / s.RMSD)
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    if args.check_refs:
        diff = df.kabsch_ref_rmsd - df.csv_ref_rmsd
        print(f"{args.dataset}: {len(df)} systems; Kabsch ref-ref RMSD minus CSV RMSD: "
              f"median {diff.median():+.2f} A, |diff| < 0.5 A for {np.mean(diff.abs() < 0.5):.0%}, "
              f"max |diff| {diff.abs().max():.2f} A")
        return
    thresholds = np.round(np.linspace(0.1, 1.0, 10), 1)
    best, alt = coverage(df.best_frac, thresholds), coverage(df.alt_frac, thresholds)
    auc = lambda c: float(np.trapz([0] + list(c.values()), [0] + list(thresholds)))
    print(f"{args.dataset}: {len(df)}/{len(systems)} systems, median {int(df.n_frames.median())} frames")
    print(f"  coverage@50%: best-matched {best[0.5]:.2f}, alternate {alt[0.5]:.2f}   "
          f"AUC: best {auc(best):.2f}, alternate {auc(alt):.2f}")


if __name__ == "__main__":
    main()
