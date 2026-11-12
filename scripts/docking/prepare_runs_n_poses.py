#!/usr/bin/env python3
"""Select Runs N' Poses pose-refinement targets and walk-jump starts.

Uses the Runs N' Poses release (Zenodo 18366081): the authors' Boltz-2 scores
(predictions/boltz2.csv), annotations.csv, ground_truth/ and their Boltz-2
predicted structures (prediction_files/boltz2/<system>/). Only Boltz-2 files are
used.

A system is a target ("clean pose failure") if, over all of the authors'
Boltz-2 samples, the best ligand RMSD is > 2 A while the pocket is correct at
that sample (lDDT-LP >= 0.7 and binding-site backbone RMSD <= 2 A), the best
ligand RMSD is <= 7 A, it is a proper ligand, and the predicted structures
contain a single ligand chain.

Walk-jump starts: the top-ranked sample, then farthest-point sampling of the
pocket-aligned ligand centroid, up to --n_starts per target.

Writes under $DOCKING_DIR/runs_n_poses/: targets.csv, starts.csv,
sequences.fasta (unique protein sequences; MSAs expected at
msa/<hash>.a3m) and start configs (configs/<system>__<start>.yaml, without
initial_structure, which prepare_walkjump.py adds).

Usage:
    python scripts/docking/prepare_runs_n_poses.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

from common import SUCCESS_RMSD_A, docking_dir, kabsch_rt, msa_path, one_letter, seq_hash

LDDT_LP_MIN = 0.7
BB_RMSD_MAX_A = 2.0
MAX_BEST_RMSD_A = float("inf")  # no post-hoc filter on seed distance


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mine_failures(pred_csv, annot_csv):
    annot = {(r["system_id"], r.get("ligand_instance_chain", "")): r for r in csv.DictReader(open(annot_csv))}
    per = defaultdict(list)
    for r in csv.DictReader(open(pred_csv)):
        if str(r.get("ligand_is_proper")).lower() not in ("true", "1") or fl(r["rmsd"]) is None:
            continue
        per[(r["target"], r.get("ligand_instance_chain", ""))].append(r)
    rows = []
    for (system, lig_chain), samples in sorted(per.items()):
        best = min(samples, key=lambda s: fl(s["rmsd"]))
        b, lp, bb = fl(best["rmsd"]), fl(best["lddt_lp"]), fl(best["bb_rmsd"])
        pocket_ok = lp is not None and lp >= LDDT_LP_MIN and bb is not None and bb <= BB_RMSD_MAX_A
        verdict = "ok" if b <= SUCCESS_RMSD_A else ("clean_pose_failure" if pocket_ok else "wrong_pocket_or_fold")
        a = annot.get((system, lig_chain)) or annot.get((system, "")) or {}
        rows.append({"system": system, "crystal_ligand_chain": lig_chain, "verdict": verdict,
                     "n_samples": len(samples), "best_rmsd": round(b, 3),
                     "ligand_heavy": a.get("ligand_num_heavy_atoms", ""), "smiles": a.get("ligand_smiles", ""),
                     "train_sim_sucos": a.get("sucos_shape_pocket_qcov", "")})
    return rows


def chains(cif):
    """Protein chains [(name, sequence, CA coords)] and ligand chains [(name, heavy-atom coords)]."""
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    prot, ligs = [], []
    for ch in st[0]:
        seq, ca, xyz = [], [], []
        for res in ch:
            code = one_letter(res.name)
            if code:
                seq.append(code)
                a = res.find_atom("CA", "*")
                if a:
                    ca.append(a.pos.tolist())
            elif res.name not in ("HOH", "WAT"):
                xyz += [a.pos.tolist() for a in res if a.element.name != "H"]
        if seq:
            prot.append((ch.name, "".join(seq), np.array(ca)))
        elif xyz:
            ligs.append((ch.name, np.array(xyz)))
    return prot, ligs


def pick_starts(sys_dir, k):
    """Top-ranked sample, then farthest-point sampling of the CA-aligned ligand centroid."""
    rows = list(csv.DictReader(open(sys_dir / "ranking_scores.csv")))
    cands = sorted(((float(r["ranking_score"]), sys_dir / f"seed-{r['seed']}_sample-{r['sample']}.cif") for r in rows
                    if (sys_dir / f"seed-{r['seed']}_sample-{r['sample']}.cif").exists()), key=lambda x: -x[0])
    if not cands:
        return []
    ref_prot, ref_ligs = chains(cands[0][1])
    if len(ref_ligs) != 1:  # every sample has the same chains; multi-ligand systems are excluded
        return []
    ref_ca = ref_prot[0][2]
    pts = []
    for _, cif in cands:
        prot, ligs = chains(cif)
        if len(ligs) != 1:
            continue
        ca = prot[0][2]
        if ca.shape == ref_ca.shape:
            R, t = kabsch_rt(ca, ref_ca)
            cent = R @ ligs[0][1].mean(0) + t
        else:
            cent = ligs[0][1].mean(0)
        pts.append((cif, cent))
    if not pts:
        return []
    picked = [0]
    while len(picked) < min(k, len(pts)):
        picked.append(max((i for i in range(len(pts)) if i not in picked),
                          key=lambda i: min(np.linalg.norm(pts[i][1] - pts[j][1]) for j in picked)))
    return [pts[i][0] for i in picked]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_starts", type=int, default=5)
    args = parser.parse_args()

    data = docking_dir() / "data/runs_n_poses"
    out = docking_dir() / "runs_n_poses"
    (out / "configs").mkdir(parents=True, exist_ok=True)
    mined = mine_failures(data / "predictions/boltz2.csv", data / "annotations.csv")
    counts = defaultdict(int)
    for r in mined:
        counts[r["verdict"]] += 1
    print(f"{len(mined)} systems with a proper ligand: {dict(counts)}")

    targets, starts, seqs, skipped = [], [], {}, defaultdict(list)
    for r in mined:
        if r["verdict"] != "clean_pose_failure":
            continue
        if r["best_rmsd"] > MAX_BEST_RMSD_A:
            skipped["best co-folded pose > 7 A"].append(r["system"])
            continue
        sys_dir = data / "boltz2_predictions" / r["system"]
        if not sys_dir.is_dir():
            skipped["no predicted structures"].append(r["system"])
            continue
        if not r["smiles"]:
            skipped["no SMILES"].append(r["system"])
            continue
        picks = pick_starts(sys_dir, args.n_starts)
        if not picks:
            skipped["multiple ligand chains"].append(r["system"])
            continue
        targets.append(r)
        for i, cif in enumerate(picks):
            prot, ligs = chains(cif)
            lines = ["version: 1", "sequences:"]
            for name, seq, _ in prot:
                seqs[seq_hash(seq)] = seq
                lines += ["  - protein:", f"      id: '{name}'", f"      sequence: {seq}",
                          f"      msa: {msa_path(seq)}"]
            lines += ["  - ligand:", f"      id: '{ligs[0][0]}'", f"      smiles: '{r['smiles']}'"]
            cfg = out / "configs" / f"{r['system']}__start{i}.yaml"
            cfg.write_text("\n".join(lines) + "\n")
            starts.append({"system": r["system"], "start": f"start{i}", "cif": str(cif), "config": str(cfg),
                           "crystal_ligand_chain": r["crystal_ligand_chain"], "smiles": r["smiles"],
                           "ligand_heavy": r["ligand_heavy"]})
    for name, rows in [("targets.csv", targets), ("starts.csv", starts)]:
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "sequences.fasta").write_text("".join(f">{h}\n{s}\n" for h, s in sorted(seqs.items())))
    print(f"{len(targets)} targets, {len(starts)} starts, {len(seqs)} unique sequences")
    print({k: len(v) for k, v in skipped.items()})


if __name__ == "__main__":
    main()
