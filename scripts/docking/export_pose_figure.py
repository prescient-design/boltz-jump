#!/usr/bin/env python3
"""Export the poses for a pose-refinement figure, in the crystal frame.

For one target, writes the crystal ligand, the co-folded seed ligand, the best
idealized walk-jump ligand, and the predicted receptor, all superposed on the
crystal using the pocket C-alpha atoms (the same transform used for scoring).
Rendering is left to render_pose_figure.pml.

Usage:
    python scripts/docking/export_pose_figure.py --target 8sx4__1__1.A__1.C --sigma 4
"""

import argparse
import csv
import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from common import Ideal, PoseScorer, Reference, docking_dir, kabsch_rt


def write_xyz_as_sdf(template_mol, xyz, path, name):
    mol = Chem.Mol(template_mol)
    conf = mol.GetConformer()
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mol.SetProp("_Name", name)
    Chem.SDWriter(str(path)).write(mol)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sigma", default="4")
    parser.add_argument("--start", default="start0")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--trajectory", action="store_true",
                        help="also write the whole walk-jump trajectory in the crystal frame")
    args = parser.parse_args()

    root = docking_dir() / "runs_n_poses"
    out = Path(args.out_dir or root / "figure" / args.target)
    out.mkdir(parents=True, exist_ok=True)

    row = next(r for r in csv.DictReader(open(root / "starts.csv"))
               if r["system"] == args.target and r["start"] == args.start)
    gt_dir = docking_dir() / "data/runs_n_poses/ground_truth" / args.target
    ref = Reference.runs_n_poses(gt_dir, row["crystal_ligand_chain"])
    ideal = Ideal(row["smiles"])

    run = next((root / "walkjump" / f"{args.target}__{args.start}" / f"sigma{args.sigma}" / "seed0")
               .glob(f"boltz_jump_results_*/predictions/*"))
    name = run.name
    top = run / f"{name}_model_0.cif"
    scorer = PoseScorer(ref, top, ideal)

    traj = md.load(str(run / f"{name}_model_0.xtc"), top=str(top))
    best, best_k = None, None
    for k in range(traj.n_frames):
        s = scorer.score(traj.xyz[k] * 10.0, idealize=True)
        if s.get("ideal_rmsd") is not None and (best is None or s["ideal_rmsd"] < best["ideal_rmsd"]):
            best, best_k = s, k

    seed_xyz = md.load(str(top)).xyz[0] * 10.0
    seed = scorer.score(seed_xyz)

    # receptor of the best frame, moved into the crystal frame by the pocket transform
    frame = traj.xyz[best_k] * 10.0
    pred_idx, cry = scorer.aligners[0]
    rot, t = kabsch_rt(frame[pred_idx], cry)
    moved = (rot @ frame.T).T + t
    out_traj = md.load(str(top))
    out_traj.xyz = (moved / 10.0)[None]
    out_traj.save_pdb(str(out / "receptor_walkjump.pdb"))

    # ligands: crystal, seed, best walk-jump (idealized) -- all already in the crystal frame
    # Coordinates must be written against a template whose atom order matches
    # them. The crystal ligand is copied from its own SDF rather than rewritten;
    # predicted poses are reordered into ideal-molecule order with the same map
    # the scorer uses.
    shutil.copyfile(gt_dir / "ligand_files" / f"{row['crystal_ligand_chain']}.sdf",
                    out / "ligand_crystal.sdf")
    rot_s, t_s = kabsch_rt(seed_xyz[pred_idx], cry)
    seed_lig = (rot_s @ seed_xyz[scorer.lig_idx].T).T + t_s
    write_xyz_as_sdf(ideal.mol, seed_lig[scorer.ideal_map], out / "ligand_seed.sdf", "cofold seed")
    write_xyz_as_sdf(ideal.mol, best["ideal_xyz"], out / "ligand_walkjump.sdf", "walk-jump")

    if args.trajectory:
        # Every frame moved into the crystal frame by its own pocket alignment, so
        # the protein stays still and the ligand is seen moving relative to the
        # crystal pose. Written as a multi-model PDB that plays as an animation.
        moved_frames = np.empty_like(traj.xyz)
        per_frame = []
        for k in range(traj.n_frames):
            f = traj.xyz[k] * 10.0
            r, tt = kabsch_rt(f[pred_idx], cry)
            moved_frames[k] = ((r @ f.T).T + tt) / 10.0
            per_frame.append(scorer.score(f))
        anim = md.load(str(top))
        anim.xyz = moved_frames
        anim.time = np.arange(traj.n_frames, dtype=float)
        anim.save_pdb(str(out / "trajectory_crystal_frame.pdb"))
        anim.save_xtc(str(out / "trajectory_crystal_frame.xtc"))
        with open(out / "trajectory_rmsd.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "lig_rmsd", "pocket_ca"])
            for k, sc in enumerate(per_frame):
                w.writerow([k, round(sc["lig_rmsd"], 3), round(sc["pocket_ca"], 3)])
        print(f"  trajectory: {traj.n_frames} frames -> trajectory_crystal_frame.{{pdb,xtc}} + trajectory_rmsd.csv")

    print(f"{args.target}: seed {seed['lig_rmsd']:.2f} A -> walk-jump {best['ideal_rmsd']:.2f} A "
          f"(frame {best_k} of {traj.n_frames})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
