#!/usr/bin/env python3
"""Export the poses for a pose-refinement figure, in the crystal frame.

For one walk-jump start, writes the crystal ligand, the co-folded seed ligand,
the best valid idealized walk-jump ligand, the predicted receptor (protein only)
and poses.json (RMSDs, frame), all superposed on the crystal using the pocket
C-alpha atoms with the same transform the scorer picks. Rendering is left to
render_pose_figure.pml / make_pose_figure.py.

Usage:
    python scripts/docking/export_pose_figure.py --target 8sx4__1__1.A__1.C --sigma 4
    python scripts/docking/export_pose_figure.py --dataset posebusters --target 8DW5_FQ7 --start start4
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from analyze_walkjump import target_info
from common import Ideal, PoseScorer, docking_dir, kabsch_rt


def write_xyz_as_sdf(template_mol, xyz, path, name):
    mol = Chem.Mol(template_mol)
    conf = mol.GetConformer()
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mol.SetProp("_Name", name)
    Chem.SDWriter(str(path)).write(mol)


def best_transform(scorer, xyz):
    """Pocket superposition onto the crystal that PoseScorer.score uses (lowest ligand RMSD)."""
    best = None
    for pred_idx, cry in scorer.aligners:
        rot, t = kabsch_rt(xyz[pred_idx], cry)
        r = scorer._best(scorer.crystal_maps, (rot @ xyz[scorer.lig_idx].T).T + t)
        if best is None or r < best[0]:
            best = (r, rot, t)
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["runs_n_poses", "posebusters"], default="runs_n_poses")
    parser.add_argument("--target", required=True)
    parser.add_argument("--sigma", default="4")
    parser.add_argument("--start", default="start0")
    parser.add_argument("--seed", type=int, default=0, help="walk-jump run seed")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--trajectory", action="store_true",
                        help="also write the whole walk-jump trajectory in the crystal frame")
    args = parser.parse_args()

    root = docking_dir() / args.dataset
    out = Path(args.out_dir or root / "figure" / args.target)
    out.mkdir(parents=True, exist_ok=True)

    _, ref, smiles = target_info(args.dataset, root, f"{args.target}__{args.start}")
    ideal = Ideal(smiles)

    run = next((root / "walkjump" / f"{args.target}__{args.start}" / f"sigma{args.sigma}" / f"seed{args.seed}")
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

    # receptor (protein only) of the best frame, moved into the crystal frame by the pocket transform
    frame = traj.xyz[best_k] * 10.0
    rot, t = best_transform(scorer, frame)
    moved = (rot @ frame.T).T + t
    out_traj = md.load(str(top))
    out_traj.xyz = (moved / 10.0)[None]
    receptor = out_traj.atom_slice(out_traj.top.select("protein"))
    for atom in receptor.top.atoms:
        atom.serial = None  # CIF serials are strings, which the PDB writer can't format
    receptor.save_pdb(str(out / "receptor_walkjump.pdb"))

    # ligands: crystal, seed, best walk-jump (idealized) -- all already in the crystal frame
    # Coordinates must be written against a template whose atom order matches
    # them. The crystal ligand is copied from its own SDF rather than rewritten;
    # predicted poses are reordered into ideal-molecule order with the same map
    # the scorer uses.
    if args.dataset == "posebusters":
        crystal_sdf = docking_dir() / "data/posebusters/posebusters_benchmark_set" / args.target / f"{args.target}_ligand.sdf"
    else:
        row = next(r for r in csv.DictReader(open(root / "starts.csv"))
                   if r["system"] == args.target and r["start"] == args.start)
        crystal_sdf = (docking_dir() / "data/runs_n_poses/ground_truth" / args.target / "ligand_files"
                       / f"{row['crystal_ligand_chain']}.sdf")
    shutil.copyfile(crystal_sdf, out / "ligand_crystal.sdf")
    rot_s, t_s = best_transform(scorer, seed_xyz)
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
            r, tt = best_transform(scorer, f)
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

    json.dump({"dataset": args.dataset, "target": args.target, "start": args.start, "sigma": args.sigma,
               "seed": args.seed, "seed_lig_rmsd": seed["lig_rmsd"], "walkjump_rmsd": best["ideal_rmsd"],
               "frame": best_k, "n_frames": traj.n_frames}, open(out / "poses.json", "w"), indent=1)
    print(f"{args.target}: seed {seed['lig_rmsd']:.2f} A -> walk-jump {best['ideal_rmsd']:.2f} A "
          f"(frame {best_k} of {traj.n_frames})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
