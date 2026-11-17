#!/usr/bin/env python3
"""Assign Abl kinase samples to the active (6XR6) or inactive I1 (6XR7) state.

Each frame is superposed on the stable kinase core and compared to both
reference structures over the regions that differ between them (P-loop, aC
helix, DFG motif + activation loop). The C-terminal tail, which moves by up to
~40 A between the two structures, is reported separately.

Usage:
    python scripts/eval/abl_kinase_states.py --name diffusion --cif_dir <predictions dir>
    python scripts/eval/abl_kinase_states.py --name walk_jump_from_6XR6 --xtc traj.xtc --top model_0.cif
"""

import argparse
import glob

import mdtraj as md
import numpy as np
import pandas as pd

FIRST_RESSEQ = 248  # crystal numbering of the first residue in the construct
STATE_REGIONS = [(265, 275), (285, 300), (398, 421)]  # P-loop, aC helix, DFG + A-loop
TAIL = (520, 534)


def ca_coords(traj):
    traj = traj.atom_slice(traj.top.select("name CA"))
    return traj


def resseq_mask(n_res, ranges):
    resseq = FIRST_RESSEQ + np.arange(n_res)
    return np.any([(resseq >= lo) & (resseq <= hi) for lo, hi in ranges], axis=0)


def region_rmsd(frames, ref, core, region):
    """RMSD over `region` after superposing every frame on `ref` using `core`."""
    frames = frames.superpose(ref, atom_indices=np.where(core)[0])
    diff = frames.xyz[:, region] - ref.xyz[0, region]
    return 10 * np.sqrt((diff**2).sum(-1).mean(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cif_dir", help="Directory of predicted CIFs (one sample each)")
    parser.add_argument("--xtc", help="Trajectory file")
    parser.add_argument("--top", help="Topology for --xtc")
    parser.add_argument("--active", default="structures/6XR6.cif")
    parser.add_argument("--inactive", default="structures/6XR7.cif")
    parser.add_argument("--out_csv")
    args = parser.parse_args()

    if args.cif_dir:
        files = sorted(glob.glob(f"{args.cif_dir}/*_model_*.cif"))
        frames = md.join([ca_coords(md.load(f)) for f in files])
    else:
        frames = ca_coords(md.load(args.xtc, top=args.top))
    refs = {"6XR6": ca_coords(md.load(args.active)), "6XR7": ca_coords(md.load(args.inactive))}

    n_res = frames.n_atoms
    assert all(r.n_atoms == n_res for r in refs.values()), "residue count mismatch with references"
    state = resseq_mask(n_res, STATE_REGIONS)
    tail = resseq_mask(n_res, [TAIL])
    core = ~state & ~tail

    df = pd.DataFrame({"frame": np.arange(frames.n_frames)})
    for label, ref in refs.items():
        df[f"state_rmsd_{label}"] = region_rmsd(frames, ref, core, state)
        df[f"tail_rmsd_{label}"] = region_rmsd(frames, ref, core, tail)
    df["closer_to"] = np.where(df.state_rmsd_6XR6 < df.state_rmsd_6XR7, "6XR6", "6XR7")
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    ref_gap = region_rmsd(refs["6XR7"], refs["6XR6"], core, state)[0]
    print(f"{args.name}: {len(df)} frames; references are {ref_gap:.2f} A apart over the state regions")
    print(f"  closer to 6XR6 (active): {np.mean(df.closer_to == '6XR6'):.0%}")
    for label in refs:
        print(f"  state-region RMSD to {label}: median {df[f'state_rmsd_{label}'].median():.2f} A "
              f"(min {df[f'state_rmsd_{label}'].min():.2f})   tail: median {df[f'tail_rmsd_{label}'].median():.2f} A")


if __name__ == "__main__":
    main()
