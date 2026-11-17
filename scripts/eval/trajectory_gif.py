#!/usr/bin/env python3
"""Movies of walk-jump trajectories, coloured by secondary structure, for slides.

Every frame is superposed on --ref by C-alpha (optionally only the core, dropping
--trim residues at each end) and rendered as a cartoon from one fixed camera
shared by all trajectories, coloured with PyMOL's helix cyan / sheet red / loop magenta
preset, reassigned per
frame (PyMOL stores secondary structure per atom, not per state).

Writes <out_prefix>_<i>.gif per trajectory and <out_prefix>_side_by_side.gif.

Usage:
    python scripts/eval/trajectory_gif.py --ref structures/6XR6.cif --pymol $(which pymol) \
        --traj "Boltz-Jump from the active state (6XR6)=<run>/..._model_0.xtc" \
        --traj "Boltz-Jump from the inactive state (6XR7)=<run>/..._model_0.xtc" \
        --out_prefix figures/abl_kinase/abl_kinase
"""

import argparse
import glob
import subprocess
import tempfile
from pathlib import Path

import mdtraj as md
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

INK, INK2 = "#0b0b0b", "#52514e"
# PyMOL's "color by ss" preset: helix cyan, sheet red, loop magenta
HELIX, STRAND, LOOP = "cyan", "red", "magenta"

MOVIE_PML = """
python
import sys
from pymol import cmd, util
pdb, view, prefix, size = sys.argv[-4], sys.argv[-3], sys.argv[-2], int(sys.argv[-1])
cmd.load(view, "view")
cmd.load(pdb, "mol")
cmd.hide("everything")
cmd.show("cartoon", "mol")
cmd.set("cartoon_discrete_colors", 1)
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.orient("view")
cmd.zoom("view", 2, state=0)
cmd.disable("view")
for s in range(1, cmd.count_states("mol") + 1):
    cmd.frame(s)
    cmd.dss("mol", state=s)
    util.cbss("mol", "%s", "%s", "%s")
    cmd.ray(size, size)
    cmd.png(f"{prefix}{s:04d}.png")
python end
""" % (HELIX, STRAND, LOOP)


def font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="structure that fixes the orientation")
    parser.add_argument("--traj", action="append", required=True, help="LABEL=path/to/model_0.xtc (topology: .cif)")
    parser.add_argument("--pymol", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--trim", type=int, default=15, help="terminal residues left out of the superposition")
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--ms_per_frame", type=int, default=70)
    args = parser.parse_args()
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    ref = md.load(args.ref)
    rca = ref.topology.select("protein and name CA")
    trajs = []
    for spec in args.traj:
        label, xtc = spec.split("=", 1)
        t = md.load(xtc, top=xtc.replace(".xtc", ".cif"), stride=args.stride)
        ca = t.topology.select("name CA")
        assert len(ca) == len(rca), f"{xtc}: {len(ca)} residues, reference has {len(rca)}"
        core = slice(args.trim, len(ca) - args.trim)
        t.superpose(ref, atom_indices=ca[core], ref_atom_indices=rca[core])
        for a in t.topology.atoms:
            a.serial = None
        trajs.append((label, t))

    movies = []
    with tempfile.TemporaryDirectory() as tmp:
        view = f"{tmp}/view.pdb"
        md.join([t for _, t in trajs]).save_pdb(view)
        pml = f"{tmp}/movie.pml"
        Path(pml).write_text(MOVIE_PML)
        for k, (label, t) in enumerate(trajs):
            pdb = f"{tmp}/traj{k}.pdb"
            t.save_pdb(pdb)
            subprocess.run([args.pymol, "-cq", pml, "--", pdb, view, f"{tmp}/traj{k}_", str(args.size)], check=True)
            frames = []
            for i, png in enumerate(sorted(glob.glob(f"{tmp}/traj{k}_*.png"))):
                im = Image.open(png).convert("RGB")
                canvas = Image.new("RGB", (im.width, im.height + 100), "white")
                canvas.paste(im, (0, 100))
                d = ImageDraw.Draw(canvas)
                d.text((20, 12), label, fill=INK, font=font(24))
                d.text((20, 50), f"step {i * args.stride:>4}", fill=INK2, font=font(26))
                frames.append(canvas)
            frames[0].save(f"{args.out_prefix}_{k}.gif", save_all=True, append_images=frames[1:],
                           duration=args.ms_per_frame, loop=0)
            movies.append(frames)

    n = min(len(f) for f in movies)
    side = []
    for i in range(n):
        ims = [m[i] for m in movies]
        canvas = Image.new("RGB", (sum(im.width for im in ims), max(im.height for im in ims)), "white")
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0))
            x += im.width
        side.append(canvas)
    side[0].save(f"{args.out_prefix}_side_by_side.gif", save_all=True, append_images=side[1:],
                 duration=args.ms_per_frame, loop=0)
    print(f"-> {args.out_prefix}_*.gif")


if __name__ == "__main__":
    main()
