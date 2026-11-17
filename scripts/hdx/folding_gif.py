#!/usr/bin/env python3
"""Folding-from-random-coil movies, unguided and guided, for slides.

For each condition, the start with the highest final native-contact fraction Q
(same definition as plot_guidance_sweep.py) is shown; every frame is superposed
on the crystal structure by C-alpha and rendered as a rainbow cartoon from one
fixed camera, labelled with the walk step and Q.

Writes to --out_dir:
    <protein>_<condition>.gif     one movie per condition
    <protein>_side_by_side.gif    the conditions next to each other, frame-locked

Usage:
    python scripts/hdx/folding_gif.py --hdx_dir $HDX_DIR --pymol $(which pymol) --out_dir figures/folding
"""

import argparse
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

import mdtraj as md
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_guidance_sweep import native_contacts  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
def label(cond):
    if cond == "unguided":
        return "Boltz-Jump (no guidance)"
    return f"Boltz-Jump (guidance with weight {cond.removeprefix('guided_w')})" if cond.startswith("guided_w") else cond

# Secondary structure is per atom in PyMOL, not per state, so it is reassigned before
# rendering each frame. The camera fits every condition's frames from --zoom_from_step on
# (the extended starting coil would otherwise shrink the folded protein).
MOVIE_PML = """
python
import sys
from pymol import cmd
pdb, ref, view, prefix = sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1]
cmd.load(ref, "ref")
cmd.load(view, "view")
cmd.load(pdb, "mol")
cmd.hide("everything")
cmd.show("cartoon", "mol")
cmd.spectrum("resi", "rainbow", "mol and name CA")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.orient("ref")
cmd.zoom("view", 2, state=0)
cmd.disable("ref view")
for s in range(1, cmd.count_states("mol") + 1):
    cmd.frame(s)
    cmd.dss("mol", state=s)
    cmd.ray(600, 600)
    cmd.png(f"{prefix}{s:04d}.png")
python end
"""


def font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def load(xtc, ref, stride):
    t = md.load(xtc, top=xtc.replace(".xtc", ".cif"), stride=stride)
    ca, rca = t.topology.select("name CA"), ref.topology.select("name CA")
    t.superpose(ref, atom_indices=ca, ref_atom_indices=rca)
    for a in t.topology.atoms:
        a.serial = None
    return t


def q_of(t, ii, jj, d0):
    x = t.xyz[:, t.topology.select("name CA")]
    return (np.linalg.norm(x[:, ii] - x[:, jj], axis=-1) < 1.5 * d0).mean(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdx_dir", required=True)
    parser.add_argument("--pymol", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--protein", default="1ubq")
    parser.add_argument("--conditions", nargs="+", default=["unguided", "guided_w0.003"])
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--ms_per_frame", type=int, default=70)
    parser.add_argument("--zoom_from_step", type=int, default=300)
    args = parser.parse_args()
    h, out = Path(args.hdx_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = md.load(str(h / "structures" / f"{args.protein}.cif"))
    ref = ref.atom_slice(ref.topology.select("protein"))
    ii, jj, d0 = native_contacts(ref)

    movies = {}
    with tempfile.TemporaryDirectory() as tmp:
        ref_pdb = f"{tmp}/ref.pdb"
        for a in ref.topology.atoms:
            a.serial = None
        ref.save_pdb(ref_pdb)
        shown = {}
        for cond in args.conditions:
            files = sorted(glob.glob(str(h / f"runs_fold_{args.protein}" / cond / "start*" / "boltz_jump_results_*"
                                         / "predictions" / "*" / "*_model_0.xtc")))
            runs = [(f, load(f, ref, args.stride)) for f in files]
            qs = [q_of(t, ii, jj, d0) for _, t in runs]
            k = int(np.argmax([q[-1] for q in qs]))
            print(cond, "final Q per start:", [round(float(q[-1]), 2) for q in qs], "-> showing", Path(runs[k][0]).name)
            shown[cond] = runs[k][1], qs[k]
        # one camera for all conditions, so the movies share a scale
        view = f"{tmp}/view.pdb"
        md.join([t[args.zoom_from_step // args.stride:] for t, _ in shown.values()]).save_pdb(view)
        for cond, (t, q) in shown.items():
            pdb = f"{tmp}/{cond}.pdb"
            t.save_pdb(pdb)
            pml = f"{tmp}/movie.pml"
            Path(pml).write_text(MOVIE_PML)
            subprocess.run([args.pymol, "-cq", pml, "--", pdb, ref_pdb, view, f"{tmp}/{cond}_"], check=True)
            pngs = sorted(glob.glob(f"{tmp}/{cond}_*.png"))
            frames = []
            for i, png in enumerate(pngs):
                im = Image.open(png).convert("RGB")
                canvas = Image.new("RGB", (im.width, im.height + 110), "white")
                canvas.paste(im, (0, 110))
                d = ImageDraw.Draw(canvas)
                d.text((20, 12), label(cond), fill=INK, font=font(24))
                d.text((20, 52), f"step {i * args.stride:>4}    Q = {q[i]:.2f}", fill=INK2, font=font(26))
                frames.append(canvas)
            movies[cond] = frames
            frames[0].save(out / f"{args.protein}_{cond}.gif", save_all=True, append_images=frames[1:],
                           duration=args.ms_per_frame, loop=0)

    n = min(len(f) for f in movies.values())
    side = []
    for i in range(n):
        ims = [movies[c][i] for c in args.conditions]
        canvas = Image.new("RGB", (sum(im.width for im in ims), max(im.height for im in ims)), "white")
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0))
            x += im.width
        side.append(canvas)
    side[0].save(out / f"{args.protein}_side_by_side.gif", save_all=True, append_images=side[1:],
                 duration=args.ms_per_frame, loop=0)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
