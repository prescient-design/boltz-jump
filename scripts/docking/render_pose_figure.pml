# Render the two panels of a pose-refinement figure from export_pose_figure.py output.
#
#   pymol -cq render_pose_figure.pml -- <pose dir> <output prefix>
#
# Writes <prefix>_seed.png (co-folded seed vs crystal) and <prefix>_walkjump.png
# (best walk-jump pose vs crystal) from the same camera. Crystal ligand grey,
# seed orange (Boltz-2), walk-jump blue (Boltz-Jump), each in a single colour so
# the poses read at slide size; receptor as a faint cartoon. Everything is already in the crystal frame.

python
from pymol import cmd
import sys
pose_dir, prefix = sys.argv[-2], sys.argv[-1]
cmd.load(f"{pose_dir}/receptor_walkjump.pdb", "receptor")
cmd.load(f"{pose_dir}/ligand_crystal.sdf", "crystal")
cmd.load(f"{pose_dir}/ligand_seed.sdf", "seed")
cmd.load(f"{pose_dir}/ligand_walkjump.sdf", "walkjump")
cmd.remove("hydro")

cmd.hide("everything")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.set("depth_cue", 0)
cmd.set("ray_trace_mode", 0)

cmd.show("cartoon", "receptor")
cmd.set("cartoon_transparency", 0.75)
cmd.color("grey80", "receptor")
# no pocket side chains: the walk-jump protein is a one-step denoise at sigma = 4 A and its
# side-chain geometry is rough (PyMOL then infers spurious bonds); the cartoon is reliable

cmd.set("stick_radius", 0.22)
cmd.color("grey40", "crystal")
cmd.color("0xeb6834", "seed")
cmd.color("0x2a78d6", "walkjump")

cmd.orient("crystal or seed or walkjump")
cmd.zoom("crystal or seed or walkjump", 3.0)
cmd.clip("slab", 60)
view = cmd.get_view()

for name in ("seed", "walkjump"):
    cmd.hide("sticks", "crystal or seed or walkjump")
    cmd.show("sticks", f"crystal or {name}")
    cmd.set("stick_transparency", 0.25, "crystal")
    cmd.set_view(view)
    cmd.ray(1400, 1100)
    cmd.png(f"{prefix}_{name}.png", dpi=300)
python end
