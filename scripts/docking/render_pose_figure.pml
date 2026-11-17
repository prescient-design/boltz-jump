# Render a pose-refinement panel from export_pose_figure.py output.
#
#   pymol -cq render_pose_figure.pml -- <pose dir> <output png>
#
# Crystal ligand maroon, walk-jump pose green, co-folded seed grey; receptor
# surface of the pocket in pale grey. Everything is already in the crystal
# frame, so no alignment is done here.

python
from pymol import cmd
import sys
pose_dir, out_png = sys.argv[-2], sys.argv[-1]
cmd.load(f"{pose_dir}/receptor_walkjump.pdb", "receptor")
cmd.load(f"{pose_dir}/ligand_crystal.sdf", "crystal")
cmd.load(f"{pose_dir}/ligand_seed.sdf", "seed")
cmd.load(f"{pose_dir}/ligand_walkjump.sdf", "walkjump")
python end

hide everything
bg_color white
set ray_opaque_background, 1

# pocket: cartoon plus thin lines for the lining residues. A surface gets
# sliced open by the near clipping plane at this zoom, so avoid it.
select pocket, byres (receptor within 6 of crystal) and polymer
show cartoon, receptor
set cartoon_transparency, 0.65
color grey80, receptor
show lines, pocket and sidechain
set line_width, 1.6
color grey70, pocket and elem C

show sticks, crystal or seed or walkjump
set stick_radius, 0.16
color grey60, seed
color 0x800000, crystal
color 0x1B5E20, walkjump
util.cnc("crystal or seed or walkjump")
color grey60, seed and elem C
color 0x800000, crystal and elem C
color 0x1B5E20, walkjump and elem C

orient crystal or seed or walkjump
zoom crystal or seed or walkjump, 3.5
turn x, -15
clip slab, 60

set ray_shadows, 0
set antialias, 2
set depth_cue, 0
set ray_trace_mode, 0

python
from pymol import cmd
import sys
cmd.ray(1600, 1200)
cmd.png(sys.argv[-1], dpi=300)
python end
