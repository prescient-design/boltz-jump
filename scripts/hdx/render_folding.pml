# Render one structure as a rainbow cartoon (N-terminus blue -> C-terminus red).
#   pymol -cq render_folding.pml -- <structure.pdb> <reference.pdb> <out.png>
# The reference fixes the view, so every panel of one protein is shown from the
# same orientation (structures are already superposed on it).

python
import sys
from pymol import cmd
pdb, ref, png = sys.argv[-3], sys.argv[-2], sys.argv[-1]
cmd.load(ref, "ref")
cmd.load(pdb, "mol")
cmd.dss("mol")
cmd.hide("everything")
cmd.show("cartoon", "mol")
cmd.spectrum("resi", "rainbow", "mol and name CA")
cmd.set("cartoon_transparency", 0.0)
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.orient("ref")
cmd.zoom("ref or mol", 2)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.ray(900, 900)
cmd.png(png, dpi=200)
python end
