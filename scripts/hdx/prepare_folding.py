#!/usr/bin/env python3
"""Walk-jump folding runs from random coil for Start2Fold folding-HDX entries.

For every entry prepared by prepare_start2fold.py that reports at least two
folding classes (EARLY / INTERMEDIATE / LATE, from pulse-labelling or
quenched-flow HDX), build random-coil starts for the same resolved sequence
(build_unfolded.py) and write walk-jump configs that start from them, with the
entry's MSA and no guidance. Each start becomes one run of 3000 steps at
sigma = 2 A.

Writes $HDX_DIR/start2fold/folding/{starts,configs}/ and folding/tasks.tsv.

Usage:
    HDX_DIR=/path python scripts/hdx/prepare_folding.py --n_starts 3
"""

import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import gemmi
import numpy as np
from Bio.PDB import PDBIO

from build_unfolded import build, clashes, draw

WJ_ARGS = ("--walk_jump_sigma 2.0 --walk_jump_delta 1.0 --walk_jump_friction 0.1 "
           "--walk_jump_steps 3000 --no_kernels --override")


def folding_classes(entry_xml):
    classes = set()
    for e in ET.parse(entry_xml).getroot().iter("experiment"):
        if e.find("method").get("type") == "folding":
            classes.add(e.find("protection").get("protection_level"))
    return classes


def coil_cif(sequence, chain, rng, out):
    """Write one clash-free random-coil structure as mmCIF with the given chain name."""
    while True:
        s = build(sequence, draw(sequence, rng))
        if not clashes(s):
            break
    tmp = out.with_suffix(".pdb")
    io = PDBIO()
    io.set_structure(s)
    io.save(str(tmp))
    st = gemmi.read_structure(str(tmp))
    for ch in st[0]:
        ch.name = chain
        for r in ch:
            r.subchain = chain
    st.setup_entities()
    ch = st[0][chain]
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer:
            ent.full_sequence = [r.name for r in ch]
            ent.polymer_type = gemmi.PolymerType.PeptideL
    st.assign_label_seq_id()
    st.make_mmcif_document().write_file(str(out))
    tmp.unlink()
    ca = np.array([r["CA"].coord for r in s.get_residues()])
    return float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdx_dir", default=os.environ.get("HDX_DIR"))
    parser.add_argument("--n_starts", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.hdx_dir) / "start2fold"
    out = root / "folding"
    for sub in ("starts", "configs", "runs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    tasks = []
    for r in csv.DictReader(open(root / "manifest.csv")):
        if r["status"] != "ready":
            continue
        eid = r["entry"]
        if len(folding_classes(root / "downloads" / f"{eid}.xml")) < 2:
            continue
        cfg = (root / "configs" / f"{eid}.yaml").read_text()
        seq = re.search(r"sequence: (\S+)", cfg).group(1)
        rng = np.random.default_rng(int(eid[3:]))  # fixed per entry
        for k in range(args.n_starts):
            start = out / "starts" / f"{eid}_coil{k}.cif"
            rg = coil_cif(seq, r["chain"], rng, start)
            new = re.sub(r"initial_structure:\n  cif: \S+", f"initial_structure:\n  cif: {start}", cfg)
            path = out / "configs" / f"{eid}_coil{k}.yaml"
            path.write_text(new)
            tasks.append(f"{path}\t{out / 'runs' / eid / f'coil{k}'}\t0\t{WJ_ARGS}\n")
            print(f"{eid} coil{k}: {len(seq)} aa, Rg {rg:.1f} A")
    (out / "tasks.tsv").write_text("".join(tasks))
    print(f"\n{len(tasks)} runs -> {out / 'tasks.tsv'}")


if __name__ == "__main__":
    main()
