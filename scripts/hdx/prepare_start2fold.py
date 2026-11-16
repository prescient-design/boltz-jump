#!/usr/bin/env python3
"""Prepare every Start2Fold entry for a native walk-jump HDX benchmark.

For each entry (https://bio2byte.be/start2fold/, Pancsa et al., NAR 2016):

1. Download the entry XML from Start2Fold and its PDB structure from RCSB.
2. Keep entries with native-state stability data (STRONG / MEDIUM classes);
   folding-kinetics-only entries cannot be scored against native fluctuations.
3. Pick the first protein chain whose *resolved* sequence is an exact substring
   of the HDX construct, and write that chain (first model, no ligands or water)
   as an mmCIF whose entity sequence is the resolved residues only, so Boltz's
   initial-structure check passes. Anything else is skipped with a reason
   rather than aligned approximately.
4. Fetch an MSA for the resolved sequence from the ColabFold server.
5. Write a walk-jump config seeded from the experimental structure.

Writes $HDX_DIR/start2fold/manifest.csv (one row per entry, with a status and,
for skipped entries, the reason) and configs/<entry>.yaml.

Usage:
    HDX_DIR=/path python scripts/hdx/prepare_start2fold.py
"""

import argparse
import csv
import hashlib
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import gemmi

S2F = "https://bio2byte.be/start2fold"
RCSB = "https://files.rcsb.org/download"


def fetch(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def entry_ids(root):
    html = fetch(f"{S2F}/proteins", root / "downloads" / "proteins.html").read_text()
    return sorted(set(re.findall(r'href="/start2fold/(STF\d{4})"', html)))


def resolved_chain(cif, construct):
    """(chain name, resolved sequence, offset) of the first chain inside the construct, or None."""
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    for ch in st[0]:
        poly = ch.get_polymer()
        if len(poly) < 20 or poly.check_polymer_type() not in (gemmi.PolymerType.PeptideL,):
            continue
        seq = gemmi.one_letter_code([r.name for r in poly])
        offset = construct.find(seq)
        if offset >= 0:
            return ch.name, seq, offset
    return None


def write_resolved(cif, chain, out):
    st = gemmi.read_structure(str(cif))
    while len(st) > 1:
        del st[1]
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    for c in [c.name for c in st[0] if c.name != chain]:
        st[0].remove_chain(c)
    st.setup_entities()
    ch = st[0][chain]
    # Keep exactly the polymer residues -- the set the sequence was derived from.
    keep = {(r.seqid.num, r.seqid.icode) for r in ch.get_polymer()}
    for i in reversed(range(len(ch))):
        if (ch[i].seqid.num, ch[i].seqid.icode) not in keep:
            del ch[i]
    for r in ch:
        r.subchain = chain
    st.setup_entities()
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer:
            ent.full_sequence = [r.name for r in ch]
            ent.polymer_type = gemmi.PolymerType.PeptideL
    st.assign_label_seq_id()
    st.make_mmcif_document().write_file(str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdx_dir", default=os.environ.get("HDX_DIR"))
    args = parser.parse_args()
    root = Path(args.hdx_dir) / "start2fold"
    for sub in ("downloads", "structures", "msa", "configs"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    from boltz.main import compute_msa  # deferred: heavy import

    rows = []
    for eid in entry_ids(root):
        row = {"entry": eid}
        try:
            xml = fetch(f"{S2F}/{eid}.xml", root / "downloads" / f"{eid}.xml")
            r = ET.parse(xml).getroot()
            prot = r.find("protein")
            row.update(name=r.get("title"), pdb=prot.get("pdb_id"), length=prot.get("number_of_residues"))
            classes = {e.find("protection").get("protection_level") for e in r.iter("experiment")
                       if e.find("method").get("type") == "stability"}
            row["classes"] = "|".join(sorted(classes))
            if not classes:
                row.update(status="skipped", reason="no native-state stability data")
                rows.append(row)
                continue
            construct = next(r.iter("sequence")).text.strip()
            cif = fetch(f"{RCSB}/{row['pdb'].upper()}.cif", root / "downloads" / f"{row['pdb']}.cif")
            hit = resolved_chain(cif, construct)
            if hit is None:
                row.update(status="skipped", reason="no chain whose resolved sequence lies within the HDX construct")
                rows.append(row)
                continue
            chain, seq, offset = hit
            out = root / "structures" / f"{eid}.cif"
            write_resolved(cif, chain, out)
            h = hashlib.sha1(seq.encode()).hexdigest()[:12]
            if not (root / "msa" / f"{h}.csv").exists():
                compute_msa(data={h: seq}, target_id=h, msa_dir=root / "msa",
                            msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy")
                time.sleep(2)  # one request at a time, gently, on a shared public service
            msa = root / "msa" / f"{h}.csv"
            (root / "configs" / f"{eid}.yaml").write_text(
                f"version: 1\nsequences:\n  - protein:\n      id: {chain}\n      sequence: {seq}\n"
                f"      msa: {msa}\n\ninitial_structure:\n  cif: {out}\n")
            row.update(status="ready", chain=chain, resolved=len(seq), offset=offset,
                       msa_depth=sum(1 for _ in open(msa)) - 1)
        except Exception as e:  # recorded per entry; one bad entry must not stop the rest
            row.update(status="skipped", reason=f"error: {e}"[:200])
        rows.append(row)
        print(f"{eid} {row.get('pdb', '?')}: {row['status']} {row.get('reason', '')}")

    fields = ["entry", "name", "pdb", "length", "classes", "status", "reason", "chain", "resolved", "offset", "msa_depth"]
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ready = [r for r in rows if r["status"] == "ready"]
    print(f"\n{len(ready)} of {len(rows)} entries ready -> {root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
