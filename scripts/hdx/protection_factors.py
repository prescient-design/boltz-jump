#!/usr/bin/env python3
"""Predict amide protection factors from a structural ensemble and compare with HDX.

Per-residue protection uses the Best-Vendruscolo phenomenological model
(Best & Vendruscolo, Structure 2006), averaged over frames:

    ln P_i = beta_c * N_c,i + beta_h * N_h,i,   beta_c = 0.35, beta_h = 2.0

N_c,i counts heavy atoms within 6.5 A of residue i's amide nitrogen, excluding
residues i-2..i+2. N_h,i is 1 if the amide N-H donates a hydrogen bond.

Boltz-2 structures carry no hydrogens. The amide hydrogen is placed from the
backbone, as DSSP does: the peptide bond is planar, so N-H lies in the plane of
C(i-1), N(i) and CA(i), along the outward bisector of the C(i-1)-N-CA angle, at
1.01 A. A hydrogen bond (default ``--hbond hydrogen``) then needs H...O < 2.5 A
and an N-H...O angle > 120 degrees (Baker-Hubbard), to any backbone carbonyl
oxygen or side-chain oxygen acceptor (Asp, Glu, Asn, Gln, Ser, Thr, Tyr) of
another residue, excluding the preceding residue's own carbonyl.
``--hbond heavy`` keeps the cruder heavy-atom criterion, N(i)...O(j) < 3.5 A to
backbone carbonyls with |i - j| >= 3, for comparison. Prolines and the first
residue have no amide hydrogen and are excluded.

Experimental data come from Start2Fold (Pancsa et al., NAR 2016) entry XML.
Native-exchange experiments label amides STRONG or MEDIUM protection. The XML
does not list which other amides were probed, so an unlabelled residue may be
fast-exchanging or simply unmeasured; only labelled residues are compared. The
test is the AUC for STRONG above MEDIUM (equivalently a rank test on the two
levels). The secondary check is protected (STRONG or MEDIUM) above the rest:
above WEAK residues where the entry reports them (probed, little protection),
otherwise above unlabelled residues, a weaker comparison because that set mixes
fast-exchanging and unmeasured amides.

Usage:
    python scripts/hdx/protection_factors.py --entry STF0040.xml \
        --trajectory traj.xtc --topology model_0.cif --out protection.csv
"""

import argparse
import xml.etree.ElementTree as ET

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BETA_C, BETA_H = 0.35, 2.0
CONTACT_CUTOFF_NM, HBOND_CUTOFF_NM = 0.65, 0.35
NH_BOND_NM = 0.101
HO_CUTOFF_NM, NHO_MIN_ANGLE_DEG = 0.25, 120.0
SIDECHAIN_ACCEPTORS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "ASN": ("OD1",), "GLN": ("OE1",),
                       "SER": ("OG",), "THR": ("OG1",), "TYR": ("OH",)}


def amide_hydrogen(xyz, c_prev, n, ca):
    """Amide H position(s) in nm: outward bisector of C(i-1)-N-CA, in the peptide plane."""
    u = xyz[:, n] - xyz[:, c_prev]
    v = xyz[:, n] - xyz[:, ca]
    b = u / np.linalg.norm(u, axis=-1, keepdims=True) + v / np.linalg.norm(v, axis=-1, keepdims=True)
    return xyz[:, n] + NH_BOND_NM * b / np.linalg.norm(b, axis=-1, keepdims=True)


def hbond_hydrogen(xyz, n, h, acceptors):
    """1 per frame if N-H donates a hydrogen bond to any acceptor (Baker-Hubbard)."""
    o = xyz[:, acceptors]                                  # frames x acceptors x 3
    ho = o - h[:, None]
    hn = (xyz[:, n] - h)[:, None]
    d_ho = np.linalg.norm(ho, axis=-1)
    cos = (ho * hn).sum(-1) / (d_ho * np.linalg.norm(hn, axis=-1) + 1e-12)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))    # angle at H between H->N and H->O
    return ((d_ho < HO_CUTOFF_NM) & (angle > NHO_MIN_ANGLE_DEG)).any(1).astype(int)


def hdx_classes(entry_xml, kind="stability"):
    """Residue index (1-based, as in the entry sequence) -> protection level, plus probed set.

    kind="stability": native-exchange classes (STRONG / MEDIUM / WEAK).
    kind="folding": pulse-labelling / quenched-flow classes (EARLY / INTERMEDIATE / LATE).
    """
    root = ET.parse(entry_xml).getroot()
    levels, probed_counts = {}, []
    for exp in root.iter("experiment"):
        if exp.find("method").get("type") != kind:
            continue
        level = exp.find("protection").get("protection_level")
        for r in exp.findall("residue"):
            levels[int(r.get("index"))] = level
        probes = exp.find("conditions").get("probes")
        probed_counts.append(probes)
    sequence = next(root.iter("sequence")).text.strip()
    return levels, sequence, probed_counts


def predict(traj, hbond="hydrogen"):
    """Per-residue ensemble-mean ln P (and its two components), keyed by residue index."""
    top = traj.topology
    heavy = np.array([a.index for a in top.atoms if a.element.symbol != "H"])
    res_of = np.array([top.atom(i).residue.index for i in heavy])
    carbonyl_o = {a.residue.index: a.index for a in top.atoms if a.name == "O" and a.residue.is_protein}
    sidechain_o = [(a.residue.index, a.index) for a in top.atoms
                   if a.residue.is_protein and a.name in SIDECHAIN_ACCEPTORS.get(a.residue.name, ())]
    atom = {(a.residue.index, a.name): a.index for a in top.atoms}
    rows = []
    for res in top.residues:
        if not res.is_protein or res.name == "PRO" or res.index == 0:
            continue
        n, ca = atom.get((res.index, "N")), atom.get((res.index, "CA"))
        c_prev = atom.get((res.index - 1, "C"))
        if n is None or ca is None or c_prev is None:
            continue
        far = heavy[np.abs(res_of - res.index) > 2]
        d = md.compute_distances(traj, np.array([[n, j] for j in far]))
        nc = (d < CONTACT_CUTOFF_NM).sum(1)
        if hbond == "hydrogen":
            acc = [o for ri, o in carbonyl_o.items() if ri not in (res.index, res.index - 1)]
            acc += [o for ri, o in sidechain_o if ri != res.index]
            h = amide_hydrogen(traj.xyz, c_prev, n, ca)
            nh = hbond_hydrogen(traj.xyz, n, h, np.array(acc))
        else:
            acc = [o for ri, o in carbonyl_o.items() if abs(ri - res.index) >= 3]
            dh = md.compute_distances(traj, np.array([[n, o] for o in acc]))
            nh = np.minimum((dh < HBOND_CUTOFF_NM).sum(1), 1)  # one amide H, at most one bond
        lnp = BETA_C * nc + BETA_H * nh
        rows.append({"residue": res.index + 1, "resname": res.name, "lnP": lnp.mean(),
                     "lnP_sd": lnp.std(), "N_c": nc.mean(), "N_h": nh.mean()})
    return pd.DataFrame(rows)


def auc(pos, neg):
    u, p = mannwhitneyu(pos, neg, alternative="greater")
    return u / (len(pos) * len(neg)), p


def compare(pred, levels):
    """AUCs: STRONG above MEDIUM (primary); protected above WEAK, or above unlabelled (secondary)."""
    pred = pred.copy()
    pred["hdx_level"] = pred.residue.map(lambda i: levels.get(i, "unlabelled"))
    lv = pred.hdx_level
    strong, medium = pred[lv == "STRONG"].lnP, pred[lv == "MEDIUM"].lnP
    protected = pred[lv.isin(["STRONG", "MEDIUM"])].lnP
    weak, unlabelled = pred[lv == "WEAK"].lnP, pred[lv == "unlabelled"].lnP
    negative, versus = (weak, "WEAK") if len(weak) else (unlabelled, "unlabelled")
    nan = (float("nan"), float("nan"))
    a1, p1 = auc(strong, medium) if len(strong) and len(medium) else nan
    a2, p2 = auc(protected, negative) if len(protected) and len(negative) else nan
    return pred, {"auc_strong_vs_medium": a1, "p1": p1, "auc_protected_vs_rest": a2, "p2": p2,
                  "secondary_versus": versus, "n_strong": len(strong), "n_medium": len(medium),
                  "n_weak": len(weak), "n_unlabelled": len(unlabelled)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--trajectory", help="omit to score the topology structure alone")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window", default="0:1",
                        help="fraction of the trajectory to score, start:end (e.g. 0:0.5 = first half)")
    parser.add_argument("--classes", choices=["stability", "folding"], default="stability")
    parser.add_argument("--hbond", choices=["hydrogen", "heavy"], default="hydrogen",
                        help="hydrogen: place amide H and use distance + angle (default); heavy: N...O distance only")
    parser.add_argument("--out")
    args = parser.parse_args()

    traj = (md.load(args.trajectory, top=args.topology, stride=args.stride)
            if args.trajectory else md.load(args.topology))
    traj = traj.atom_slice(traj.topology.select("protein"))
    a, b = (float(x) for x in args.window.split(":"))
    traj = traj[int(a * traj.n_frames):max(int(b * traj.n_frames), int(a * traj.n_frames) + 1)]
    levels, sequence, probes = hdx_classes(args.entry, args.classes)
    model_seq = "".join(r.code or "X" for r in traj.topology.residues if r.is_protein)
    # Crystal structures often lack disordered terminal residues. The model may
    # therefore be an exact substring of the HDX construct; shift its numbering
    # onto the entry's. Anything else is an error rather than a guess.
    offset = sequence.find(model_seq)
    if offset < 0:
        raise ValueError(f"model sequence is not part of the HDX entry:\n  {model_seq}\n  {sequence}")
    if offset or len(model_seq) != len(sequence):
        missing = [i + 1 for i in range(len(sequence)) if not offset <= i < offset + len(model_seq)]
        lost = sorted(i for i in missing if i in levels)
        print(f"model covers HDX residues {offset + 1}-{offset + len(model_seq)}; "
              f"labelled residues outside it: {lost or 'none'}")
    pred = predict(traj, args.hbond)
    pred["residue"] = pred.residue + offset
    pred, stats = compare(pred, levels)
    if args.out:
        pred.to_csv(args.out, index=False)
    print(f"{traj.n_frames} frames; STRONG {stats['n_strong']}, MEDIUM {stats['n_medium']}, WEAK {stats['n_weak']}, "
          f"unlabelled {stats['n_unlabelled']}; probes per experiment {probes}")
    print(f"  AUC STRONG > MEDIUM (primary):          {stats['auc_strong_vs_medium']:.2f} (p={stats['p1']:.2g})")
    print(f"  AUC protected > {stats['secondary_versus']} (secondary): "
          f"{stats['auc_protected_vs_rest']:.2f} (p={stats['p2']:.2g})")


if __name__ == "__main__":
    main()
