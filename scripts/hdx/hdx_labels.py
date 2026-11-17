"""Start2Fold HDX labels, one experiment at a time.

A Start2Fold entry can hold several independent experiments, each defining its
own protection tiers on its own scale: e.g. native exchange (STRONG: P > 50000)
alongside exchange in a partially folded state (STRONG: P > 30), or two
pulse-labelling studies with different timings. Their labels must not be
merged: a residue's class only means something within the experiment that
assigned it.

An *experiment set* is the tiers that share a method, conditions and
publication. Residue indices refer to that experiment's own sequence, which can
differ from other experiments in the entry (STF0025 numbers its stability data
by UniProt position), so indices are mapped onto the model through the set's
sequence.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict

PARTIALLY_FOLDED = "partially folded"


def experiment_sets(entry_xml, kind, native_only=True):
    """[(name, sequence, {index: level})] for experiments of the given kind.

    kind: "stability" (STRONG / MEDIUM / WEAK) or "folding" (EARLY / INTERMEDIATE / LATE).
    native_only: for stability, drop exchange measured in partially folded states.
    """
    sets = defaultdict(lambda: {"levels": {}, "sequence": None})
    for ex in ET.parse(entry_xml).getroot().iter("experiment"):
        method = ex.find("method")
        if method.get("type") != kind:
            continue
        if kind == "stability" and native_only and PARTIALLY_FOLDED in (method.text or "").lower():
            continue
        cond = ex.find("conditions")
        pub = ex.find("publication")
        key = (method.text, cond.get("pH"), cond.get("temperature"), pub.get("pmid") if pub is not None else None)
        s = sets[key]
        s["sequence"] = ex.find("sequence").text.strip()
        level = ex.find("protection").get("protection_level")
        for r in ex.findall("residue"):
            s["levels"][int(r.get("index"))] = level
    return [(" / ".join(str(k) for k in key if k), s["sequence"], s["levels"]) for key, s in sets.items()]


def labels_for_model(sets, model_sequence):
    """Per set: {model residue (1-based): level}, using each set's own sequence. Sets that do
    not contain the model sequence are dropped."""
    out = []
    for name, seq, levels in sets:
        offset = seq.find(model_sequence)
        if offset < 0:
            continue
        mapped = {i - offset: lv for i, lv in levels.items() if 1 <= i - offset <= len(model_sequence)}
        if mapped:
            out.append((name, mapped))
    return out
