"""
GMSimA V3 — interaction-matrix configuration (EDITABLE).

The V3 design principle: the science stays the same; the assumptions
about WHO BINDS WHAT are the user's choice. Anchor presets (the peptide
side), membrane partner kinds (the membrane side) and the automatic
pairing live HERE, not inside the algorithms. A lab can add or edit
presets in this file and every entry appears in the GUI dropdowns
automatically — the analyses never need touching.

Anchor preset values are ordinary MDAnalysis selection strings applied
WITHIN the resolved protein group. Partner kinds are symbolic names
resolved family-aware (CHARMM36 / AMBER Lipid14-17-21) by
core.lipids.partner_selection().
"""

# ----------------------------------------------------------------------
# Peptide side: anchor presets. No chemistry is privileged — 'basic' is
# simply one entry among equals (it remains the AUTO pairing partner
# default for backward compatibility with published GMSimA results).
# ----------------------------------------------------------------------
# AMBER GROMACS ports name terminal residues NXXX/CXXX (NTYR, CARG,
# ...): every preset lists those variants too, so a terminal anchor
# residue is never silently dropped (v2.0.4 missed e.g. the C-terminal
# Arg of a TAT peptide for exactly this reason).
def _v(*resnames):
    out = []
    for rn in resnames:
        out += [rn, f"N{rn}", f"C{rn}"]
    return " ".join(out)


ANCHOR_PRESETS = {
    "Basic (Lys/Arg)":
        (f"(resname {_v('LYS')} and name NZ) or "
         f"(resname {_v('ARG')} and (name NH1 NH2))"),
    "Acidic (Asp/Glu)":
        (f"(resname {_v('ASP')} and name OD1 OD2) or "
         f"(resname {_v('GLU')} and name OE1 OE2)"),
    "Aromatic rings":
        (f"resname {_v('PHE', 'TYR', 'TRP', 'HIS', 'HID', 'HIE', 'HIP')} "
         "HSD HSE HSP and name CG CD1 CD2 "
         "CE1 CE2 CE3 CZ CZ2 CZ3 CH2 NE1 ND1 NE2 OH"),
    "Hydrophobic side chains":
        (f"resname {_v('LEU', 'ILE', 'VAL', 'ALA', 'MET', 'PHE', 'TRP', 'PRO')} "
         "and not backbone and not name H*"),
    "Polar (Ser/Thr/Asn/Gln/Tyr)":
        (f"resname {_v('SER', 'THR', 'ASN', 'GLN', 'TYR')} "
         "and not backbone and not name H*"),
    "All heavy atoms": "not name H*",
}

# Short aliases kept for the API / older scripts (lowercase keys).
ANCHOR_ALIASES = {
    "basic": "Basic (Lys/Arg)",
    "acidic": "Acidic (Asp/Glu)",
    "aromatic": "Aromatic rings",
    "hydrophobic": "Hydrophobic side chains",
    "polar": "Polar (Ser/Thr/Asn/Gln/Tyr)",
    "all": "All heavy atoms",
}

# ----------------------------------------------------------------------
# Membrane side: partner kinds, resolved by core.lipids.partner_selection.
# ----------------------------------------------------------------------
PARTNER_KINDS = (
    "Auto",                        # picked from AUTO_PARTNER below
    "Whole lipids (heavy atoms)",
    "Phosphate groups",
    "Headgroup + glycerol",
    "Acyl chains",
    "Sterol hydroxyls",
)

# 'Auto' pairing: which partner a preset uses when the user leaves the
# membrane side on Auto. Electrostatic anchors pair with phosphates
# (the historical default); everything else pairs with whole lipids,
# because chain insertion is invisible to a phosphate-only definition.
AUTO_PARTNER = {
    "Basic (Lys/Arg)": "Phosphate groups",
    "Acidic (Asp/Glu)": "Headgroup + glycerol",
}
AUTO_PARTNER_DEFAULT = "Whole lipids (heavy atoms)"


def resolve_anchor(choice):
    """Preset display name or alias -> selection string; custom text
    is returned verbatim. Returns (selection, display_name)."""
    if choice in ANCHOR_PRESETS:
        return ANCHOR_PRESETS[choice], choice
    key = str(choice).strip().lower()
    if key in ANCHOR_ALIASES:
        name = ANCHOR_ALIASES[key]
        return ANCHOR_PRESETS[name], name
    return choice, f"custom ({choice})"


def auto_partner_for(anchor_display_name):
    """Partner kind used when the membrane side is set to Auto."""
    return AUTO_PARTNER.get(anchor_display_name, AUTO_PARTNER_DEFAULT)
