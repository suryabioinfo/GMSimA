"""
Lipid registry and group-suggestion helpers for GMSimA.

The registry maps residue names to per-species metadata so analyses can
generalize beyond the original PC/PS study without hardcoded selections.
Naming follows the CHARMM36 convention, where — note carefully —

    C2x atoms belong to the *sn-2* acyl chain (ester at glycerol C2),
    C3x atoms belong to the *sn-1* acyl chain (ester at glycerol C3).

(The original GMSimA SCD code had these two swapped; every consumer of
chain identity should now go through this registry.)

To support a new lipid, add one entry to LIPID_REGISTRY — no analysis
code needs to change.
"""

# ----------------------------------------------------------------------
# Registry
#
# phosphate  : atom name(s) of the phosphorus used for leaflet/thickness
#              work (None for sterols and other phosphate-free species)
# headgroup  : representative headgroup atom names (selection fragment)
# sn1_prefix : atom-name prefix of the sn-1 chain carbons (CHARMM36: C3x)
# sn2_prefix : atom-name prefix of the sn-2 chain carbons (CHARMM36: C2x)
# charge     : nominal net charge of the lipid at pH 7
# tail_unsat : (sn1, sn2) count of C=C double bonds, informational
# ----------------------------------------------------------------------

LIPID_REGISTRY = {
    # --- Phosphatidylcholine ---
    "POPC": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 1)},
    "DOPC": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (1, 1)},
    "DPPC": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 0)},
    "DMPC": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 0)},
    "DLPC": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 0)},
    "DSPC": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 0)},
    # --- Phosphatidylserine (anionic) ---
    "POPS": {"phosphate": "P", "headgroup": "N P C13", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (0, 1)},
    "DOPS": {"phosphate": "P", "headgroup": "N P C13", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (1, 1)},
    # --- Phosphatidylethanolamine ---
    "POPE": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 1)},
    "DOPE": {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (1, 1)},
    # --- Phosphatidylglycerol (anionic) ---
    "POPG": {"phosphate": "P", "headgroup": "P C13", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (0, 1)},
    "DOPG": {"phosphate": "P", "headgroup": "P C13", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (1, 1)},
    # --- Phosphatidic acid (anionic) ---
    "POPA": {"phosphate": "P", "headgroup": "P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (0, 1)},
    # --- Phosphatidylinositol (anionic) ---
    "POPI": {"phosphate": "P", "headgroup": "P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (0, 1)},
    "SAPI": {"phosphate": "P", "headgroup": "P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": -1, "tail_unsat": (0, 4)},
    # --- Sphingomyelin (chain naming differs: use fatty-acid chain only) ---
    "PSM":  {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 0)},
    "SSM":  {"phosphate": "P", "headgroup": "N P", "sn1_prefix": "C3",
             "sn2_prefix": "C2", "charge": 0, "tail_unsat": (0, 0)},
    # --- Sterols (no phosphate, no glycerol chains) ---
    "CHL1": {"phosphate": None, "headgroup": "O3", "sn1_prefix": None,
             "sn2_prefix": None, "charge": 0, "tail_unsat": (0, 0)},
    "ERG":  {"phosphate": None, "headgroup": "O3", "sn1_prefix": None,
             "sn2_prefix": None, "charge": 0, "tail_unsat": (0, 0)},
}

# Residue names that are never lipids (used when flagging unknowns).
_NON_LIPID_RESNAMES = {
    "SOL", "TIP3", "TIP4", "TIP5", "SPC", "SPCE", "HOH", "WAT",
    "NA", "CL", "K", "MG", "CA", "ZN", "SOD", "CLA", "POT", "MGA", "CAL",
    "NA+", "CL-", "K+",
}

# ----------------------------------------------------------------------
# AMBER lipid family (Lipid14/17/21)
#
# AMBER splits every phospholipid into THREE residues sharing one
# molecule: sn-1 acyl chain, head group, sn-2 acyl chain (in that
# residue order). The head residue carries the phosphorus (named P31)
# and defines the species identity; the chain residues name the tails.
# ----------------------------------------------------------------------

AMBER_HEADS = {           # head resname -> nominal charge at pH 7
    "PC": 0,   # phosphatidylcholine
    "PE": 0,   # phosphatidylethanolamine
    "PS": -1,  # phosphatidylserine
    "PGR": -1, # phosphatidylglycerol (R)
    "PGS": -1, # phosphatidylglycerol (S)
    "PH-": -1, # phosphatidic acid (deprotonated)
    "H2-": -1,
    "PI": -1,  # phosphatidylinositol
    "SPM": 0,  # sphingomyelin
}

AMBER_CHAINS = {          # acyl-chain resnames (tails)
    "PA",   # palmitoyl 16:0
    "OL",   # oleoyl 18:1
    "MY",   # myristoyl 14:0
    "ST",   # stearoyl 18:0
    "LAL",  # lauroyl 12:0
    "LIN",  # linoleoyl 18:2
    "AR",   # arachidonoyl 20:4
    "DHA",  # docosahexaenoyl 22:6
    "SA",   # sphingosine backbone (SPM)
}

AMBER_STEROLS = {"CHL"}   # AMBER cholesterol

AMBER_PHOSPHATE = "P31"


def detect_ff_family(universe):
    """
    'amber'   — AMBER Lipid14/17/21-style topology (3-residue lipids,
                head residues from AMBER_HEADS carrying atom P31)
    'charmm'  — CHARMM36-style one-residue lipids from LIPID_REGISTRY
    'unknown' — neither convention recognized
    """
    resnames = set(universe.residues.resnames)
    if resnames & set(AMBER_HEADS):
        heads = " ".join(sorted(resnames & set(AMBER_HEADS)))
        if len(universe.select_atoms(
                f"resname {heads} and name {AMBER_PHOSPHATE}")):
            return "amber"
    if resnames & set(LIPID_REGISTRY):
        return "charmm"
    return "unknown"


def _amber_lipid_resnames(universe):
    resnames = set(universe.residues.resnames)
    heads = sorted(resnames & set(AMBER_HEADS))
    chains = sorted(resnames & AMBER_CHAINS)
    sterols = sorted(resnames & AMBER_STEROLS)
    return heads, chains, sterols


def detect_lipids(universe):
    """
    Return {species: n_lipids} for every recognized lipid present.

    CHARMM: species = the lipid resname (one residue per lipid).
    AMBER : species = the HEAD residue name (PC, PS, ...); one head
            residue per lipid, so counting heads counts lipids.
    """
    if detect_ff_family(universe) == "amber":
        heads, _, sterols = _amber_lipid_resnames(universe)
        found = {}
        for resname in heads + sterols:
            n = len(universe.select_atoms(f"resname {resname}").residues)
            if n:
                found[resname] = n
        return found

    found = {}
    for resname in sorted(set(universe.residues.resnames)):
        if resname in LIPID_REGISTRY:
            n = len(universe.select_atoms(f"resname {resname}").residues)
            if n:
                found[resname] = n
    return found


def detect_unknown_lipid_like(universe):
    """
    Residue names that look lipid-like (contain a P atom, are not water/
    ion/protein, not in the registry). Returned so callers can WARN the
    user that these species are not covered by lipid-aware analyses.
    """
    unknown = []
    protein_resnames = set(universe.select_atoms("protein").residues.resnames)
    amber_known = set(AMBER_HEADS) | AMBER_CHAINS | AMBER_STEROLS
    for resname in sorted(set(universe.residues.resnames)):
        if (resname in LIPID_REGISTRY or resname in _NON_LIPID_RESNAMES
                or resname in protein_resnames or resname in amber_known):
            continue
        ag = universe.select_atoms(f"resname {resname}")
        if len(ag) and len(ag.select_atoms("name P P8 P11")) > 0:
            unknown.append(resname)
    return unknown


def membrane_selection(universe):
    """
    Selection string covering all recognized lipid atoms.

    CHARMM: 'resname POPC POPS CHL1'.
    AMBER : heads + chains + sterols, e.g. 'resname PC PS PA OL CHL',
            so the whole 3-residue lipids are included.
    Empty string if nothing is recognized.
    """
    if detect_ff_family(universe) == "amber":
        heads, chains, sterols = _amber_lipid_resnames(universe)
        names = heads + chains + sterols
        return ("resname " + " ".join(names)) if names else ""

    lipids = detect_lipids(universe)
    if not lipids:
        return ""
    return "resname " + " ".join(sorted(lipids))


def phosphate_selection(universe, base_selection=None):
    """
    Selection string for lipid phosphorus atoms, restricted to
    base_selection when given. CHARMM registry names (usually 'P');
    AMBER head-residue 'P31'. Falls back to 'name P' so systems with
    unregistered lipids still work.
    """
    if detect_ff_family(universe) == "amber":
        heads, _, _ = _amber_lipid_resnames(universe)
        core = (f"resname {' '.join(heads)} and name {AMBER_PHOSPHATE}"
                if heads else f"name {AMBER_PHOSPHATE}")
    else:
        lipids = detect_lipids(universe)
        names = sorted({meta["phosphate"]
                        for res, meta in LIPID_REGISTRY.items()
                        if res in lipids and meta["phosphate"]})
        core = f"name {' '.join(names)}" if names else "name P"
    if base_selection:
        return f"({base_selection}) and {core}"
    return core


def anionic_selection(universe, base_selection=None):
    """
    Selection string for the phosphorus atoms of anionic lipids present
    (PS/PG/PA/PI...). Empty string when the system has none.
    """
    if detect_ff_family(universe) == "amber":
        heads, _, _ = _amber_lipid_resnames(universe)
        anionic = sorted(h for h in heads if AMBER_HEADS.get(h, 0) < 0)
        if not anionic:
            return ""
        sel = f"resname {' '.join(anionic)} and name {AMBER_PHOSPHATE}"
    else:
        lipids = detect_lipids(universe)
        anionic = sorted(res for res in lipids
                         if LIPID_REGISTRY[res]["charge"] < 0
                         and LIPID_REGISTRY[res]["phosphate"])
        if not anionic:
            return ""
        sel = f"resname {' '.join(anionic)} and name P*"
    if base_selection:
        return f"({base_selection}) and {sel}"
    return sel


def chain_prefixes(resname):
    """
    (sn1_prefix, sn2_prefix) for a registry lipid; ('C3', 'C2') CHARMM36
    default for unknown species; (None, None) for sterols.
    """
    meta = LIPID_REGISTRY.get(resname)
    if meta is None:
        return ("C3", "C2")
    return (meta["sn1_prefix"], meta["sn2_prefix"])


# ----------------------------------------------------------------------
# V3 interaction matrix: membrane-side partner resolution
# ----------------------------------------------------------------------

_STEROL_NAMES = "CHL1 CHOL CHL ERG"
# CHARMM acyl-chain carbons: C2x (sn-2) / C3x (sn-1) naming, with the
# glycerol C1/C2/C3 excluded by exact name.
_CHARMM_CHAIN_ATOMS = "name C2* C3* and not name C1 C2 C3"

_PARTNER_KIND_MAP = {
    "whole": "whole", "whole lipids (heavy atoms)": "whole",
    "phosphates": "phosphates", "phosphate groups": "phosphates",
    "headgly": "headgly", "headgroup + glycerol": "headgly",
    "chains": "chains", "acyl chains": "chains",
    "sterol_oh": "sterol_oh", "sterol hydroxyls": "sterol_oh",
}


def partner_selection(universe, kind, species=None):
    """
    Membrane-side selection string for the V3 interaction matrix,
    resolved family-aware (heavy atoms only).

    kind    : 'whole' | 'phosphates' | 'headgly' | 'chains' |
              'sterol_oh' (the display names in
              config.interactions.PARTNER_KINDS are also accepted).
    species : None / 'all' for every detected lipid, or a list of
              species names (CHARMM lipid resnames; AMBER head or
              sterol resnames) restricting the partner to those lipids.

    Returns (selection_string, description). The selection may match
    zero atoms for impossible combinations (e.g. sterol hydroxyls in a
    sterol-free membrane) — the caller should check and report.
    """
    k = _PARTNER_KIND_MAP.get(str(kind).strip().lower())
    if k is None:
        raise ValueError(f"Unknown membrane partner kind: {kind!r}")

    if species in (None, "all", "All", "All species"):
        sp = None
    else:
        sp = [species] if isinstance(species, str) else list(species)

    fam = detect_ff_family(universe)
    memb = membrane_selection(universe)

    def _whole_scope():
        """Whole-lipid scope, species-filtered when requested."""
        if sp is None:
            return memb
        if fam == "amber":
            return f"same fragment as resname {' '.join(sp)}"
        return f"resname {' '.join(sp)}"

    sp_txt = "" if sp is None else f" of {'/'.join(sp)}"

    if k == "whole":
        sel = f"({_whole_scope()}) and not name H*"
        desc = f"whole lipids (heavy atoms){sp_txt}"

    elif k == "phosphates":
        core = phosphate_selection(universe)
        sel = (f"({core}) and resname {' '.join(sp)}" if sp else core)
        desc = f"lipid phosphates{sp_txt}"

    elif k == "headgly":
        if fam == "amber":
            heads, _, _ = _amber_lipid_resnames(universe)
            use = [h for h in (sp or heads) if h in heads]
            sel = (f"resname {' '.join(use)} and not name H*"
                   if use else "resname NONE_")
        else:
            base = _whole_scope()
            sel = (f"({base}) and not name H* and not resname "
                   f"{_STEROL_NAMES} and not ({_CHARMM_CHAIN_ATOMS})")
        desc = f"headgroup + glycerol region{sp_txt}"

    elif k == "chains":
        if fam == "amber":
            _, chains, _ = _amber_lipid_resnames(universe)
            core = (f"resname {' '.join(chains)} and not name H*"
                    if chains else "resname NONE_")
            sel = (f"(same fragment as resname {' '.join(sp)}) and ({core})"
                   if sp else core)
        else:
            base = _whole_scope()
            sel = (f"({base}) and ({_CHARMM_CHAIN_ATOMS}) "
                   f"and not resname {_STEROL_NAMES}")
        desc = f"acyl chains{sp_txt}"

    else:  # sterol_oh
        base = (f"resname {' '.join(sp)}" if sp
                else f"resname {_STEROL_NAMES}")
        sel = f"({base}) and resname {_STEROL_NAMES} and name O*"
        desc = f"sterol hydroxyl oxygens{sp_txt}"

    return sel, desc


# ----------------------------------------------------------------------
# GUI group suggestions
# ----------------------------------------------------------------------

def suggest_groups(universe):
    """
    Build the dropdown contents for the GUI group selectors.

    Returns
    -------
    dict with:
      protein_options  : list of MDAnalysis selection strings (best first)
      membrane_options : list of MDAnalysis selection strings (best first)
      lipids           : {resname: count}
      unknown_lipids   : [resname, ...] lipid-like but unregistered
      protein_present  : bool
    """
    protein = universe.select_atoms("protein")
    protein_options = []
    if len(protein) > 0:
        protein_options.append("protein")
        segids = sorted({s for s in protein.segments.segids if s.strip()})
        if len(segids) > 1:
            protein_options.extend(f"protein and segid {s}" for s in segids)

    lipids = detect_lipids(universe)
    membrane_options = []
    if lipids:
        membrane_options.append(membrane_selection(universe))
        if len(lipids) > 1:
            if detect_ff_family(universe) == "amber":
                # a species' whole 3-residue lipids = every fragment
                # containing that head residue
                membrane_options.extend(
                    f"same fragment as resname {r}" for r in sorted(lipids))
            else:
                membrane_options.extend(
                    f"resname {r}" for r in sorted(lipids))

    return {
        "protein_options": protein_options,
        "membrane_options": membrane_options,
        "lipids": lipids,
        "unknown_lipids": detect_unknown_lipid_like(universe),
        "protein_present": len(protein) > 0,
        "ff_family": detect_ff_family(universe),
    }
