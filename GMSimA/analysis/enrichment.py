"""
Lipid-type enrichment around a bound protein/peptide and per-residue
membrane contact statistics.

Enrichment factor for lipid type i at time t (Corradi et al. convention):

    E_i(t) = ( n_i,shell / n_shell ) / ( N_i / N_total )

where the "shell" holds every lipid whose head atom lies within
`shell_cutoff` of any protein atom (PBC-aware). E > 1 means the protein
locally concentrates that lipid type (e.g. PS around a cationic
peptide); E < 1 means depletion.

Per-residue contacts: for each protein residue, the fraction of analyzed
frames in which any of its atoms is within `contact_cutoff` of any
membrane atom — the standard way to locate the membrane-binding face.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import capped_distance

from core.lipids import (LIPID_REGISTRY, detect_lipids, membrane_selection,
                         detect_ff_family, AMBER_HEADS, AMBER_PHOSPHATE)

SHELL_CUTOFF = 10.0      # Å, protein–lipid-head distance defining the shell
CONTACT_CUTOFF = 4.5     # Å, heavy-duty residue–membrane contact criterion

_BASIC_RESIDUES = {"LYS", "ARG", "HIS", "HSD", "HSE", "HSP"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _head_atom_groups(universe, membrane_sel):
    """
    One representative head atom per lipid, grouped by resname.

    Returns {resname: AtomGroup}. Uses the registry phosphate where the
    species has one, otherwise the first headgroup atom (sterol O3).
    """
    amber = detect_ff_family(universe) == "amber"
    groups = {}
    for resname in sorted(detect_lipids(universe)):
        if amber:
            # species identity = AMBER head residue; sterols use O atoms
            head = (AMBER_PHOSPHATE if resname in AMBER_HEADS else "O*")
        else:
            meta = LIPID_REGISTRY[resname]
            head = meta["phosphate"] or meta["headgroup"].split()[0]
        sel = f"({membrane_sel}) and resname {resname} and name {head}"
        ag = universe.select_atoms(sel)
        if len(ag):
            groups[resname] = ag
    return groups


# ----------------------------------------------------------------------
# Lipid-type enrichment
# ----------------------------------------------------------------------

def calculate_enrichment(universe, protein_sel="protein", membrane_sel=None,
                         stride=1, shell_cutoff=SHELL_CUTOFF):
    protein = universe.select_atoms(protein_sel)
    if membrane_sel is None or not membrane_sel.strip():
        membrane_sel = membrane_selection(universe)

    heads = _head_atom_groups(universe, membrane_sel)
    if len(protein) == 0 or not heads:
        return None

    resnames = sorted(heads)
    total_counts = {r: len(heads[r]) for r in resnames}
    N_total = sum(total_counts.values())

    times = []
    series = {r: [] for r in resnames}

    for ts in universe.trajectory[::stride]:
        times.append(ts.time / 1000.0)
        shell_counts = {}
        for r in resnames:
            pairs = capped_distance(protein.positions, heads[r].positions,
                                    max_cutoff=shell_cutoff,
                                    box=ts.dimensions,
                                    return_distances=False)
            # unique lipids (a lipid contacted by many protein atoms
            # counts once)
            shell_counts[r] = len(np.unique(pairs[:, 1])) if len(pairs) else 0
        n_shell = sum(shell_counts.values())
        for r in resnames:
            if n_shell == 0:
                series[r].append(np.nan)
            else:
                bulk_frac = total_counts[r] / N_total
                series[r].append((shell_counts[r] / n_shell) / bulk_frac)

    return {
        "time": np.array(times),
        "series": {r: np.array(v) for r, v in series.items()},
        "counts": total_counts,
        "shell_cutoff": shell_cutoff,
    }


def write_enrichment_csv(data, outfile):
    cols = {"time_ns": data["time"]}
    for r, v in data["series"].items():
        cols[f"E_{r}"] = v
    pd.DataFrame(cols).to_csv(outfile, index=False)


def plot_enrichment(data, outfile):
    plt.figure(figsize=(6, 4))
    species = sorted(data["series"])
    try:
        from analysis.curvature import species_colors
        colors = species_colors(species)
    except Exception:
        shades = ["black", "gray", "#b0b0b0", "#707070", "#d0d0d0"]
        colors = {r: shades[i % len(shades)] for i, r in enumerate(species)}
    for r in species:
        plt.plot(data["time"], data["series"][r], lw=1.0,
                 color=colors[r], label=r)
    plt.axhline(1.0, ls="--", color="gray", lw=0.9)
    plt.xlim(0, data["time"].max() if len(data["time"]) else 1)
    plt.xlabel("Time (ns)", fontsize=12, fontfamily="DejaVu Sans")
    plt.ylabel("Enrichment factor", fontsize=12, fontfamily="DejaVu Sans")
    plt.title(f"Lipid Enrichment (shell {data['shell_cutoff']:.0f} Å)",
              fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
    plt.legend(loc="upper right", prop={"family": "DejaVu Sans", "size": 9},
               frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# Per-residue membrane contacts
# ----------------------------------------------------------------------

def calculate_residue_contacts(universe, protein_sel="protein",
                               membrane_sel=None, stride=1,
                               contact_cutoff=CONTACT_CUTOFF):
    protein = universe.select_atoms(protein_sel)
    if membrane_sel is None or not membrane_sel.strip():
        membrane_sel = membrane_selection(universe)
    membrane = universe.select_atoms(membrane_sel)
    if len(protein) == 0 or len(membrane) == 0:
        return None

    residues = protein.residues
    # map: position in `protein` atom array -> residue slot
    res_index = {res.resindex: i for i, res in enumerate(residues)}
    atom_to_res = np.array([res_index[a.resindex] for a in protein.atoms])

    contact_frames = np.zeros(len(residues), dtype=int)
    n_frames = 0

    for ts in universe.trajectory[::stride]:
        n_frames += 1
        pairs = capped_distance(protein.positions, membrane.positions,
                                max_cutoff=contact_cutoff,
                                box=ts.dimensions,
                                return_distances=False)
        if len(pairs):
            touched = np.unique(atom_to_res[pairs[:, 0]])
            contact_frames[touched] += 1

    return {
        "resids": residues.resids.copy(),
        "resnames": residues.resnames.copy(),
        "contact_fraction": contact_frames / max(n_frames, 1),
        "n_frames": n_frames,
        "contact_cutoff": contact_cutoff,
    }


def write_residue_contacts_csv(data, outfile):
    pd.DataFrame({
        "resid": data["resids"],
        "resname": data["resnames"],
        "contact_fraction": data["contact_fraction"],
    }).to_csv(outfile, index=False)


def plot_residue_contacts(data, outfile, highlight_resids=None,
                          highlight_label=None):
    """
    V3: the highlighted residues are the USER'S anchor class (passed as
    highlight_resids + a label for the title), not a hardwired chemistry.
    Without an explicit highlight the historical basic-residue coloring
    is kept for backward compatibility.
    """
    n = len(data["resids"])
    plt.figure(figsize=(max(6, 0.28 * n), 4))
    if highlight_resids is not None:
        hi = set(int(r) for r in highlight_resids)
        colors = ["black" if int(ri) in hi else "gray"
                  for ri in data["resids"]]
        note = highlight_label or "anchor residues in black"
    else:
        colors = ["black" if rn in _BASIC_RESIDUES else "gray"
                  for rn in data["resnames"]]
        note = "basic residues in black"
    x = np.arange(n)
    plt.bar(x, data["contact_fraction"], color=colors, edgecolor="black",
            linewidth=0.4)
    labels = [f"{rn}{ri}" for rn, ri in zip(data["resnames"], data["resids"])]
    plt.xticks(x, labels, rotation=90, fontsize=7,
               fontfamily="DejaVu Sans")
    plt.ylim(0, 1.05)
    plt.ylabel("Membrane contact fraction", fontsize=12,
               fontfamily="DejaVu Sans")
    # two-line title: the previous single line overflowed the figure edge
    plt.title(f"Per-residue Membrane Contacts (< {data['contact_cutoff']} Å)\n"
              + note,
              fontsize=11, fontweight="bold", fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# Master
# ----------------------------------------------------------------------

def run_enrichment_analysis(universe, protein_sel="protein",
                            membrane_sel=None, stride=1,
                            output_prefix="enrich",
                            anchor_sel=None, anchor_label=None):
    """
    Runs both analyses; skips gracefully (returns a status dict, never
    raises for missing groups) when the system has no protein or no
    registry lipids. When anchor_sel is given (V3), the residues it
    matches are the ones highlighted in the contact plot.
    """
    enrich = calculate_enrichment(universe, protein_sel, membrane_sel, stride)
    contacts = calculate_residue_contacts(universe, protein_sel,
                                          membrane_sel, stride)

    if enrich is None and contacts is None:
        return {"status": "skipped",
                "reason": "no protein atoms or no registry lipids detected"}

    if enrich is not None:
        write_enrichment_csv(enrich, f"{output_prefix}_lipid_enrichment.csv")
        plot_enrichment(enrich, f"{output_prefix}_lipid_enrichment.png")

    if contacts is not None:
        write_residue_contacts_csv(contacts,
                                   f"{output_prefix}_residue_contacts.csv")
        hi_resids = None
        hi_label = None
        if anchor_sel:
            try:
                hi = universe.select_atoms(
                    f"({protein_sel}) and ({anchor_sel})")
                hi_resids = list(hi.residues.resids)
                hi_label = (f"anchor residues ({anchor_label}) in black"
                            if anchor_label else "anchor residues in black")
            except Exception:
                hi_resids = None
        plot_residue_contacts(contacts,
                              f"{output_prefix}_residue_contacts.png",
                              highlight_resids=hi_resids,
                              highlight_label=hi_label)

    return {"status": "ok", "enrichment": enrich, "contacts": contacts}
