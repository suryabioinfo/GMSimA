"""
Binding-site detection by residue-community clustering, with
site-resolved residence kinetics (PyLipID-style; Song et al., JCTC
2022).

Idea: residues that repeatedly contact the SAME lipid at the SAME time
belong to one binding site. The analysis therefore

  1. builds, in one trajectory pass, a dual-cutoff (4.5/7.0 Å
     hysteresis) contact tensor B[t, residue, lipid] between the
     peptide's anchor atoms and whole lipid molecules (each lipid =
     one bonded fragment, so AMBER's 3-residue lipids count once);
  2. forms the residue-residue co-contact matrix
        W[r1, r2] = sum_t sum_l  B[t, r1, l] * B[t, r2, l]
     (how often two residues share a lipid simultaneously);
  3. clusters the residue graph weighted by W with Louvain community
     detection (networkx) — each community is one candidate binding
     site; communities whose site occupancy is negligible are dropped;
  4. computes, per site: occupancy, lipid-species composition of its
     contacts, and a SITE-RESOLVED residence time — the dual-cutoff
     contact series of each lipid with ANY residue of the site, events
     pooled over lipids, fit with the same survival/MLE/AIC/bootstrap
     machinery as the global tau-off — plus a representative bound
     pose (whole-system PDB at the frame of maximal site contact).

Outputs: <prefix>_sites.csv, <prefix>_cocontact.png (community-ordered
heat map), <prefix>_site<k>_pose.pdb, and per-site survival plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import capped_distance

from analysis.tauoff import (CONTACT_CUTOFF, CONTACT_CUTOFF_END,
                             GAP_TOLERANCE_NS, MIN_CONTACT_NS,
                             ANCHOR_PRESETS, _clean_series, extract_events,
                             fit_residence_time, plot_survival)

MIN_SITE_OCCUPANCY = 0.05      # drop communities bound < 5% of frames
EDGE_WEIGHT_FRACTION = 0.10    # keep edges >= 10% of the strongest edge


# ----------------------------------------------------------------------
# Contact tensor
# ----------------------------------------------------------------------

def compute_contact_tensor(universe, protein_sel="protein",
                           membrane_sel=None, anchor_sel="all", stride=1):
    """
    Returns (times, B, res_labels, lipid_species):
      B             : (n_frames, n_res, n_lip) bool, dual-cutoff contacts
      res_labels    : e.g. ['LYS6', ...]
      lipid_species : per-lipid species label (head resname)
    Lipids are bonded fragments of the membrane selection, so the tensor
    is force-field independent (CHARMM 1-residue and AMBER 3-residue
    lipids both count once).
    """
    from core.lipids import (membrane_selection, detect_ff_family,
                             AMBER_HEADS)

    if membrane_sel is None or not membrane_sel.strip():
        membrane_sel = membrane_selection(universe)
    anchor = ANCHOR_PRESETS.get(anchor_sel or "all", anchor_sel)
    anchors = universe.select_atoms(f"({protein_sel}) and ({anchor})")
    memb = universe.select_atoms(f"({membrane_sel}) and not name H*")
    if len(anchors) == 0 or len(memb) == 0:
        return None

    residues = anchors.residues
    n_res = len(residues)
    res_index = {r.resindex: i for i, r in enumerate(residues)}
    atom_to_res = np.array([res_index[a.resindex] for a in anchors.atoms])

    # lipid unit = bonded fragment (AMBER 3-residue lipids count once);
    # without bond information fall back to residues (correct for
    # CHARMM-style one-residue lipids).
    try:
        fragids = memb.atoms.fragindices
    except Exception:
        fragids = memb.atoms.resindices
    uniq_frags, atom_to_lip = np.unique(fragids, return_inverse=True)
    n_lip = len(uniq_frags)

    # species label per lipid: head resname (AMBER) / resname (CHARMM)
    amber = detect_ff_family(universe) == "amber"
    lipid_species = []
    for fid in uniq_frags:
        frag_atoms = memb.atoms[fragids == fid]
        names = list(frag_atoms.residues.resnames)
        if amber:
            head = next((n for n in names if n in AMBER_HEADS), names[0])
            lipid_species.append(head)
        else:
            lipid_species.append(names[0])
    lipid_species = np.array(lipid_species)

    times = []
    B = np.zeros((0, n_res, n_lip), dtype=bool)
    frames = []
    bound = np.zeros((n_res, n_lip), dtype=bool)
    flat_min = np.empty(n_res * n_lip)

    for ts in universe.trajectory[::stride]:
        times.append(ts.time / 1000.0)
        flat_min.fill(np.inf)
        pairs, dists = capped_distance(anchors.positions, memb.positions,
                                       max_cutoff=CONTACT_CUTOFF_END,
                                       box=ts.dimensions,
                                       return_distances=True)
        if len(pairs):
            keys = (atom_to_res[pairs[:, 0]] * n_lip
                    + atom_to_lip[pairs[:, 1]])
            np.minimum.at(flat_min, keys, dists)
        min_rl = flat_min.reshape(n_res, n_lip)
        bound = np.where(bound, min_rl <= CONTACT_CUTOFF_END,
                         min_rl <= CONTACT_CUTOFF)
        frames.append(bound.copy())

    B = np.array(frames, dtype=bool)
    labels = [f"{rn}{ri}" for rn, ri in zip(residues.resnames,
                                            residues.resids)]
    return np.array(times), B, labels, lipid_species


# ----------------------------------------------------------------------
# Community detection
# ----------------------------------------------------------------------

def cocontact_matrix(B):
    """W[r1, r2] = number of (frame, lipid) pairs bound by both."""
    n_frames, n_res, n_lip = B.shape
    W = np.zeros((n_res, n_res))
    for t in range(n_frames):
        Bf = B[t].astype(float)
        W += Bf @ Bf.T
    return W


def detect_communities(W, res_contact_frames, min_frames=1):
    """
    Louvain communities on the co-contact graph. Residues with no
    membrane contact are excluded. Falls back to greedy modularity,
    then to a single community, if networkx/louvain is unavailable.
    Returns a list of residue-index lists.
    """
    active = [i for i in range(W.shape[0])
              if res_contact_frames[i] >= min_frames]
    if not active:
        return []
    if len(active) == 1:
        return [active]

    Wa = W[np.ix_(active, active)]
    w_max = Wa.max()
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(active)
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                w = Wa[a, b]
                if w_max > 0 and w >= EDGE_WEIGHT_FRACTION * w_max:
                    G.add_edge(active[a], active[b], weight=float(w))
        try:
            comms = nx.community.louvain_communities(G, weight="weight",
                                                     seed=0)
        except Exception:
            comms = nx.community.greedy_modularity_communities(
                G, weight="weight")
        return [sorted(c) for c in comms if len(c) > 0]
    except ImportError:
        return [active]        # no networkx: one aggregate site


# ----------------------------------------------------------------------
# Site kinetics
# ----------------------------------------------------------------------

def site_kinetics(times, B, members, gap_ns=GAP_TOLERANCE_NS,
                  min_contact_ns=MIN_CONTACT_NS):
    """
    Per-lipid contact series with ANY member residue of the site,
    events pooled over lipids -> survival fit; site occupancy = fraction
    of frames with at least one lipid bound to the site.
    """
    site_lip = B[:, members, :].any(axis=1)       # (frames, n_lip)
    occupancy = float(site_lip.any(axis=1).mean())

    events_list = []
    n_censored = 0
    for l in range(site_lip.shape[1]):
        col = site_lip[:, l]
        if not col.any():
            continue
        cleaned = _clean_series(times, col, gap_ns, min_contact_ns)
        evs, cens = extract_events(times, cleaned)
        events_list.append(evs)
        if cens:
            n_censored += 1
    events = (np.concatenate(events_list) if events_list
              else np.array([]))
    fit = fit_residence_time(events) if len(events) else None
    return occupancy, events, fit, n_censored, site_lip


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_cocontact_heatmap(W, labels, communities, outfile):
    order = [i for comm in communities for i in comm]
    order += [i for i in range(len(labels)) if i not in order]
    Wo = W[np.ix_(order, order)]
    lab = [labels[i] for i in order]

    n = len(lab)
    fig, ax = plt.subplots(figsize=(max(5, 0.32 * n + 2),
                                    max(4, 0.32 * n + 1.5)))
    im = ax.imshow(Wo, cmap="Greys", aspect="equal")
    # community block boundaries
    edge = 0
    for comm in communities:
        edge += len(comm)
        if edge < n:
            ax.axhline(edge - 0.5, color="red", lw=1)
            ax.axvline(edge - 0.5, color="red", lw=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(lab, rotation=90, fontsize=7,
                       fontfamily="DejaVu Sans")
    ax.set_yticks(range(n))
    ax.set_yticklabels(lab, fontsize=7, fontfamily="DejaVu Sans")
    fig.colorbar(im, ax=ax, label="co-contact count")
    ax.set_title("Residue Co-contact Matrix (sites blocked in red)",
                 fontsize=11, fontweight="bold", fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# Master
# ----------------------------------------------------------------------

def run_binding_site_analysis(universe, protein_sel="protein",
                              membrane_sel=None, anchor_sel="all",
                              stride=1, output_prefix="sites"):
    """
    Full binding-site analysis; returns a status dict (never raises for
    missing groups). Writes sites CSV, co-contact heat map, per-site
    survival plots and representative pose PDBs.
    """
    tensor = compute_contact_tensor(universe, protein_sel, membrane_sel,
                                    anchor_sel, stride)
    if tensor is None:
        return {"status": "skipped",
                "reason": "no anchor or membrane atoms"}
    times, B, labels, lipid_species = tensor
    if len(times) < 2 or not B.any():
        return {"status": "skipped",
                "reason": "no residue-lipid contacts detected"}

    res_contact_frames = B.any(axis=2).sum(axis=0)
    W = cocontact_matrix(B)
    communities = detect_communities(W, res_contact_frames)
    if not communities:
        return {"status": "skipped", "reason": "no contacting residues"}

    n_frames = B.shape[0]
    rows = []
    kept = []
    site_id = 0
    for comm in sorted(communities,
                       key=lambda c: -B[:, c, :].any(axis=(1, 2)).sum()):
        occupancy, events, fit, n_cens, site_lip = site_kinetics(times, B,
                                                                 comm)
        if occupancy < MIN_SITE_OCCUPANCY:
            continue
        kept.append(comm)

        # composition: which lipid species this site contacts (by
        # bound frame-lipid counts)
        counts = site_lip.sum(axis=0)
        comp = {}
        for sp in np.unique(lipid_species):
            c = counts[lipid_species == sp].sum()
            if c:
                comp[sp] = int(c)
        total = sum(comp.values()) or 1
        comp_str = "; ".join(f"{sp}:{100*c/total:.0f}%"
                             for sp, c in sorted(comp.items(),
                                                 key=lambda x: -x[1]))

        # representative pose: frame with most site-bound lipids
        best_local = int(np.argmax(site_lip.sum(axis=1)))
        best_frame = best_local * stride
        pose_file = f"{output_prefix}_site{site_id}_pose.pdb"
        try:
            universe.trajectory[min(best_frame,
                                    len(universe.trajectory) - 1)]
            universe.atoms.write(pose_file)
            with open(pose_file, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f"TITLE     smGROANA BINDING SITE {site_id} "
                        f"residues {'+'.join(labels[i] for i in comm)} "
                        f"frame {best_frame}\n" + content)
        except Exception:
            pose_file = ""

        if fit is not None and len(events):
            plot_survival(events, fit,
                          f"{output_prefix}_site{site_id}_survival.png",
                          censored=n_cens > 0,
                          title=f"Site {site_id} Residence Survival")

        lo, hi = fit["tau_off_ci"] if fit else (float("nan"), float("nan"))
        rows.append({
            "site": site_id,
            "residues": "+".join(labels[i] for i in comm),
            "n_residues": len(comm),
            "occupancy": round(occupancy, 3),
            "n_events": int(len(events)),
            "n_censored_events": int(n_cens),
            "tau_off_ns": round(fit["tau_off"], 3) if fit else float("nan"),
            "tau_off_ci_low": round(lo, 3) if fit else float("nan"),
            "tau_off_ci_high": round(hi, 3) if fit else float("nan"),
            "model": fit["model"] if fit else "",
            "lipid_composition": comp_str,
            # basename: keeps sites.csv portable and byte-identical
            # across runs regardless of the output directory path
            "pose_pdb": (os.path.basename(pose_file)
                         if pose_file else pose_file),
        })
        site_id += 1

    if not rows:
        return {"status": "skipped",
                "reason": f"no site above {100*MIN_SITE_OCCUPANCY:.0f}% "
                          "occupancy"}

    pd.DataFrame(rows).to_csv(f"{output_prefix}.csv", index=False)
    plot_cocontact_heatmap(W, labels, kept, f"{output_prefix}_cocontact.png")

    return {"status": "ok", "sites": rows, "n_frames": n_frames}
