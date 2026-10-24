import numpy as np
import matplotlib.pyplot as plt

from core.lipids import (chain_prefixes, phosphate_selection, LIPID_REGISTRY,
                         detect_ff_family)


def detect_charmm36_tails(residue):
    """
    Return (sn1_atoms, sn2_atoms) for one lipid residue.

    CHARMM36 convention (now taken from the lipid registry):
        sn-1 chain carbons are the C3x atoms,
        sn-2 chain carbons are the C2x atoms.

    NOTE: an earlier version of this function had the two chains
    swapped (C2x labelled sn-1). Chain identity is now resolved through
    core.lipids.LIPID_REGISTRY, which is the single source of truth; the
    old bond-length "unsaturation swap" heuristic was removed because it
    could silently undo a correct assignment.
    """
    sn1_prefix, sn2_prefix = chain_prefixes(residue.resname)
    if sn1_prefix is None:          # sterols: no glycerol chains
        return [], []

    sn1, sn2 = [], []
    for atom in residue.atoms:
        name = atom.name
        for prefix, chain in ((sn1_prefix, sn1), (sn2_prefix, sn2)):
            if name.startswith(prefix) and name[len(prefix):].isdigit():
                num = int(name[len(prefix):])
                # carbon 1 is the ester carbonyl (C21/C31): exclude it
                if 2 <= num <= 22:
                    chain.append(atom)
                break

    sn1 = sorted(sn1, key=lambda a: int(a.name[2:]))
    sn2 = sorted(sn2, key=lambda a: int(a.name[2:]))
    return sn1, sn2


# ------------------------------------------------------------------
# C–H pair table (built once, used every frame)
# ------------------------------------------------------------------

def _build_ch_pairs(lipid_residues):
    """
    For every chain carbon of every lipid, find its bonded hydrogens.

    Returns {("sn1"|"sn2", carbon_number): (c_idx array, h_idx array)}
    of global atom indices, or None when the topology carries no usable
    C–H bonds (no bond info, or a united-atom force field).
    """
    pairs = {}
    found_any = False
    for res in lipid_residues:
        sn1_atoms, sn2_atoms = detect_charmm36_tails(res)
        for tag, chain in (("sn1", sn1_atoms), ("sn2", sn2_atoms)):
            for carbon in chain:
                num = int(carbon.name[2:])
                try:
                    bonded = carbon.bonded_atoms
                except Exception:
                    return None       # topology without bond information
                for partner in bonded:
                    is_h = (partner.name.startswith("H")
                            or getattr(partner, "mass", 99.0) < 1.2)
                    if is_h:
                        key = (res.resname, tag, num)
                        pairs.setdefault(key, ([], []))
                        pairs[key][0].append(carbon.index)
                        pairs[key][1].append(partner.index)
                        found_any = True
    if not found_any:
        return None
    return {k: (np.array(c), np.array(h)) for k, (c, h) in pairs.items()}


# ------------------------------------------------------------------
# AMBER Lipid14/17/21 support: three-residue lipids (sn-1 chain, head,
# sn-2 chain in residue order). Chain carbons are identified by BOND
# CONNECTIVITY rather than atom-name tables: starting from the ester
# carbonyl carbon (the chain carbon with the most bonded oxygens), the
# linear C-C chain is walked to the terminal methyl. This makes the
# analysis independent of the family's atom-naming details.
# ------------------------------------------------------------------

def _walk_chain(residue):
    """
    Ordered acyl-chain carbons of one AMBER chain residue, starting at
    carbon 2 (the carbonyl carbon 1 is excluded, as in the CHARMM
    path). Returns [] when connectivity is unusable.

    Two Lipid14/17/21 residue layouts are handled:
      (a) the ester carbonyl carbon sits INSIDE the chain residue
          (it has bonded oxygens) -> start the walk there and drop it;
      (b) real GROMACS-ported Lipid17 topologies keep both ester
          carbonyls in the HEAD residue, so the chain residue holds
          carbons 2..N and no oxygen at all. Carbon 2 is then the one
          carbon bonded across the residue boundary (to the head's
          carbonyl) -> start there and keep every carbon.
    """
    try:
        carbons = [a for a in residue.atoms if 11.5 < a.mass < 12.5]
        if len(carbons) < 3:
            return []
        cset = {a.index for a in carbons}
        # neighbors among chain carbons, bonded oxygens, and bonds that
        # leave this residue (the ester link back to the head group)
        neigh, n_ox, n_ext = {}, {}, {}
        for c in carbons:
            bonded = c.bonded_atoms
            neigh[c.index] = [b for b in bonded if b.index in cset]
            n_ox[c.index] = sum(1 for b in bonded
                                if 15.5 < getattr(b, "mass", 0) < 16.5)
            n_ext[c.index] = sum(1 for b in bonded
                                 if b.resindex != residue.resindex)
        by_index = {a.index: a for a in carbons}

        if any(n_ox[c.index] > 0 for c in carbons):
            # layout (a): carbonyl carbon in-residue (most oxygens;
            # ties -> chain end); dropped after the walk
            start = max(carbons,
                        key=lambda a: (n_ox[a.index], -len(neigh[a.index])))
            drop_first = True
        else:
            # layout (b): no oxygen in the chain residue — carbon 2 is
            # the carbon with a bond into another residue (the head)
            linked = [c for c in carbons if n_ext[c.index] > 0]
            if len(linked) == 1:
                start = linked[0]
            else:
                # fallback: of the two chain ends, the terminal methyl
                # carries three hydrogens while carbon 2 carries two —
                # pick the end with fewer bonded hydrogens
                ends = [c for c in carbons if len(neigh[c.index]) == 1]
                if not ends:
                    return []

                def _n_h(a):
                    return sum(1 for b in a.bonded_atoms
                               if getattr(b, "mass", 99.0) < 1.2)

                start = min(ends, key=_n_h)
            drop_first = False

        ordered = [start]
        prev = None
        cur = start
        while True:
            nxt = [by_index[b.index] for b in neigh[cur.index]
                   if prev is None or b.index != prev.index]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            ordered.append(cur)
            if len(ordered) > len(carbons):
                return []          # cycle: not a linear chain
        return ordered[1:] if drop_first else ordered
    except Exception:
        return []


def _amber_tails(head_residue):
    """
    (sn1_atoms, sn2_atoms) for one AMBER lipid identified by its head
    residue: the chain residue BEFORE the head (by resid) is sn-1, the
    one AFTER is sn-2 (Lipid14 residue order: sn1-head-sn2).
    """
    try:
        frag_residues = sorted(head_residue.atoms[0].fragment.residues,
                               key=lambda r: r.resid)
    except Exception:
        return [], []
    chains = [r for r in frag_residues if r != head_residue]
    sn1_res = next((r for r in chains
                    if r.resid < head_residue.resid), None)
    sn2_res = next((r for r in chains
                    if r.resid > head_residue.resid), None)
    sn1 = _walk_chain(sn1_res) if sn1_res is not None else []
    sn2 = _walk_chain(sn2_res) if sn2_res is not None else []
    return sn1, sn2


def _build_ch_pairs_amber(lipid_head_residues):
    """C-H pair table for AMBER lipids, keyed (head_resname, tag, num)."""
    pairs = {}
    found_any = False
    for head in lipid_head_residues:
        sn1, sn2 = _amber_tails(head)
        for tag, chain in (("sn1", sn1), ("sn2", sn2)):
            for pos, carbon in enumerate(chain):
                num = pos + 2            # first walked carbon = carbon 2
                try:
                    bonded = carbon.bonded_atoms
                except Exception:
                    return None
                for partner in bonded:
                    if (partner.name.startswith("H")
                            or getattr(partner, "mass", 99.0) < 1.2):
                        key = (head.resname, tag, num)
                        pairs.setdefault(key, ([], []))
                        pairs[key][0].append(carbon.index)
                        pairs[key][1].append(partner.index)
                        found_any = True
    if not found_any:
        return None
    return {k: (np.array(c), np.array(h)) for k, (c, h) in pairs.items()}


# -------------------- SCD calculation -----------------------

def calculate_scd(universe, stride=1):
    """
    Deuterium order parameter per chain carbon.

    Primary method (all-atom topologies): the true NMR definition
        S_CD(i) = < (3 cos^2 theta_CH - 1) / 2 >
    over all C(i)-H bond vectors, directly comparable to experimental
    |S_CD| and to `gmx order`. Requires bonded hydrogens in the tpr.

    Fallback (no bonds / united-atom): the previous chord-vector
    approximation using C(i-1)->C(i+1); this measures a molecular-axis
    order parameter roughly TWICE |S_CD| — the returned dict carries a
    'method' flag so plots and file headers can say which was used.
    """
    lipids = universe.select_atoms(phosphate_selection(universe)).residues

    if detect_ff_family(universe) == "amber":
        # AMBER 3-residue lipids: chains found by connectivity walk;
        # requires bond information (any all-atom .tpr has it).
        ch_pairs = _build_ch_pairs_amber(lipids)
        if ch_pairs is None:
            print("SCD: AMBER lipids without usable bond/H information — "
                  "order parameters skipped.")
            return {"sn1_carbon": np.array([]), "sn1_scd": [],
                    "sn2_carbon": np.array([]), "sn2_scd": [],
                    "method": "C-H"}
    else:
        ch_pairs = _build_ch_pairs(lipids)
    method = "C-H" if ch_pairs else "chord"

    carbon_range = range(2, 23)
    # Weighted sums keyed by (resname, tag, num): the global curve is the
    # pair-count-weighted combination of the per-species curves, so both
    # are consistent by construction.
    sums, counts = {}, {}

    def _accumulate(key, total, n):
        sums[key] = sums.get(key, 0.0) + total
        counts[key] = counts.get(key, 0) + n

    if method == "C-H":
        for ts in universe.trajectory[::stride]:
            normal = detect_membrane_normal(universe)
            pos = universe.atoms.positions
            for (rn, tag, num), (c_idx, h_idx) in ch_pairs.items():
                if num not in carbon_range:
                    continue
                vec = pos[h_idx] - pos[c_idx]
                norm = np.linalg.norm(vec, axis=1)
                ok = norm > 0
                if not np.any(ok):
                    continue
                cos = (vec[ok] @ normal) / norm[ok]
                scd = 0.5 * (3.0 * cos**2 - 1.0)      # true S_CD (negative)
                _accumulate((rn, tag, num), float(np.sum(scd)), int(ok.sum()))
    else:
        # ---------------- chord-vector fallback ----------------
        lipid_tails = []
        for res in lipids:
            sn1_atoms, sn2_atoms = detect_charmm36_tails(res)
            if len(sn1_atoms) > 2 and len(sn2_atoms) > 2:
                lipid_tails.append((res.resname, sn1_atoms, sn2_atoms))

        for ts in universe.trajectory[::stride]:
            normal = detect_membrane_normal(universe)
            for rn, sn1_atoms, sn2_atoms in lipid_tails:
                for tag, chain in (("sn1", sn1_atoms), ("sn2", sn2_atoms)):
                    if len(chain) < 3:
                        continue
                    # loop MUST start at 1: chain[i-1] at i=0 wraps to the
                    # LAST atom (negative indexing) and corrupts C2.
                    for i in range(1, len(chain) - 1):
                        vec = chain[i + 1].position - chain[i - 1].position
                        norm = np.linalg.norm(vec)
                        if norm == 0:
                            continue
                        cos_theta = np.dot(vec / norm, normal)
                        scd = -0.5 * (3 * cos_theta**2 - 1)
                        carbon_index = int(chain[i].name[2:])
                        if carbon_index in carbon_range:
                            _accumulate((rn, tag, carbon_index), scd, 1)

    # -------------------------------------------------
    # Collapse weighted sums -> global and per-species curves
    # -------------------------------------------------
    def collapse(resname=None):
        out = {}
        for tag in ("sn1", "sn2"):
            carbons, values = [], []
            for i in carbon_range:
                if resname is None:
                    tot = sum(sums.get((rn, tag, i), 0.0)
                              for rn in {k[0] for k in sums})
                    n = sum(counts.get((rn, tag, i), 0)
                            for rn in {k[0] for k in counts})
                else:
                    tot = sums.get((resname, tag, i), 0.0)
                    n = counts.get((resname, tag, i), 0)
                if n > 0:
                    carbons.append(i)
                    values.append(tot / n)
            out[f"{tag}_carbon"] = np.array(carbons)
            out[f"{tag}_scd"] = values
        return out

    result = collapse(None)
    result["method"] = method

    species = sorted({k[0] for k in sums})
    if len(species) > 1:
        result["per_species"] = {rn: collapse(rn) for rn in species}

    return result


# ------------------------------------------------------------
# PCA based membrane normal detection
# ------------------------------------------------------------
def detect_membrane_normal(universe):

    # select lipid phosphorus atoms (registry-aware, falls back to name P)
    p_atoms = universe.select_atoms(phosphate_selection(universe))

    coords = p_atoms.positions.copy()
    midplane = coords[:, 2].mean()
    upper_leaflet = coords[coords[:, 2] > midplane]
    if len(upper_leaflet) < 10:
        upper_leaflet = coords  # handle small systems with only 10 lipids

    coords = upper_leaflet - upper_leaflet.mean(axis=0)

    cov = np.cov(coords.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # smallest eigenvalue → membrane normal
    normal = eigvecs[:, np.argmin(eigvals)]

    # normalize
    normal = normal / np.linalg.norm(normal)

    if normal[2] < 0:  # ensure normal points upwards
        normal = -normal

    return normal


# ---------------------------------------------------------
# Plot SCD
# ---------------------------------------------------------

def plot_scd(data, outfile):

    sn1_c = np.array(data["sn1_carbon"], dtype=int)
    sn2_c = np.array(data["sn2_carbon"], dtype=int)

    sn1_v = -np.array(data["sn1_scd"], dtype=float)
    sn2_v = -np.array(data["sn2_scd"], dtype=float)

    if len(sn1_c) == 0 and len(sn2_c) == 0:
        print("SCD calculation returned no values.")
        return

    plt.figure(figsize=(6, 4))

    plt.plot(sn1_c, sn1_v, linestyle="-", linewidth=1, color="black", label="sn-1")
    plt.plot(sn2_c, sn2_v, linestyle="-", linewidth=1, color="gray", label="sn-2")

    ylabel = r"$-S_{CD}$"
    title = "Deuterium Order Parameter"
    if data.get("method") == "chord":
        ylabel = r"$-S_{mol}$ (chord approx.)"
        title = "Chain Order Parameter (molecular axis)"

    plt.xlabel("Carbon index", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.ylabel(ylabel, fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.title(title, fontsize=12, fontweight="bold", color="black", fontfamily="DejaVu Sans")

    xmax = int(max(sn1_c.max() if len(sn1_c) else 2,
                   sn2_c.max() if len(sn2_c) else 2))

    plt.xlim(1, max(18, xmax) + 1)

    plt.xticks(range(2, xmax + 1))

    plt.legend(loc='upper right', prop={"family": "DejaVu Sans", "size": 10}, frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()

    plt.savefig(outfile, dpi=300)
    plt.close()


# ---------------------------------------------------------
# Per-species outputs
# ---------------------------------------------------------

def plot_scd_per_species(data, outfile):
    per = data.get("per_species")
    if not per:
        return
    plt.figure(figsize=(6, 4))
    species = sorted(per)
    try:
        from analysis.curvature import species_colors
        colors = species_colors(species)
    except Exception:
        shades = ["black", "gray", "#909090", "#505050", "#c0c0c0"]
        colors = {rn: shades[i % len(shades)]
                  for i, rn in enumerate(species)}
    xmax = 2
    for rn, d in sorted(per.items()):
        c = colors[rn]
        if len(d["sn1_carbon"]):
            plt.plot(d["sn1_carbon"], -np.array(d["sn1_scd"]), color=c,
                     ls="-", lw=1, label=f"{rn} sn-1")
            xmax = max(xmax, int(d["sn1_carbon"].max()))
        if len(d["sn2_carbon"]):
            plt.plot(d["sn2_carbon"], -np.array(d["sn2_scd"]), color=c,
                     ls="--", lw=1, label=f"{rn} sn-2")
            xmax = max(xmax, int(d["sn2_carbon"].max()))
    ylabel = (r"$-S_{CD}$" if data.get("method") != "chord"
              else r"$-S_{mol}$ (chord approx.)")
    plt.xlabel("Carbon index", fontsize=12, fontfamily="DejaVu Sans")
    plt.ylabel(ylabel, fontsize=12, fontfamily="DejaVu Sans")
    plt.title("Order Parameter by Lipid Species", fontsize=12,
              fontweight="bold", fontfamily="DejaVu Sans")
    plt.xticks(range(2, xmax + 1))
    plt.legend(loc="upper right", prop={"family": "DejaVu Sans", "size": 8},
               frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def write_scd_per_species_csv(data, outfile):
    per = data.get("per_species")
    if not per:
        return
    with open(outfile, "w") as f:
        f.write("species,chain,carbon,neg_scd\n")
        for rn, d in sorted(per.items()):
            for tag in ("sn1", "sn2"):
                for c, v in zip(d[f"{tag}_carbon"], d[f"{tag}_scd"]):
                    f.write(f"{rn},{tag},{c},{-v:.6f}\n")


# ---------------------------------------------------------
# Write XVG
# ---------------------------------------------------------

def write_scd_xvg(data, outfile):

    sn1_c = np.asarray(data["sn1_carbon"], dtype=int)
    sn2_c = np.asarray(data["sn2_carbon"], dtype=int)

    sn1_v = -np.asarray(data["sn1_scd"], dtype=float)
    sn2_v = -np.asarray(data["sn2_scd"], dtype=float)

    method = data.get("method", "C-H")

    with open(outfile, "w") as f:

        f.write('@ title "Deuterium Order Parameter"\n')
        if method == "C-H":
            f.write('# method: C-H bond vectors (NMR convention, comparable to gmx order)\n')
        else:
            f.write('# method: chord-vector approximation (molecular axis, ~2x |S_CD|)\n')
        f.write('@ xaxis label "Carbon index"\n')
        f.write('@ yaxis label "-SCD"\n')
        f.write('@ s0 legend "sn-1"\n')
        f.write('@ s1 legend "sn-2"\n')
        f.write("# sn1_carbon sn1_scd sn2_carbon sn2_scd\n")

        maxlen = max(len(sn1_c), len(sn2_c))

        for i in range(maxlen):

            c1 = sn1_c[i] if i < len(sn1_c) else ""
            v1 = sn1_v[i] if i < len(sn1_v) else ""

            c2 = sn2_c[i] if i < len(sn2_c) else ""
            v2 = sn2_v[i] if i < len(sn2_v) else ""

            f.write(f"{c1:4} {v1:12} {c2:6} {v2}\n")
