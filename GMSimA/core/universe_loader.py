"""
Central, format-agnostic Universe loader for smGROANA.

Every place that used to call ``mda.Universe(tpr, xtc)`` calls
``load_universe(topology, xtc)`` instead. The topology may be:

  .tpr  -> exact topology (bonds incl. constraints, masses, charges).
           If the TPX version is newer than the installed MDAnalysis
           (e.g. GROMACS 2026 / TPX 138 vs MDAnalysis <= 2.10), the
           optional pure-Python ``TprParser`` package is used instead;
           without it the user is told to load a .gro/.pdb.
  .gro  -> coordinates + names only; bonds GUESSED from distances,
  .pdb     masses guessed from atom names, no charges.

No GROMACS installation is needed at run time.

Why bonds matter: fragments, unwrap/make_whole, "same fragment as ..."
selections and S_CD bonded hydrogens all use them, so for GRO/PDB we
always guess them (MDAnalysis does not by default).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import MDAnalysis as mda

__all__ = ["load_universe", "UnsupportedTPRError", "tpr_backend",
           "TOPOLOGY_EXTENSIONS", "TOPOLOGY_FILE_FILTER"]

TOPOLOGY_EXTENSIONS = (".tpr", ".gro", ".pdb", ".ent")
TOPOLOGY_FILE_FILTER = ("Topology (*.tpr *.gro *.pdb);;GROMACS run input (*.tpr);;"
                        "GROMACS structure (*.gro);;PDB (*.pdb *.ent);;All files (*)")

# human-readable description of each backend (shown in GUI / logs)
BACKEND_LABELS = {
    "tpr": "TPR (exact topology)",
    "tprparser": "TPR via TprParser (TPR newer than installed MDAnalysis)",
    "coords": "GRO/PDB (bonds guessed, masses guessed, no charges)",
}


class UnsupportedTPRError(RuntimeError):
    """TPR format too new for MDAnalysis and no fallback available."""


def _is_tpx_version_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    # MDAnalysis wraps the NotImplementedError from TPRParser in a
    # ValueError("Failed to construct topology ... Error: Your tpx version...")
    return "tpx version" in msg or "tpx generation" in msg


def tpr_backend(u) -> str:
    """'tpr', 'tprparser' or 'coords' — how this Universe's topology was built."""
    return getattr(u, "_smg_tpr_backend", "tpr")


def load_universe(topology, *trajectory, **kwargs):
    """
    Drop-in replacement for ``mda.Universe(topology, *trajectory, **kw)``
    accepting .tpr, .gro or .pdb topologies.

    Errors other than an unsupported TPX version (e.g. atom-count
    mismatch with the XTC) are raised unchanged so callers can report them.
    """
    topology = os.fspath(topology)
    ext = os.path.splitext(topology)[1].lower()

    if ext in (".gro", ".pdb", ".ent"):
        return _universe_from_coordinates(topology, *trajectory, **kwargs)

    try:
        u = mda.Universe(topology, *trajectory, **kwargs)
        u._smg_tpr_backend = "tpr"
        return u
    except Exception as exc:
        if not (ext == ".tpr" and _is_tpx_version_error(exc)):
            raise
        native_err = exc

    # TPR too new for MDAnalysis -> optional TprParser (exact topology)
    try:
        u = _universe_from_tprparser(topology, *trajectory, **kwargs)
    except ImportError:
        stem = os.path.splitext(os.path.basename(topology))[0]
        raise UnsupportedTPRError(
            f"{native_err}\n\n"
            f"This .tpr was written by a newer GROMACS than the installed "
            f"MDAnalysis ({mda.__version__}) can read.\n\n"
            "Load a .gro or .pdb of the SAME (e.g. dry) system as topology "
            "instead, e.g.\n"
            f"    gmx editconf -f {stem}.tpr -o {stem}.gro\n"
            "or install the exact TPR fallback reader:\n"
            "    pip install TprParser"
        ) from native_err

    warnings.warn(
        f"MDAnalysis {mda.__version__} cannot read "
        f"{os.path.basename(topology)} (tpx too new); topology was built "
        f"with the TprParser fallback.", RuntimeWarning, stacklevel=2)
    return u


# ----------------------------------------------------------------------
# GRO / PDB topologies
# ----------------------------------------------------------------------

_CYS_SG = ("CYS", "CYX", "CYS2", "CYM", "CSS")


def _clean_guessed_bonds(u):
    """
    Distance-based bond guessing (vdW radii × 0.55) has two systematic
    errors in solvated protein–membrane frames:
      * misses disulfides (S–S ≈ 2.05 Å > 1.98 Å cutoff);
      * adds bonds across close contacts — salt bridges, H-bonds,
        clashes — which can merge a peptide with a lipid into one
        "fragment" and corrupt fragment-based selections/unwrapping.
    Rules applied (valid for CHARMM36, AMBER Lipid14/17/21, OPLS, …):
      1. a hydrogen never bonds across residues, and keeps only its
         single shortest bond;
      2. inter-residue heavy-atom bonds are kept only between
         CONSECUTIVE residues — for protein–protein only backbone C–N
         (AMBER multi-residue lipids PA/PC/OL are consecutive, so kept);
      3. Cys SG–SG pairs within 2.5 Å are added as disulfides.
    Returns (n_removed, n_disulfides_added).
    """
    from MDAnalysis.core.selection import ProteinSelection
    from MDAnalysis.lib.distances import calc_bonds, capped_distance

    ag = u.atoms
    if not hasattr(u, "bonds") or len(u.bonds) == 0:
        return 0, 0
    idx = np.sort(u.bonds.indices, axis=1)
    i, j = idx[:, 0], idx[:, 1]

    types = np.char.upper(np.asarray(ag.types, dtype=str))
    is_h = (types == "H") | ((ag.masses > 0.5) & (ag.masses < 1.5))
    resix = ag.resindices
    names = np.asarray(ag.names, dtype=str)
    prot_res = np.isin(np.asarray(ag.resnames, dtype=str),
                       list(ProteinSelection.prot_res))

    inter = resix[i] != resix[j]
    anyh = is_h[i] | is_h[j]
    consecutive = np.abs(resix[i] - resix[j]) == 1
    both_prot = prot_res[i] & prot_res[j]
    backbone_cn = (((names[i] == "C") & (names[j] == "N")) |
                   ((names[i] == "N") & (names[j] == "C")))

    keep = ~inter | (~anyh & consecutive & (~both_prot | backbone_cn))

    # rule 1b: each hydrogen keeps only its shortest remaining bond
    d = calc_bonds(ag.positions[i], ag.positions[j], box=u.dimensions)
    bi = np.r_[np.where(keep & is_h[i])[0], np.where(keep & is_h[j])[0]]
    hatom = np.r_[i[keep & is_h[i]], j[keep & is_h[j]]]
    if len(bi):
        order = np.lexsort((d[bi], hatom))          # by H atom, then length
        hs, bs = hatom[order], bi[order]
        dup = np.r_[False, hs[1:] == hs[:-1]]       # all but the shortest
        keep[bs[dup]] = False

    # rule 3: disulfides
    sg = u.select_atoms("name SG and resname " + " ".join(_CYS_SG))
    ss = []
    if len(sg) > 1:
        pairs = capped_distance(sg.positions, sg.positions, max_cutoff=2.5,
                                box=u.dimensions, return_distances=False)
        ss = sorted({(min(sg[a].index, sg[b].index), max(sg[a].index, sg[b].index))
                     for a, b in pairs if a != b})

    removed = int((~keep).sum())
    new = [tuple(x) for x in idx[keep]]
    existing = set(new)
    ss = [p for p in ss if p not in existing]
    u.delete_bonds([tuple(x) for x in idx])
    u.add_bonds(new + ss)
    return removed, len(ss)


def _universe_from_coordinates(path, *trajectory, **kwargs):
    from packaging.version import Version
    if Version(mda.__version__.split("-")[0]) >= Version("2.8.0"):
        kwargs.setdefault("to_guess", ("types", "masses", "bonds"))
    else:
        kwargs.setdefault("guess_bonds", True)
    with warnings.catch_warnings():
        # element/mass guessing chatter for CHARMM/AMBER names is expected
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        u = mda.Universe(path, *trajectory, **kwargs)
    u._smg_tpr_backend = "coords"
    u._smg_bond_cleanup = _clean_guessed_bonds(u)

    # atoms whose mass could not be guessed get 0 -> breaks COM/curvature
    zero = u.atoms[u.atoms.masses <= 0]
    if len(zero):
        names = ", ".join(sorted(set(zero.names))[:10])
        warnings.warn(
            f"{len(zero)} atoms in {os.path.basename(path)} have no guessable "
            f"mass (names: {names}). Mass-weighted quantities may be "
            f"affected; use a .tpr if possible.", RuntimeWarning, stacklevel=3)
    return u


# ----------------------------------------------------------------------
# TprParser fallback
# ----------------------------------------------------------------------

_ELEMENTS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
]


def _residue_index(resids, resnames, names):
    """
    Per-atom residue index. A new residue starts when resid or resname
    changes, or when an atom name repeats inside the current residue
    (consecutive identical molecules that share a resid).
    """
    n = len(names)
    resindex = np.empty(n, dtype=np.int64)
    r = -1
    seen = set()
    prev = (None, None)
    for i in range(n):
        key = (resids[i], resnames[i])
        if key != prev or names[i] in seen:
            r += 1
            seen = set()
            prev = key
        seen.add(names[i])
        resindex[i] = r
    return resindex


def _fix_unbonded_hydrogens(bonds, is_h, heavy_ok, resindex, xyz_A,
                            cutoff=1.3):
    """
    Hydrogens that carry no bond in the TPR (e.g. constraints or SETTLE
    not exported as bonds) are bonded to the nearest heavy atom of the
    same residue within `cutoff` Å, using the TPR reference coordinates.
    """
    n = len(is_h)
    has_bond = np.zeros(n, dtype=bool)
    if len(bonds):
        has_bond[bonds[:, 0]] = True
        has_bond[bonds[:, 1]] = True
    lonely = np.where(is_h & ~has_bond)[0]
    if len(lonely) == 0 or xyz_A is None:
        return bonds, 0

    extra = []
    # group by residue: tiny searches, no PBC issues within a molecule
    heavy_by_res = {}
    for i in np.where(heavy_ok)[0]:
        heavy_by_res.setdefault(resindex[i], []).append(i)
    for h in lonely:
        cand = heavy_by_res.get(resindex[h])
        if not cand:
            continue
        cand = np.asarray(cand)
        d = np.linalg.norm(xyz_A[cand] - xyz_A[h], axis=1)
        j = int(np.argmin(d))
        if d[j] <= cutoff:
            extra.append((min(h, cand[j]), max(h, cand[j])))
    if extra:
        bonds = np.vstack([bonds, np.asarray(extra, dtype=np.int64)]) \
            if len(bonds) else np.asarray(extra, dtype=np.int64)
    return bonds, len(extra)


def _connected_components(n, bonds):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in bonds:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def _universe_from_tprparser(tpr, *trajectory, **kwargs):
    from TprParser import TprReader  # optional dependency (GPL)
    from MDAnalysis.core.selection import ProteinSelection
    from MDAnalysis.lib.mdamath import triclinic_box

    r = TprReader(tpr)
    names = np.asarray(r.get_name("atom"), dtype=object)
    types = np.asarray(r.get_name("type"), dtype=object)
    resnames_atom = np.asarray(r.get_name("res"), dtype=object)
    resids_atom = np.asarray(r.get_ivector("resid"), dtype=np.int64)
    masses = np.asarray(r.get_mq("m"), dtype=np.float64)
    charges = np.asarray(r.get_mq("q"), dtype=np.float64)
    n = len(names)

    try:
        z = np.asarray(r.get_ivector("atomicnum"), dtype=np.int64)
        elements = np.array([_ELEMENTS[k] if 0 < k < len(_ELEMENTS) else ""
                             for k in z], dtype=object)
    except Exception:
        z = np.zeros(n, dtype=np.int64)
        elements = np.array([""] * n, dtype=object)

    try:
        xyz_A = np.asarray(r.get_xvf("x"), dtype=np.float32) * 10.0
    except Exception:
        xyz_A = None
    try:
        box_A = triclinic_box(*(np.asarray(r.get_xvf("box"),
                                           dtype=np.float64) * 10.0))
    except Exception:
        box_A = None

    # ---- residues --------------------------------------------------
    resindex = _residue_index(resids_atom, resnames_atom, names)
    n_res = int(resindex[-1]) + 1 if n else 0
    first = np.zeros(n_res, dtype=np.int64)
    first[resindex[::-1]] = np.arange(n)[::-1]   # first atom of each residue
    res_resids = resids_atom[first]
    res_resnames = resnames_atom[first]

    # ---- bonds (TPR indices are 1-based) ---------------------------
    raw = r.get_bonded("bonds")
    if len(raw):
        bonds = np.array([[int(b[0]) - 1, int(b[1]) - 1] for b in raw],
                         dtype=np.int64)
    else:
        bonds = np.empty((0, 2), dtype=np.int64)
    is_h = (z == 1) if z.any() else (masses < 1.5) & (masses > 0.5)
    heavy = (~is_h) & (masses > 1.5)
    bonds, n_fixed = _fix_unbonded_hydrogens(bonds, is_h, heavy,
                                             resindex, xyz_A)
    if len(bonds):
        bonds = np.unique(np.sort(bonds, axis=1), axis=0)

    # ---- segments: one per protein fragment, else per resname run --
    prot_res = set(ProteinSelection.prot_res)
    comp = _connected_components(n, bonds)
    seg_keys, prot_label = [], {}
    for ri in range(n_res):
        if res_resnames[ri] in prot_res:
            c = comp[first[ri]]
            if c not in prot_label:
                k = len(prot_label)
                prot_label[c] = "PRO" + (chr(65 + k) if k < 26 else str(k))
            seg_keys.append(prot_label[c])
        else:
            seg_keys.append(str(res_resnames[ri]))
    res_segindex = np.zeros(n_res, dtype=np.int64)
    segids = []
    for ri, key in enumerate(seg_keys):
        if ri == 0 or key != seg_keys[ri - 1]:
            segids.append(key)
        res_segindex[ri] = len(segids) - 1

    # ---- assemble ---------------------------------------------------
    u = mda.Universe.empty(n, n_residues=n_res, n_segments=len(segids),
                           atom_resindex=resindex,
                           residue_segindex=res_segindex,
                           trajectory=False)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("types", types)
    u.add_TopologyAttr("elements", elements)
    u.add_TopologyAttr("masses", masses)
    u.add_TopologyAttr("charges", charges)
    u.add_TopologyAttr("ids", np.arange(1, n + 1))
    u.add_TopologyAttr("resnames", res_resnames)
    u.add_TopologyAttr("resids", res_resids)
    u.add_TopologyAttr("resnums", res_resids.copy())
    u.add_TopologyAttr("segids", np.asarray(segids, dtype=object))
    u.add_TopologyAttr("bonds", [tuple(b) for b in bonds])

    # ---- coordinates -----------------------------------------------
    if trajectory:
        u.load_new(trajectory[0] if len(trajectory) == 1 else list(trajectory),
                   **{k: v for k, v in kwargs.items()
                      if k in ("in_memory", "dt", "continuous")})
    elif xyz_A is not None:
        from MDAnalysis.coordinates.memory import MemoryReader
        u.trajectory = MemoryReader(xyz_A[None, :, :],
                                    dimensions=box_A, order="fac")

    u._smg_tpr_backend = "tprparser"
    u._smg_fixed_h_bonds = n_fixed
    return u


if __name__ == "__main__":  # quick CLI check: python -m core.universe_loader x.tpr [x.xtc]
    import sys
    uu = load_universe(*sys.argv[1:])
    print(f"backend={tpr_backend(uu)} atoms={uu.atoms.n_atoms} "
          f"residues={uu.residues.n_residues} bonds={len(uu.bonds)} "
          f"segids={sorted(set(uu.segments.segids))[:8]} "
          f"frames={len(uu.trajectory)}")
