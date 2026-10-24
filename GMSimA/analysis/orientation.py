"""
smGROANA / GMSimA — biomolecule orientation in a lipid bilayer.

Tilt (theta) and rotation (rho) angles computed the way the solid-state
NMR literature defines them, so that simulated values are directly
comparable with experiment:

    Strandberg, Ozdirekcan, Rijkers, van der Wel, Koeppe, Liskamp &
    Killian (2004) "Tilt Angles of Transmembrane Model Peptides in
    Oriented and Non-Oriented Lipid Bilayers as Determined by 2H
    Solid-State NMR", Biophys. J. 86:3709-3721  (the GALA method).

Definitions used here (Fig. 1 and Appendix of that paper)
---------------------------------------------------------
theta Tilt angle: the angle between the *helix axis* and the bilayer
      normal. The helix axis is a director (a and -a describe the same
      orientation), so theta is folded into [0, 90] degrees.

      NOTE ON NOTATION: Strandberg et al. write the tilt angle as tau.
      smGROANA/GMSimA already uses tau for the binding RESIDENCE time
      (tau-off) throughout the suite, so the tilt is written theta here
      to avoid a symbol clash in the same report. theta (this module)
      == tau (the GALA paper); rho is unchanged.

rho   Rotation angle: the rotation of the helix about its own axis. It
      is the anticlockwise angle, measured about the helix axis, from
      the membrane-facing (down-tilt) direction to the projection of a
      reference C-alpha. rho = 0 therefore means the reference residue
      points at the bilayer, i.e. it lies in the plane containing the
      helix axis and the bilayer normal, on the downhill side. This
      reproduces the GALA convention in which rho is the rotation of
      C-alpha of residue 1 relative to a standard orientation with that
      C-alpha in the tilt plane.

      rho is undefined when theta -> 0 (an untilted helix has no tilt
      direction), so it is reported as NaN below `rho_min_tilt`.

Why this module exists
----------------------
The v2/v3 tilt used the end-to-end vector COM(residue 1) -> COM(residue
N). That is adequate for a single straight helix but fails in three
common cases:

  * cyclic peptides -- the first and last residues are covalently
    bonded, so the "end-to-end" vector is a single peptide bond and
    carries no orientational information at all;
  * multipass proteins -- the vector is dominated by the disordered
    terminal tails and says nothing about the transmembrane bundle;
  * frayed or kinked helices -- two terminal residues carry all the
    noise of the measurement.

This module therefore fits a real axis and picks the appropriate
definition from the molecule's topology:

  linear helical peptide  -> smoothed helix axis, theta + rho
  cyclic peptide          -> normal of the macrocycle plane, theta only
  single-pass protein     -> helix axis of the TM segment, theta + rho
  multipass protein       -> per-segment theta + rho, plus a bundle axis
  no helix found          -> principal axis of C-alpha, theta only (warned)

Only numpy / MDAnalysis are required; pandas and matplotlib are used by
the writer and plotting helpers.
"""

from __future__ import annotations

import numpy as np

# Ideal alpha-helix geometry (used for helix detection and diagnostics)
_CA_I_IP4_MIN = 5.0      # Angstrom; ideal alpha-helix C-alpha(i)-C-alpha(i+4)
_CA_I_IP4_MAX = 7.2      # is 6.2 A. 3-10 and pi helices sit just outside.
_CA_BOND_MAX = 4.5       # consecutive C-alpha separation; larger = chain break

_DEG = 180.0 / np.pi


# ======================================================================
# Geometry primitives
# ======================================================================
def _unit(v):
    """Normalise a vector; return None if it is degenerate."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return None if n < 1e-8 else v / n


def _principal_axes(pos):
    """
    Eigen-decomposition of the gyration tensor of `pos`.

    Returns (w, V) with eigenvalues w descending and V[:, k] the k-th
    eigenvector. V[:, 0] is the long axis, V[:, 2] the normal of the
    best-fit plane (the direction of least extent).
    """
    p = np.asarray(pos, dtype=float)
    p = p - p.mean(axis=0)
    w, V = np.linalg.eigh(np.cov(p.T))
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def _running_mean(p, w):
    """Running mean of a point series over a window of `w` points."""
    p = np.asarray(p, dtype=float)
    if len(p) < w:
        return p
    c = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), axis=0)
    return (c[w:] - c[:-w]) / float(w)


def helix_axis(ca_pos):
    """
    Axis of a helical C-alpha trace.

    A running mean over four consecutive C-alpha atoms (approximately
    one alpha-helical turn, 3.6 residues) removes the helical
    oscillation and leaves points lying on the axis itself; a straight
    line is then fitted through those points by SVD. This is far more
    robust than an end-to-end vector because every residue contributes,
    and it is insensitive to terminal fraying.

    Falls back to the principal axis for very short segments.

    Returns (axis_unit_vector, axis_centroid) or (None, None).
    """
    p = np.asarray(ca_pos, dtype=float)
    if len(p) < 4:
        return None, None
    if len(p) >= 5:
        # One-turn running mean -> points lying on the axis. A window of
        # four does not span exactly one turn (3.6 residues), so a small
        # residual oscillation of amplitude
        #     |sin(4*100/2) / (4 sin(100/2))| = 0.11 * r
        # survives and biases the fitted direction by ~0.15 deg. A second
        # pass squares that factor down to ~0.01, which removes the bias;
        # it is skipped for short segments, where points are too scarce.
        q = _running_mean(p, 4)
        if len(q) >= 8:
            q = _running_mean(q, 4)
        c = q.mean(axis=0)
        # principal direction of the smoothed trace
        _, _, Vt = np.linalg.svd(q - c, full_matrices=False)
        v = Vt[0]
    else:
        w, V = _principal_axes(p)
        v = V[:, 0]
        c = p.mean(axis=0)
    # orient N -> C
    if np.dot(v, p[-1] - p[0]) < 0.0:
        v = -v
    return _unit(v), c


def ring_normal(pos):
    """
    Normal of the best-fit plane of a macrocycle (direction of least
    extent). Also returns the ring's planarity, lambda2 / lambda3: large
    values mean a genuinely disc-like ring for which the normal is well
    defined, values near 1 mean a puckered ring for which it is not.

    Returns (normal_unit_vector, centroid, planarity).
    """
    p = np.asarray(pos, dtype=float)
    if len(p) < 3:
        return None, None, 0.0
    w, V = _principal_axes(p)
    planarity = float(w[1] / max(w[2], 1e-12))
    return _unit(V[:, 2]), p.mean(axis=0), planarity


def tilt_angle(axis, normal):
    """
    Tilt angle in degrees between a director and the membrane normal,
    folded into [0, 90] because `axis` and `-axis` are equivalent.
    """
    a = _unit(axis)
    n = _unit(normal)
    if a is None or n is None:
        return float("nan")
    t = _DEG * np.arccos(np.clip(abs(float(np.dot(a, n))), 0.0, 1.0))
    return float(t)


def rotation_angle(axis, axis_point, ref_pos, normal):
    """
    GALA rotation angle rho, in degrees, in [0, 360).

    Measured anticlockwise about `axis` (right-hand rule) from the
    membrane-facing (down-tilt) direction -- the reversed membrane normal
    projected perpendicular to the axis -- to the component of
    (ref_pos - axis_point) perpendicular to the axis. For theta < 90 deg
    this is the GALA tilt direction; see the comment in the body.

    Returns NaN if the helix is untilted (no tilt direction exists) or
    if the reference atom sits on the axis.
    """
    a = _unit(axis)
    n = _unit(normal)
    if a is None or n is None:
        return float("nan")

    # Reference direction (rho = 0): the membrane normal, reversed and
    # projected into the plane perpendicular to the axis -- i.e. the
    # "downhill" side of the tilted helix, the face turned towards the
    # bilayer.
    #
    # For theta < 90 deg this is identical to the GALA construction (the
    # axis projected into the membrane plane, then made perpendicular to
    # the axis): both reduce to cos(theta)*e - sin(theta)*n. The difference
    # is the normalisation. Building it from the axis divides by
    # |cos(theta)|, which CHANGES SIGN as the helix crosses theta = 90 and
    # makes rho jump by exactly 180 deg; building it from the normal
    # divides by sin(theta), which stays positive throughout. A surface-
    # bound helix sits at theta ~ 90 and crosses it constantly, so the
    # first form splits one physical orientation into two phantom
    # populations 180 deg apart. This form is degenerate only at
    # theta = 0, which `rho_min_tilt` already guards.
    t = _unit(-n + float(np.dot(n, a)) * a)
    if t is None:                      # axis parallel to n -> no tilt
        return float("nan")

    w = np.asarray(ref_pos, dtype=float) - np.asarray(axis_point, dtype=float)
    w = _unit(w - float(np.dot(w, a)) * a)
    if w is None:                      # reference atom lies on the axis
        return float("nan")

    x = float(np.dot(w, t))
    y = float(np.dot(np.cross(t, w), a))
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


# ======================================================================
# Topology detection
# ======================================================================
def is_cyclic(sel):
    """
    True when a bond joins the first and last residue of `sel`, i.e. the
    molecule is a head-to-tail macrocycle. Requires bond information
    (always present when the topology comes from a .tpr).
    """
    try:
        first = set(sel.residues[0].atoms.indices)
        last = set(sel.residues[-1].atoms.indices)
        if not first or not last:
            return False
        for b in sel.bonds:
            i, j = int(b.atoms[0].index), int(b.atoms[1].index)
            if (i in first and j in last) or (i in last and j in first):
                return True
    except Exception:
        pass
    return False


def _ca_atoms(sel):
    """C-alpha atoms of `sel`, or None if the selection is not a protein."""
    try:
        ca = sel.select_atoms("name CA")
    except Exception:
        return None
    return ca if len(ca) >= 4 else None


def find_helical_segments(ca, min_len=7):
    """
    Locate alpha-helical stretches in a C-alpha trace from the
    C-alpha(i)-C-alpha(i+4) distance, which is 6.2 A in an ideal
    alpha-helix and much larger in extended or coil regions. No external
    secondary-structure program is needed.

    Chain breaks (consecutive C-alpha further apart than _CA_BOND_MAX)
    terminate a segment.

    Returns a list of (start_index, end_index) inclusive index pairs into
    `ca`.
    """
    p = ca.positions
    n = len(p)
    if n < min_len:
        return []

    broken = np.zeros(n, dtype=bool)
    d1 = np.linalg.norm(p[1:] - p[:-1], axis=1)
    broken[1:] = d1 > _CA_BOND_MAX

    ok = np.zeros(max(n - 4, 0), dtype=bool)
    for i in range(n - 4):
        if broken[i + 1:i + 5].any():
            continue
        d = float(np.linalg.norm(p[i + 4] - p[i]))
        ok[i] = _CA_I_IP4_MIN <= d <= _CA_I_IP4_MAX

    segments = []
    i = 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        start, end = i, j + 4          # window i covers residues i..i+4
        if end - start + 1 >= min_len:
            segments.append((start, end))
        i = j + 1
    return segments


def _leaflet_planes(memb_sel):
    """
    Mean z of the upper and lower phosphate planes, from the current
    frame. Returns (z_lower, z_upper) or None.
    """
    p = _phosphates(memb_sel)
    if p is None:
        return None
    try:
        z = p.positions[:, 2]
        mid = float(z.mean())
        up, lo = z[z > mid], z[z <= mid]
        if len(up) == 0 or len(lo) == 0:
            return None
        return float(lo.mean()), float(up.mean())
    except Exception:
        return None


def _spans_membrane(ca_pos, planes, frac=0.6):
    """True if a segment crosses the midplane and covers most of the
    hydrophobic core -- i.e. it is transmembrane rather than an
    interfacial or soluble helix."""
    if planes is None:
        return False
    lo, up = planes
    mid = 0.5 * (lo + up)
    z = ca_pos[:, 2]
    if not (z.min() < mid < z.max()):
        return False
    return (z.max() - z.min()) >= frac * abs(up - lo)


def parse_segments(spec, sel):
    """
    Parse a user-supplied segment specification such as
    "6-33, 41-68, 76-105" (resids, inclusive) into index pairs into the
    C-alpha selection. Unknown resids are skipped.
    """
    ca = _ca_atoms(sel)
    if ca is None or not spec:
        return []
    resids = np.array([a.resid for a in ca])
    out = []
    for chunk in str(spec).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            a, _, b = chunk.partition("-")
            lo, hi = int(a), int(b if b else a)
        except ValueError:
            continue
        idx = np.where((resids >= lo) & (resids <= hi))[0]
        if len(idx) >= 5:
            out.append((int(idx[0]), int(idx[-1])))
    return out


# ======================================================================
# Membrane normal
# ======================================================================
def _leaflet_normal(pos):
    """
    Normal of the plane through the phosphates of a *single* leaflet,
    by total least squares (smallest principal axis), oriented to +z.

    Two points of care, both of which cost accuracy if ignored:

    * The fit must be done one leaflet at a time. The phosphates of a
      whole bilayer form two sheets ~40 A apart, so across both leaflets
      the z-variance rivals the lateral variance of a typical patch and
      the smallest principal axis is then not the membrane normal at all.
      Within one leaflet the problem is well conditioned: ~80 A of
      lateral extent against ~3 A of roughness.
    * Total least squares rather than a z = a*x + b*y + c regression.
      The latter treats x and y as error-free, which biases the normal
      towards z once the membrane is appreciably tilted (regression
      dilution); the symmetric fit has no such bias.
    """
    p = np.asarray(pos, dtype=float)
    if len(p) < 3:
        return None
    w, V = _principal_axes(p)
    if w[1] <= 0 or w[2] / max(w[1], 1e-12) > 0.5:
        return None                    # not a well-defined sheet
    n = _unit(V[:, 2])
    if n is None:
        return None
    return n if n[2] >= 0 else -n


def _phosphates(memb_sel):
    """Phosphate atoms of the membrane, across naming conventions."""
    if memb_sel is None:
        return None
    try:
        p = memb_sel.select_atoms("name P P31 P8 PO4")
        if len(p) < 4:
            p = memb_sel.select_atoms("name P*")
        return p if len(p) >= 4 else None
    except Exception:
        return None


def membrane_normal(memb_sel, mode="z", near_point=None, local_radius=25.0):
    """
    Bilayer normal for the current frame.

      "z"       lab z axis. Correct whenever the trajectory has been
                preprocessed with `gmx trjconv -pbc mol -center`
                (and -fit rot+trans), which smGROANA already requires.
      "global"  normal of the plane fitted through every phosphate atom;
                follows a bilayer that drifts or tilts as a whole.
      "local"   normal fitted to phosphates within `local_radius` (in
                the xy plane) of `near_point`; follows undulations under
                a large protein. Falls back to "global" if too few
                phosphates are nearby.

    Always oriented towards +z so that the folded tilt is unambiguous.
    """
    z_hat = np.array([0.0, 0.0, 1.0])
    if mode == "z" or memb_sel is None:
        return z_hat
    try:
        p = _phosphates(memb_sel)
        if p is None:
            return z_hat
        pos = p.positions
        if mode == "local" and near_point is not None:
            d = np.linalg.norm(pos[:, :2] - np.asarray(near_point)[:2], axis=1)
            near = pos[d < local_radius]
            if len(near) >= 12:          # need enough per leaflet to fit
                pos = near
        # Fit each leaflet separately and average: a single fit through
        # both sheets is dominated by the leaflet separation.
        #
        # Leaflets are separated along the current normal estimate, not
        # along z. Splitting on raw z misassigns edge phosphates as soon
        # as the bilayer is tilted or strongly undulating, and those
        # outliers sit far laterally, so they drag the fitted plane back
        # towards z. Two or three refinement passes remove that bias.
        centre = pos.mean(axis=0)
        n = z_hat
        for _ in range(4):
            s = (pos - centre) @ n
            normals = []
            for leaflet in (pos[s > 0], pos[s <= 0]):
                if len(leaflet) >= 3:
                    nl = _leaflet_normal(leaflet)
                    if nl is not None:
                        normals.append(nl)
            if not normals:
                return z_hat
            n_new = _unit(np.mean(normals, axis=0))
            if n_new is None:
                return z_hat
            if n_new[2] < 0:
                n_new = -n_new
            if float(np.dot(n_new, n)) > 0.9999999:   # converged
                n = n_new
                break
            n = n_new
        return n
    except Exception:
        return z_hat


# ======================================================================
# Main driver
# ======================================================================
def compute_orientation(u, biomol_sel, stride=1, memb_sel=None,
                        mode="auto", segments=None, ref_resid=None,
                        normal="z", rho_min_tilt=5.0, min_helix=7,
                        verbose=True):
    """
    Tilt (and, where defined, rotation) of a biomolecule in a bilayer.

    Parameters
    ----------
    u            MDAnalysis Universe.
    biomol_sel   AtomGroup for one peptide / protein.
    stride       Frame stride.
    memb_sel     AtomGroup of the membrane (needed for transmembrane
                 classification and for normal="global"/"local").
    mode         "auto"      choose from the topology (recommended)
                 "helix"     fit one helix axis over the whole molecule
                 "segments"  use `segments` / auto-detected TM helices
                 "ring"      macrocycle plane normal
                 "principal" principal axis of all C-alpha
                 "endtoend"  legacy v2/v3 definition, for reproducing
                             previously published numbers
    segments     Residue ranges, e.g. "6-33, 41-68". Overrides detection.
    ref_resid    Resid whose C-alpha defines rho = 0. Defaults to the
                 first residue of each segment.
    normal       "z" | "global" | "local"  (see membrane_normal).
    rho_min_tilt Below this tilt (degrees) rho is reported as NaN.
    min_helix    Minimum residues for a detected helical segment.

    Returns
    -------
    dict with, at minimum, "time" and "tilt" so that it is a drop-in
    replacement for compute_tilt_timeseries(); plus "rho", "segments"
    (per-segment time series), "mode", "topology" and "warnings".
    """
    warn = []

    def _log(msg):
        warn.append(msg)
        if verbose:
            print(f"[orientation] {msg}")

    if len(biomol_sel.residues) < 2:
        raise ValueError("Biomolecule must have at least 2 residues.")

    ca = _ca_atoms(biomol_sel)
    cyclic = is_cyclic(biomol_sel)
    first_res = biomol_sel.residues[0].atoms
    last_res = biomol_sel.residues[-1].atoms

    # Topology detection reads coordinates, so pin it to the first frame:
    # otherwise the segments found depend on wherever a previous analysis
    # happened to leave the trajectory reader.
    try:
        u.trajectory[0]
    except Exception:
        pass

    # ---------------- decide what to measure -------------------------
    seg_idx = []
    topology = "unknown"

    if mode == "auto":
        if cyclic:
            mode, topology = "ring", "cyclic peptide"
            _log("cyclic (head-to-tail) molecule detected: the end-to-end "
                 "vector would be a single peptide bond, so the tilt is "
                 "measured as the angle between the macrocycle plane "
                 "normal and the bilayer normal (theta = 0 means the ring "
                 "lies flat in the membrane plane).")
        elif ca is None:
            mode, topology = "principal", "non-protein"
            _log("no C-alpha atoms found; using the principal axis of the "
                 "selection.")
        else:
            if segments:
                seg_idx = parse_segments(segments, biomol_sel)
            if not seg_idx:
                seg_idx = find_helical_segments(ca, min_len=min_helix)
            planes = _leaflet_planes(memb_sel)
            tm = [s for s in seg_idx
                  if _spans_membrane(ca.positions[s[0]:s[1] + 1], planes)]
            if planes is not None and tm:
                seg_idx = tm
            if len(seg_idx) >= 2:
                mode = "segments"
                topology = (f"multipass protein ({len(seg_idx)} "
                            f"{'transmembrane ' if tm else ''}segments)")
                _log(f"{topology}: reporting theta and rho per segment; the "
                     "whole-molecule tilt is the bundle axis (mean of the "
                     "segment directors).")
            elif len(seg_idx) == 1:
                mode = "segments"
                topology = ("single-pass protein" if tm
                            else "single helical peptide")
                _log(f"{topology}: helix axis fitted over residues "
                     f"{ca[seg_idx[0][0]].resid}-{ca[seg_idx[0][1]].resid}.")
            else:
                mode, topology = "principal", "no helix detected"
                _log("no alpha-helical segment found (disordered or "
                     "non-helical peptide). Falling back to the principal "
                     "axis of C-alpha; interpret the tilt with care -- for "
                     "a disordered peptide, orientation is weakly defined.")
    elif mode == "segments":
        user_given = False
        if segments:
            seg_idx = parse_segments(segments, biomol_sel)
            user_given = bool(seg_idx)
        if not seg_idx and ca is not None:
            seg_idx = find_helical_segments(ca, min_len=min_helix)
        if not seg_idx:
            mode, topology = "principal", "no helix detected"
            _log("no segments given or detected; using the principal axis.")
        else:
            topology = (f"{len(seg_idx)} user-specified segment(s)"
                        if user_given else
                        f"{len(seg_idx)} detected helical segment(s)")

    # single whole-molecule helix requested explicitly
    if mode == "helix" and ca is not None:
        seg_idx = [(0, len(ca) - 1)]
        mode = "segments"
        topology = topology if topology != "unknown" else "single helix"

    # An explicitly requested mode never went through auto-detection, so
    # give it a descriptive label rather than leaving it "unknown".
    if topology == "unknown":
        topology = {"ring": "macrocycle (user-selected)",
                    "principal": "principal axis (user-selected)",
                    "endtoend": "end-to-end vector (legacy)",
                    "helix": "single helix (user-selected)",
                    }.get(mode, topology)

    # ---------------- shape diagnostics on the first frame ------------
    if mode == "endtoend":
        r1 = first_res.center_of_mass()
        r2 = last_res.center_of_mass()
        pos = (ca if ca is not None else biomol_sel).positions
        w, _ = _principal_axes(pos)
        ete = float(np.linalg.norm(r2 - r1))
        rg = float(np.sqrt(w.sum()))
        if cyclic:
            _log("WARNING: end-to-end mode on a cyclic molecule -- the "
                 "result is meaningless. Use mode='ring'.")
        elif ete < 1.5 * rg:
            _log(f"WARNING: end-to-end distance ({ete:.1f} A) is small "
                 f"relative to Rg ({rg:.1f} A); the axis is dominated by "
                 "terminal fluctuations.")

    # ---------------- per-segment bookkeeping -------------------------
    seg_meta = []
    if mode == "segments":
        for k, (a, b) in enumerate(seg_idx):
            sub = ca[a:b + 1]
            if ref_resid is not None and ref_resid in [x.resid for x in sub]:
                ref_local = [x.resid for x in sub].index(ref_resid)
            else:
                ref_local = 0
            seg_meta.append({
                "name": f"S{k + 1}",
                "i0": a, "i1": b,
                "resid_start": int(sub[0].resid),
                "resid_end": int(sub[-1].resid),
                "n_res": len(sub),
                "ref_resid": int(sub[ref_local].resid),
                "ref_local": ref_local,
                "tilt": [], "rho": [],
            })

    # ---------------- trajectory loop ---------------------------------
    times, tilts, rhos = [], [], []
    planarity_seen = []

    for ts in u.trajectory[::stride]:
        times.append(ts.time / 1000.0)          # ns

        if mode == "segments":
            axes, points = [], []
            for m in seg_meta:
                sub_pos = ca.positions[m["i0"]:m["i1"] + 1]
                ax, ctr = helix_axis(sub_pos)
                n_hat = membrane_normal(memb_sel, normal, near_point=ctr)
                if ax is None:
                    m["tilt"].append(np.nan)
                    m["rho"].append(np.nan)
                    continue
                t = tilt_angle(ax, n_hat)
                r = (rotation_angle(ax, ctr, sub_pos[m["ref_local"]], n_hat)
                     if t >= rho_min_tilt else np.nan)
                m["tilt"].append(t)
                m["rho"].append(r)
                axes.append(ax)
                points.append(ctr)

            if axes:
                # bundle axis: flip every director to the +normal side
                n0 = membrane_normal(memb_sel, normal,
                                     near_point=np.mean(points, axis=0))
                flipped = [a if np.dot(a, n0) >= 0 else -a for a in axes]
                bundle = _unit(np.mean(flipped, axis=0))
                tilts.append(tilt_angle(bundle, n0))
                if len(seg_meta) == 1:
                    rhos.append(seg_meta[0]["rho"][-1])
                else:
                    rhos.append(np.nan)
            else:
                tilts.append(np.nan)
                rhos.append(np.nan)

        elif mode == "ring":
            src = ca if ca is not None else biomol_sel.select_atoms(
                "name N CA C")
            pos = (src if src is not None and len(src) >= 3
                   else biomol_sel).positions
            nrm, ctr, planarity = ring_normal(pos)
            planarity_seen.append(planarity)
            n_hat = membrane_normal(memb_sel, normal, near_point=ctr)
            tilts.append(tilt_angle(nrm, n_hat))
            rhos.append(np.nan)

        elif mode == "principal":
            pos = (ca if ca is not None else biomol_sel).positions
            _, V = _principal_axes(pos)
            ctr = pos.mean(axis=0)
            n_hat = membrane_normal(memb_sel, normal, near_point=ctr)
            tilts.append(tilt_angle(V[:, 0], n_hat))
            rhos.append(np.nan)

        else:                                    # legacy end-to-end
            r1 = first_res.center_of_mass()
            r2 = last_res.center_of_mass()
            ctr = 0.5 * (r1 + r2)
            n_hat = membrane_normal(memb_sel, normal, near_point=ctr)
            tilts.append(tilt_angle(r2 - r1, n_hat))
            rhos.append(np.nan)

    # ---------------- finalise ----------------------------------------
    for m in seg_meta:
        m["tilt"] = np.asarray(m["tilt"], dtype=float)
        m["rho"] = np.asarray(m["rho"], dtype=float)
        m.pop("ref_local", None)

    if mode == "ring" and planarity_seen:
        pl = float(np.mean(planarity_seen))
        if pl < 2.0:
            _log(f"WARNING: the macrocycle is only weakly planar "
                 f"(lambda2/lambda3 = {pl:.1f}); its plane normal, and "
                 "hence the tilt, is poorly defined.")

    tilt_arr = np.asarray(tilts, dtype=float)
    if np.isfinite(tilt_arr).any() and float(np.nanstd(tilt_arr)) < 1e-3:
        _log("tilt is essentially constant over the trajectory.")

    return {
        "time": np.asarray(times, dtype=float),
        "tilt": tilt_arr,
        "rho": np.asarray(rhos, dtype=float),
        "mode": mode,
        "topology": topology,
        "normal": normal,
        "cyclic": bool(cyclic),
        "segments": seg_meta,
        "warnings": warn,
    }


# ======================================================================
# Inter-segment geometry (multipass bundles)
# ======================================================================
def segment_crossing_angles(u, biomol_sel, seg_meta, frame=None):
    """
    Angles between every pair of segment axes, N->C sense preserved, for
    one frame. Consecutive transmembrane helices of a bundle are
    antiparallel, so values near 180 degrees are expected.

    Returns a list of dicts: {"a", "b", "angle_deg"}.
    """
    ca = _ca_atoms(biomol_sel)
    if ca is None or len(seg_meta) < 2:
        return []
    if frame is not None:
        u.trajectory[frame]
    axes = []
    for m in seg_meta:
        ax, _ = helix_axis(ca.positions[m["i0"]:m["i1"] + 1])
        axes.append(ax)
    out = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if axes[i] is None or axes[j] is None:
                continue
            c = float(np.clip(np.dot(axes[i], axes[j]), -1.0, 1.0))
            out.append({"a": seg_meta[i]["name"], "b": seg_meta[j]["name"],
                        "angle_deg": round(_DEG * float(np.arccos(c)), 2)})
    return out


# ======================================================================
# Output
# ======================================================================
def write_orientation_csv(data, outfile):
    """Per-frame tilt (and rho) for the molecule and every segment."""
    import pandas as pd

    cols = {"time_ns": data["time"], "tilt_deg": data["tilt"]}
    if np.isfinite(data["rho"]).any():
        cols["rho_deg"] = data["rho"]
    for m in data["segments"]:
        tag = f'{m["name"]}_{m["resid_start"]}-{m["resid_end"]}'
        cols[f"tilt_{tag}"] = m["tilt"]
        if np.isfinite(m["rho"]).any():
            cols[f"rho_{tag}"] = m["rho"]
    pd.DataFrame(cols).to_csv(outfile, index=False)


def write_orientation_summary(data, outfile):
    """Human-readable summary with the definitions actually used."""
    lines = [
        "Biomolecule orientation in the bilayer",
        "=" * 56,
        f"topology        : {data['topology']}",
        f"axis definition : {data['mode']}",
        f"membrane normal : {data['normal']}",
        "",
        "theta : angle between the axis and the bilayer normal, folded",
        "        into [0, 90] deg (the axis is a director).",
        "rho   : rotation about the axis, measured anticlockwise from the",
        "        membrane-facing (down-tilt) direction to the reference",
        "        C-alpha; NaN when the helix is too close to upright for",
        "        that direction to be defined.",
        "Convention: Strandberg et al., Biophys. J. 86:3709 (2004), where",
        "the tilt is written tau; it is theta here because tau denotes the",
        "binding residence time (tau-off) elsewhere in this suite.",
        "",
    ]
    t = data["tilt"]
    if np.isfinite(t).any():
        lines.append(f"whole molecule  : theta = {np.nanmean(t):.2f} +/- "
                     f"{np.nanstd(t):.2f} deg")
    for m in data["segments"]:
        lines.append(
            f"  {m['name']} ({m['resid_start']}-{m['resid_end']}, "
            f"{m['n_res']} res): theta = {np.nanmean(m['tilt']):.2f} +/- "
            f"{np.nanstd(m['tilt']):.2f} deg"
            + (f", rho = {np.nanmean(m['rho']):.1f} deg "
               f"(ref C-alpha resid {m['ref_resid']})"
               if np.isfinite(m["rho"]).any() else ", rho undefined"))
    if data["warnings"]:
        lines += ["", "Notes:"] + [f"  - {w}" for w in data["warnings"]]
    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_tilt_rotation(data, outfile, segment=None):
    """
    Polar tilt-rotation map: radius = theta, angle = rho, colour = time.
    The direct analogue of the NMR "PISA wheel" -- a tight cluster means
    a well-defined orientation, a ring means the helix rotates freely
    about its own axis.

    Because the radius carries theta and the colour carries time, this
    single panel replaces a separate tilt time series for one segment.
    Every segment gets its own panel, so a multipass bundle shows the
    orientation of each transmembrane helix side by side.

    `segment` restricts the figure to one segment index; the default
    (None) draws them all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = data["segments"]
    if not segs:
        return None
    if segment is not None:
        segs = [segs[min(segment, len(segs) - 1)]]

    # keep only segments with enough defined (theta, rho) pairs
    usable = []
    for m in segs:
        ok = np.isfinite(m["tilt"]) & np.isfinite(m["rho"])
        if ok.sum() >= 5:
            usable.append((m, ok))
    if not usable:
        return None

    n = len(usable)
    ncols = 2 if n == 4 else min(3, n)      # 4 helices read better as 2x2
    nrows = int(np.ceil(n / ncols))
    rmax = max(float(np.nanmax(m["tilt"][ok])) for m, ok in usable)
    rmax = max(10.0, rmax * 1.15)

    fig = plt.figure(figsize=(4.7 * ncols + 1.1, 5.0 * nrows))
    sc = None
    for k, (m, ok) in enumerate(usable):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="polar")
        sc = ax.scatter(np.radians(m["rho"][ok]), m["tilt"][ok],
                        c=data["time"][ok], cmap="viridis", s=9, alpha=0.85)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)                # anticlockwise, as in GALA
        ax.set_rlabel_position(135)
        ax.set_ylim(0, rmax)                     # common scale across panels
        ax.set_title(f'{m["name"]} ({m["resid_start"]}-{m["resid_end"]})',
                     fontsize=11, fontweight="bold",
                     fontfamily="DejaVu Sans", pad=12)
    fig.suptitle(r"Tilt-rotation map:  radius = $\theta$,  angle = $\rho$"
                 "\n" r"($\rho$ = 0 faces the bilayer)",
                 fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
    # polar tick labels sit outside the axes, so panels need explicit
    # room or a lower row's title lands on the row above's 270 label
    fig.subplots_adjust(hspace=0.40, wspace=0.28,
                        top=0.86 if nrows > 1 else 0.80)
    cb = fig.colorbar(sc, ax=fig.axes, pad=0.06, shrink=0.75,
                      fraction=0.045)
    cb.set_label("Time (ns)", fontsize=10, fontfamily="DejaVu Sans")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outfile


# ======================================================================
# Stand-alone use:  python -m analysis.orientation -s a.tpr -f a.xtc
# ======================================================================
def _cli():
    import argparse
    import MDAnalysis as mda

    ap = argparse.ArgumentParser(
        description="Tilt (theta) and rotation (rho) of a biomolecule in a "
                    "bilayer, GALA convention (Strandberg 2004).")
    ap.add_argument("-s", required=True, help="topology (.tpr)")
    ap.add_argument("-f", default=None, help="trajectory (.xtc)")
    ap.add_argument("--sel", default="protein", help="biomolecule selection")
    ap.add_argument("--memb", default=None, help="membrane selection")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "helix", "segments", "ring",
                             "principal", "endtoend"])
    ap.add_argument("--segments", default=None, help='e.g. "6-33,41-68"')
    ap.add_argument("--ref-resid", type=int, default=None)
    ap.add_argument("--normal", default="z",
                    choices=["z", "global", "local"])
    ap.add_argument("-o", default="orientation", help="output prefix")
    a = ap.parse_args()

    from core.universe_loader import load_universe
    u = load_universe(a.s, a.f) if a.f else load_universe(a.s)
    bio = u.select_atoms(a.sel)
    memb = u.select_atoms(a.memb) if a.memb else None

    data = compute_orientation(u, bio, stride=a.stride, memb_sel=memb,
                               mode=a.mode, segments=a.segments,
                               ref_resid=a.ref_resid, normal=a.normal)
    write_orientation_csv(data, f"{a.o}.csv")
    write_orientation_summary(data, f"{a.o}.txt")
    if data["segments"]:
        plot_tilt_rotation(data, f"{a.o}_wheel.png")
    print(open(f"{a.o}.txt").read())


if __name__ == "__main__":
    _cli()
