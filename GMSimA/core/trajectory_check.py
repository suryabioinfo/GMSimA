"""
smGROANA: warn-only trajectory preprocessing sanity checks.

smGROANA expects trajectories preprocessed with:

    gmx trjconv -s a.tpr -f a.xtc -pbc mol -center -ur compact -o b.xtc
    (optionally followed by a rot+trans fit)

Within the larger suite this preprocessing is performed by an upstream
stage, so these checks NEVER block execution: they return a list of
human-readable warning strings that the worker logs and displays. An
empty list means no anomaly detected.

The screen inspects the FIRST, MIDDLE and LAST frames, not just the
first: defects that grow with time — most importantly a '-pbc nojump'
style trajectory whose coordinates drift out of the box as lipids
diffuse — are invisible at t=0 and only show up late. The phosphate
xy-footprint check below was added after exactly such a case slipped
through a first-frame-only screen.
"""

import numpy as np


def _frame_checks(universe, box, frame_label):
    """Checks evaluated on the currently loaded frame."""
    warnings = []
    half_box = 0.5 * np.array(box[:3])

    # ------------------------------------------------------------
    # 1. Broken molecules (i.e. '-pbc mol' probably NOT applied)
    #    A whole residue should never span more than half the box.
    # ------------------------------------------------------------
    broken = 0
    checked = 0
    for res in universe.residues:
        pos = res.atoms.positions
        if len(pos) < 3:
            continue
        extent = pos.max(axis=0) - pos.min(axis=0)
        if np.any(extent > half_box):
            broken += 1
        checked += 1
        if checked >= 500:  # sampling is sufficient for a warning
            break
    if broken > 0:
        warnings.append(
            f"{broken} residue(s) appear split across the periodic box "
            f"({frame_label}). The trajectory does not look '-pbc mol' "
            "processed; run 'gmx trjconv -pbc mol -center -ur compact' "
            "before analysis.")

    p_atoms = universe.select_atoms("name P")
    if len(p_atoms) >= 10:
        coords = p_atoms.positions

        # --------------------------------------------------------
        # 2. Coordinates confined to the box (xy footprint).
        #    On a properly wrapped trajectory the phosphate cloud
        #    spans at most the lateral box. A footprint clearly
        #    larger than the box is the signature of un-rewrapped
        #    ('-pbc nojump'-style) coordinates: lipids diffuse out
        #    of the box and geometry-based modules (curvature,
        #    Voronoi APL) degrade progressively.
        # --------------------------------------------------------
        for axis, label in ((0, "x"), (1, "y")):
            extent = coords[:, axis].max() - coords[:, axis].min()
            if box[axis] > 0 and extent > 1.15 * box[axis]:
                warnings.append(
                    f"Phosphate {label}-footprint ({extent:.0f} Å) exceeds "
                    f"the box ({box[axis]:.0f} Å) at {frame_label}: "
                    "coordinates are not wrapped into the box "
                    "('-pbc nojump'-like trajectory). Curvature and "
                    "Voronoi-APL results will be unreliable; re-wrap with "
                    "'gmx trjconv -pbc mol -center -ur compact'.")
                break

        # --------------------------------------------------------
        # 3. Membrane normal alignment with z (rot+trans fit check)
        # --------------------------------------------------------
        mid = coords[:, 2].mean()
        upper = coords[coords[:, 2] > mid]
        ref = upper if len(upper) >= 10 else coords
        centered = ref - ref.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        tilt_deg = np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0)))
        if tilt_deg > 10.0:
            warnings.append(
                f"Membrane normal deviates {tilt_deg:.1f}° from the z-axis "
                f"({frame_label}). Thickness/ΔZ/tilt analyses assume a "
                "z-aligned membrane; consider a rot+trans fitted trajectory.")

        # --------------------------------------------------------
        # 4. Bilayer roughly centered in the box (z)
        # --------------------------------------------------------
        z_center_offset = abs(coords[:, 2].mean() - 0.5 * box[2])
        if z_center_offset > 0.25 * box[2]:
            warnings.append(
                "Bilayer midplane is far from the box center "
                f"(offset {z_center_offset:.1f} Å, {frame_label}). "
                "'-center' preprocessing may be missing; density/ΔZ "
                "profiles may be shifted.")

    # ------------------------------------------------------------
    # (removed) Protein near the box edge in xy: this was a false
    # alarm — a peptide diffusing laterally over a membrane-centered
    # trajectory legitimately visits the box edges, and every contact/
    # distance in smGROANA is computed minimum-image (PBC-correct), so
    # nothing is under-counted. The check produced noise on properly
    # prepared runs and was dropped.
    # ------------------------------------------------------------

    return warnings


def check_trajectory_preprocessing(universe):
    """
    Return a list of warning strings (empty list = all checks passed).
    Screens the first, middle and last frames; duplicate findings from
    different frames are collapsed (the first occurrence is kept, so
    the frame label tells WHERE the defect first becomes visible).
    """
    all_warnings = []
    seen_kinds = set()

    try:
        n_frames = len(universe.trajectory)
    except Exception:
        return all_warnings

    if n_frames < 1:
        return all_warnings
    frame_ids = sorted({0, n_frames // 2, n_frames - 1})

    for fi in frame_ids:
        try:
            universe.trajectory[fi]
        except Exception:
            continue

        box = universe.dimensions
        if box is None or box[0] <= 0:
            if "nobox" not in seen_kinds:
                seen_kinds.add("nobox")
                all_warnings.append(
                    "Trajectory carries no box information; PBC-dependent "
                    "analyses (distances, contacts) may be unreliable.")
            continue

        label = ("first frame" if fi == 0
                 else "last frame" if fi == n_frames - 1
                 else f"frame {fi} (midpoint)")
        for w in _frame_checks(universe, box, label):
            # collapse duplicates by warning type (text before the
            # parenthesized frame label varies only in numbers)
            kind = w.split("(")[0][:40]
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                all_warnings.append(w)

    # rewind so downstream modules start from the beginning
    try:
        universe.trajectory[0]
    except Exception:
        pass

    return all_warnings
