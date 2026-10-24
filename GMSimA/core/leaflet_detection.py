import numpy as np
from MDAnalysis.analysis.leaflet import LeafletFinder
from core.cholesterol_assignment import assign_cholesterol_leaflets
from core.lipids import phosphate_selection


def detect_leaflets(universe):
    """
    Detect membrane leaflets from the lipid phosphorus atoms and select
    the two largest clusters.

    Works with CHARMM36 (one-residue lipids, atom 'P') and AMBER
    Lipid14/17/21 (three-residue lipids, atom 'P31') via the
    family-aware phosphate selection.
    """

    result = {}

    # ---------------- Select phosphate atoms ----------------

    phosphate_atoms = universe.select_atoms(phosphate_selection(universe))

    if len(phosphate_atoms) == 0:
        raise ValueError(
            "No lipid phosphorus atoms detected (looked for CHARMM 'P' "
            "and AMBER 'P31' conventions). Cannot determine leaflets."
        )

    # ---------------- Run leaflet clustering ----------------

    leaflets = LeafletFinder(universe, phosphate_atoms)

    groups = leaflets.groups()

    if len(groups) < 2:
        raise ValueError("Leaflet detection failed: fewer than two clusters detected.")

    # ---------------- Sort clusters by size ----------------
    # This ensures robustness when small clusters appear

    groups_sorted = sorted(groups, key=lambda g: len(g), reverse=True)

    leaflet1 = groups_sorted[0]
    leaflet2 = groups_sorted[1]

    # ---------------- Determine upper/lower leaflet ----------------

    z1 = leaflet1.center_of_mass()[2]
    z2 = leaflet2.center_of_mass()[2]

    if z1 > z2:
        upper = leaflet1
        lower = leaflet2
    else:
        upper = leaflet2
        lower = leaflet1

    # ---------------- Cholesterol assignment ----------------

    chol = assign_cholesterol_leaflets(universe, upper, lower)

    # ---------------- Store results ----------------

    result["upper_leaflet"] = upper
    result["lower_leaflet"] = lower

    result["upper_count"] = len(upper.residues)
    result["lower_count"] = len(lower.residues)

    result["upper_chol"] = chol["upper_chol"]
    result["lower_chol"] = chol["lower_chol"]

    # ---------------- Leaflet symmetry check ----------------

    total = result["upper_count"] + result["lower_count"]

    imbalance = abs(result["upper_count"] - result["lower_count"]) / total

    if imbalance > 0.1:
        print("WARNING: Leaflet lipid imbalance detected.")

    return result