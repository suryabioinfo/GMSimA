import numpy as np


def assign_cholesterol_leaflets(universe, upper_leaflet, lower_leaflet):
    """
    Assign cholesterol molecules to the closest leaflet
    based on center-of-mass Z distance.
    """

    result = {}

    # ---------------- detect cholesterol ----------------
    # CHARMM: CHL1/CHOL; AMBER Lipid family: CHL; ergosterol: ERG

    chol_residues = universe.select_atoms(
        "resname CHOL CHL1 CHL ERG").residues

    upper_chol = []
    lower_chol = []

    if len(chol_residues) == 0:

        result["upper_chol"] = 0
        result["lower_chol"] = 0
        return result

    # ---------------- leaflet COM ----------------

    upper_com = upper_leaflet.center_of_mass()
    lower_com = lower_leaflet.center_of_mass()

    z_upper = upper_com[2]
    z_lower = lower_com[2]

    # ---------------- assign cholesterol ----------------

    for res in chol_residues:

        com = res.atoms.center_of_mass()

        dist_upper = abs(com[2] - z_upper)
        dist_lower = abs(com[2] - z_lower)

        if dist_upper < dist_lower:
            upper_chol.append(res)
        else:
            lower_chol.append(res)

    result["upper_chol"] = len(upper_chol)
    result["lower_chol"] = len(lower_chol)

    return result