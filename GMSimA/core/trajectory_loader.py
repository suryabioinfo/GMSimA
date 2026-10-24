import MDAnalysis as mda
import os
from core.leaflet_detection import detect_leaflets



# ---------------- CHARMM36 lipid library ----------------

CHARMM36_LIPIDS = {
    "POPC","POPE","POPS","POPG",
    "DPPC","DOPC","DMPC",
    "DPPS","DOPS","DSPS",
    "CHOL"
}


def load_trajectory_metadata(tpr_path, xtc_path):
    """
    Load trajectory using MDAnalysis and extract metadata
    for GUI display.
    """

    metadata = {}

    # ---------------- Load universe ----------------

    from core.universe_loader import load_universe, tpr_backend
    u = load_universe(tpr_path, xtc_path)
    metadata["tpr_reader"] = tpr_backend(u)

    # Only inspect first frame (fast)
    u.trajectory[0]

    # ---------------- Basic trajectory info ----------------

    metadata["tpr"] = os.path.basename(tpr_path)
    metadata["xtc"] = os.path.basename(xtc_path)

    metadata["atoms"] = len(u.atoms)
    metadata["frames"] = len(u.trajectory)

    timestep = u.trajectory.dt if hasattr(u.trajectory, "dt") else None

    if timestep:
        metadata["timestep_ps"] = timestep
        metadata["sim_length_ns"] = round((metadata["frames"] * timestep)/1000, 2)
    else:
        metadata["timestep_ps"] = "Unknown"
        metadata["sim_length_ns"] = "Unknown"

    # ---------------- Box size ----------------
    # Formatted strings (nm, 3 decimals): the raw values are float32,
    # which f-strings would otherwise print with ~16 meaningless digits.

    box = u.dimensions

    metadata["box"] = (
        f"{float(box[0])/10.0:.3f}",
        f"{float(box[1])/10.0:.3f}",
        f"{float(box[2])/10.0:.3f}",
    )

    # ---------------- Lipid detection (family-aware) ----------------
    # Uses the central registry (CHARMM36 one-residue lipids AND the
    # AMBER Lipid14/17/21 three-residue family), not the old local
    # CHARMM-only list — so AMBER systems report their composition too.

    from core.lipids import detect_ff_family, detect_lipids

    lipids = detect_lipids(u)
    metadata["lipids"] = lipids

    # ---------------- Leaflet detection ----------------
    leaflets = detect_leaflets(u)

    metadata["upper_leaflet_lipids"] = leaflets["upper_count"]
    metadata["lower_leaflet_lipids"] = leaflets["lower_count"]
    metadata["upper_chol"] = leaflets["upper_chol"]
    metadata["lower_chol"] = leaflets["lower_chol"]

    # ---------------- Forcefield estimation ----------------
    # A .tpr does not store the force-field name/version, so this is an
    # honest inference from the lipid residue-naming convention — and
    # it is labeled as such.

    family = detect_ff_family(u)
    if family == "amber":
        metadata["forcefield"] = ("AMBER Lipid14/17/21 family "
                                  "(inferred from lipid naming)")
    elif family == "charmm" or len(lipids) > 0:
        metadata["forcefield"] = ("CHARMM36 family "
                                  "(inferred from lipid naming)")
    else:
        metadata["forcefield"] = "Unrecognized (no registry lipids found)"

    # ---------------- Protein detection ----------------

    protein = u.select_atoms("protein")

    metadata["protein_atoms"] = len(protein)

    if len(protein) > 0:
        metadata["protein_present"] = True
    else:
        metadata["protein_present"] = False

    return metadata