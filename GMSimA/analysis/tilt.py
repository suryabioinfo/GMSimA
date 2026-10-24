import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import minimum_filter


def compute_tilt_timeseries(u, biomol_sel, stride, memb_sel=None,
                            axis_mode="auto", segments=None,
                            ref_resid=None, normal="z"):
    """Peptide/protein tilt vs the membrane normal.

    Since v3.3 the axis is chosen from the topology (helix fit, ring
    normal, per-segment for multipass) following the GALA convention of
    Strandberg et al., Biophys. J. 86:3709 (2004). Pass axis_mode=
    "endtoend" to reproduce v2/v3 numbers exactly.
    """
    from analysis.orientation import compute_orientation
    return compute_orientation(u, biomol_sel, stride=stride,
                               memb_sel=memb_sel, mode=axis_mode,
                               segments=segments, ref_resid=ref_resid,
                               normal=normal)
    
    # if len(biomol_sel.residues) < 2:
    #     raise ValueError("Biomolecule must have at least 2 residues to compute tilt.")

    # first_res = biomol_sel.residues[0].atoms
    # last_res = biomol_sel.residues[-1].atoms
    # membrane_normal = np.array([0.0, 0.0, 1.0])
    # times = []
    # tilts = []
    
    # for ts in u.trajectory[::stride]:
    #     r1 = first_res.center_of_mass()
    #     r2 = last_res.center_of_mass()
    #     vec = r2 - r1
    #     vec = vec / np.linalg.norm(vec)
    #     cos_theta = np.dot(vec, membrane_normal)
    #     cos_theta = np.clip(cos_theta, -1.0, 1.0)
    #     theta = np.degrees(np.arccos(cos_theta))
    #     # fold angles into 0–90 range
    #     if theta > 90:
    #         theta = 180 - theta
    #     times.append(ts.time / 1000.0)   # ns
    #     tilts.append(theta)
    # return {
    #     "time": np.array(times),
    #     "tilt": np.array(tilts)
    # }


# ---------------------------------------------------------
# XVG writer
# ---------------------------------------------------------

def write_tilt_xvg(data, outfile):

    with open(outfile, "w") as f:
        f.write("@    title \"Peptide Tilt Angle\"\n")
        f.write("@    xaxis  label \"Time (ns)\"\n")
        f.write("@    yaxis  label \"Tilt Angle (deg)\"\n")
        for t, tilt in zip(data["time"], data["tilt"]):
            f.write(f"{t:.3f} {tilt:.3f}\n")


# ---------------------------------------------------------
# Time series plot
# ---------------------------------------------------------

def plot_tilt_timeseries(data, outfile):

    time = data["time"]
    tilt = data["tilt"]
    plt.figure(figsize=(6,4))
    plt.plot(time, tilt, color="black", linewidth=1.5)
    plt.axhline(y=90, linestyle="--", color="gray", linewidth=1)
    plt.xlim(0, data["time"].max())
    plt.ylim(0, 100)
    plt.xlabel("Time (ns)", fontfamily="DejaVu Sans", fontsize=12)
    plt.ylabel(r"Tilt Angle ($\theta$°)", fontfamily="DejaVu Sans", fontsize=12)
    plt.title("Peptide Tilt Against Membrane Normal", fontfamily="DejaVu Sans",fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ---------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------

def plot_tilt_distribution(data, outfile):

    tilt = data["tilt"]
    plt.figure(figsize=(5,4))
    plt.hist(tilt,bins=10,density=True,color="gray",edgecolor="black")
    plt.xlabel(r"Tilt Angle ($\theta$°)", fontfamily="DejaVu Sans", fontsize=12)
    plt.ylabel("Probability Density", fontfamily="DejaVu Sans", fontsize=12)
    plt.title("Tilt Angle Distribution", fontfamily="DejaVu Sans", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


  
# --------------------------------- Metastable States from the FEL -----------------------------------

def detect_metastable_states(FEL, X, Y, energy_cutoff=2.0, min_dist_scaled=0.35):

    local_min = (FEL == minimum_filter(FEL, size=7))
    candidates = np.where(local_min)

    # Scale factors so the minima-separation test is dimensionless.
    # Tilt (degrees) and ΔZ (Å) must not be mixed in one Euclidean
    # metric with raw units; distances are measured in units of the
    # spread of each coordinate instead.
    sx = max(X.max() - X.min(), 1e-8)
    sy = max(Y.max() - Y.min(), 1e-8)

    states = []
    for i, j in zip(candidates[0], candidates[1]):
        energy = FEL[i, j]
        # energy filter
        if energy > energy_cutoff:
            continue
        x = X[i, j]
        y = Y[i, j]
        # distance filter (avoid clustered minima) in scaled units
        keep = True
        for s in states:
            dist = np.sqrt(((x - s["tilt"]) / sx)**2
                           + ((y - s["deltaz"]) / sy)**2)
            if dist < min_dist_scaled:
                keep = False
                break
        if keep:
            states.append({"tilt": x, "deltaz": y, "energy": energy})
    return states

# ---------------------------------------------------------
# Tilt vs Penetration FEL
# ---------------------------------------------------------
def plot_tilt_deltaz_fel(tilt_data, deltaz_data, outfile, return_states=False):


    tilt = np.array(tilt_data["tilt"])
    dz = np.array(deltaz_data["deltaz"])
    n = min(len(tilt), len(dz))
    tilt = tilt[:n]
    dz = dz[:n]

    # A (nearly) constant observable makes the 2D KDE covariance
    # singular — e.g. a completely rigid binding mode. Skip the FEL
    # gracefully instead of crashing the whole tilt module.
    if n < 10 or np.std(tilt) < 1e-3 or np.std(dz) < 1e-3:
        print("FEL skipped: tilt and/or ΔZ nearly constant "
              "(single rigid binding mode).")
        if return_states:
            return []
        return None

    # Create evaluation grid
    xi = np.linspace(tilt.min(), tilt.max(), 200)
    yi = np.linspace(dz.min(), dz.max(), 200)
    X, Y = np.meshgrid(xi, yi)
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([tilt, dz])

    # KDE probability estimate
    kernel = gaussian_kde(values, bw_method=0.16)
    Z = np.reshape(kernel(positions), X.shape)

    # Free energy
    kBT = 0.593
    FEL = -kBT * np.log(Z)
    FEL -= np.nanmin(FEL)
    # Clip unrealistic high energy
    max_energy = np.nanpercentile(FEL,90)
    FEL[FEL > max_energy] = max_energy

    plt.figure(figsize=(6,5))
    im = plt.imshow(FEL, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin="lower", aspect="auto", cmap="jet")
    # contour lines
    plt.contour(X, Y, FEL, levels=np.arange(0,6,0.5),
                colors="#505050", linewidths=0.5)
    #plt.xlim(0, 90)
    #plt.ylim(0, 40)
    # Now identify metastable states
    states = detect_metastable_states(FEL, X, Y)
    # for s in states:
    #     plt.scatter(s["tilt"], s["deltaz"], color="white", edgecolor="red", s=120, marker="*", zorder=5)

    for i, s in enumerate(states):
        plt.scatter(s["tilt"], s["deltaz"], color="white", edgecolor="red", s=120, marker="*", zorder=5)
        # label state
        plt.text(s["tilt"] + 0.5, s["deltaz"] + 0.25, f"{i}", fontsize=10,
                 fontfamily="DejaVu Sans", fontweight="bold", color="black",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                           pad=1.5))
    plt.xlabel(r"Tilt Angle ($\theta$°)", fontfamily="DejaVu Sans", fontsize=12, fontweight="normal")
    plt.ylabel(r"$\Delta Z$ ($\AA$)", fontfamily="DejaVu Sans", fontsize=12, fontweight="normal")
    plt.title("Tilt–Penetration FEL", fontfamily="DejaVu Sans", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im)
    cbar.set_label(r"Free Energy (kcal$\cdot$mol$^{-1}$)", fontweight ="bold", fontstyle="italic",  fontsize=10, fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

    if return_states:
        return states

# --------------------- Extract State Representative Structures -----------------------
def export_state_structures(u, tilt, deltaz, states, biomol_sel, output_prefix="state", stride=1):
    """Not called by the GUI pipeline since v3.0:
     KDE-state structures are exported only on explicit user request — see tilt_states.csv for the frame indices."""
    for i, s in enumerate(states):
        center = np.array([s["tilt"], s["deltaz"]])
        coords = np.vstack([tilt, deltaz]).T
        d = np.sum((coords - center)**2, axis=1)
        # tilt/deltaz were sampled every `stride` frames, so index i in the
        # analysis arrays corresponds to trajectory frame i*stride. Without
        # this mapping the exported PDB comes from the wrong frame whenever
        # stride > 1.
        frame = int(np.argmin(d)) * int(stride)
        frame = min(frame, len(u.trajectory) - 1)
        u.trajectory[frame]
        outfile = f"{output_prefix}_state{i}.pdb"

        # biomol_sel.write(outfile)
        system = u.atoms
        system.write(outfile)

        with open(outfile, "r+") as f:
            content = f.read()
            f.seek(0,0)
            f.write(f"TITLE     smGROANA FEL STATE {i} Trajectory Frame:{frame}\n" + content)




# -----------------------------------------------------------------------
#            FEL State Analysis (replaces the former MSM module)
#
# Purpose: identify well-separated basins of the tilt–ΔZ free-energy
# landscape and report their occupancies, so the exported representative
# structures let a reader see the peptide–membrane interaction pattern
# directly. (The previous MSM layer — transition matrices, lag times,
# lifetimes, MFPTs, pathways — was removed as its implementation was
# not kinetically valid; equilibrium state populations from the FEL are
# what the structure export actually needs.)
# -----------------------------------------------------------------------

def assign_states(tilt, deltaz, states):
    """
    Assign each frame to the nearest FEL minimum (Voronoi in
    standardized coordinates — tilt is in degrees and ΔZ in Å, so a raw
    Euclidean metric would weight the observables by their units).
    """
    tilt = np.asarray(tilt, dtype=float)
    deltaz = np.asarray(deltaz, dtype=float)
    sx = max(np.std(tilt), 1e-8)
    sy = max(np.std(deltaz), 1e-8)

    centers = np.array([[s["tilt"] / sx, s["deltaz"] / sy] for s in states])
    points = np.column_stack([tilt / sx, deltaz / sy])

    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)


def analyze_states(tilt, deltaz, states, stride=1, output_prefix="tilt_states"):
    """
    Occupancy and characterization of each FEL state; writes
    <output_prefix>.csv with one row per state:
    state, tilt_center, deltaz_center, energy, population_percent,
    mean_tilt, mean_deltaz, representative_frame (trajectory frame of
    the exported PDB, stride mapping included).
    """
    if not states:
        return None
    tilt = np.asarray(tilt, dtype=float)
    deltaz = np.asarray(deltaz, dtype=float)
    n = min(len(tilt), len(deltaz))
    tilt, deltaz = tilt[:n], deltaz[:n]

    ids = assign_states(tilt, deltaz, states)

    rows = []
    for i, s in enumerate(states):
        mask = ids == i
        pop = 100.0 * mask.sum() / len(ids)
        center = np.array([s["tilt"], s["deltaz"]])
        d = np.sum((np.column_stack([tilt, deltaz]) - center) ** 2, axis=1)
        rep_frame = int(np.argmin(d)) * int(stride)
        rows.append({
            "state": i,
            "tilt_center_deg": round(float(s["tilt"]), 2),
            "deltaz_center_A": round(float(s["deltaz"]), 2),
            "energy_kcal_mol": round(float(s["energy"]), 3),
            "population_percent": round(pop, 2),
            "mean_tilt_deg": (round(float(tilt[mask].mean()), 2)
                              if mask.any() else float("nan")),
            "mean_deltaz_A": (round(float(deltaz[mask].mean()), 2)
                              if mask.any() else float("nan")),
            "representative_frame": rep_frame,
        })

    pd.DataFrame(rows).to_csv(f"{output_prefix}.csv", index=False)
    return rows
