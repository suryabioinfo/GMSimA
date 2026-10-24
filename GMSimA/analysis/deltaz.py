import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------
# ΔZ penetration calculation
# ---------------------------------------------------------

def _local_plane_z(p_pos, leaflet_mask, com, box_xy, local_radius):
    """
    Mean z of the phosphates of one leaflet within `local_radius` of the
    peptide COM laterally (minimum-image in xy). Falls back to the whole
    leaflet plane when fewer than 5 local phosphates are found.
    """
    leaflet = p_pos[leaflet_mask]
    d = leaflet[:, :2] - com[:2]
    d -= np.round(d / box_xy) * box_xy       # minimum image, lateral
    near = (d[:, 0]**2 + d[:, 1]**2) <= local_radius**2
    if near.sum() >= 5:
        return leaflet[near, 2].mean(), True
    return leaflet[:, 2].mean(), False


def calculate_deltaz(universe, stride=1, protein_sel="protein",
                     phosphate_sel="name P", local_radius=15.0,
                     per_residue=True):
    """
    Peptide ΔZ relative to the bilayer MIDPLANE (the original smGROANA
    convention: ΔZ = COM z − mean phosphate z). This is the primary
    observable — read against the membrane reference lines (midplane 0,
    phosphate planes at ±thickness/2) it shows at a glance where the
    peptide anchors.

    A SECONDARY value, 'deltaz_local', is also computed: depth relative
    to the local phosphate plane of the proximal leaflet (phosphates
    within `local_radius` Å laterally; positive = inserted past the
    headgroups). It captures the local membrane deformation under the
    peptide and is written as an extra column in the .xvg.
    """
    protein = universe.select_atoms(protein_sel)
    lipids = universe.select_atoms(phosphate_sel)
    residues = protein.residues

    times = []
    dz_mid = []           # PRIMARY: COM z - global midplane
    depth_local = []      # secondary: local proximal-plane insertion depth
    leaflet_used = []     # "upper"/"lower" per frame
    local_ok_frames = 0

    res_acc = [[] for _ in residues] if per_residue else None

    for ts in universe.trajectory[::stride]:

        p_pos = lipids.positions
        box_xy = ts.dimensions[:2]
        box_z = float(ts.dimensions[2])

        def _mic_z(d):
            # minimum image along z: a whole molecule sitting across the
            # periodic z-boundary (as -pbc mol legitimately allows) can
            # put its COM outside [0, Lz], producing spurious |ΔZ| that
            # even exceeds Lz/2; folding into [-Lz/2, +Lz/2) restores
            # the physically meaningful signed distance to the midplane.
            return d - box_z * np.round(d / box_z)

        midplane = p_pos[:, 2].mean()
        upper_mask = p_pos[:, 2] > midplane
        lower_mask = ~upper_mask

        com = protein.center_of_mass()
        dz_mid.append(_mic_z(com[2] - midplane))

        # proximal leaflet = the one whose mean plane is nearest the COM
        # (minimum-image distances, so a wrapped COM picks correctly)
        z_up = p_pos[upper_mask, 2].mean()
        z_lo = p_pos[lower_mask, 2].mean()
        use_upper = abs(_mic_z(com[2] - z_up)) <= abs(_mic_z(com[2] - z_lo))
        mask = upper_mask if use_upper else lower_mask
        leaflet_used.append("upper" if use_upper else "lower")

        plane_z, local_ok = _local_plane_z(p_pos, mask, com, box_xy,
                                           local_radius)
        if local_ok:
            local_ok_frames += 1
        depth_local.append(_mic_z(plane_z - com[2]) if use_upper
                           else _mic_z(com[2] - plane_z))

        if per_residue:
            # per-residue ΔZ in the SAME midplane convention as the
            # primary observable, so both plots read identically
            for i, res in enumerate(residues):
                res_acc[i].append(
                    _mic_z(res.atoms.center_of_mass()[2] - midplane))

        times.append(ts.time / 1000.0)

    result = {
        "time": np.array(times),
        "deltaz": np.array(dz_mid),
        "deltaz_local": np.array(depth_local),
        "leaflet": leaflet_used,
        "local_radius": local_radius,
        "local_plane_fraction": local_ok_frames / max(len(times), 1),
    }
    if per_residue:
        result["per_residue"] = {
            "resids": residues.resids.copy(),
            "resnames": residues.resnames.copy(),
            "mean": np.array([np.mean(a) for a in res_acc]),
        }
    return result


# ---------------------------------------------------------
# Plot ΔZ
# ---------------------------------------------------------

def plot_deltaz(data, outfile, thickness=None):

    time = data["time"]
    dz = data["deltaz"]

    plt.figure(figsize=(6,4))

    plt.plot(time, dz, color="black", linewidth=1, label="Penetration Depth")

    plt.xlim(0, data["time"].max())
    plt.xlabel("Time (ns)", fontsize=12, fontfamily="DejaVu Sans")
    plt.ylabel(r"$\Delta Z$ ($\AA$)", fontsize=12, fontfamily="DejaVu Sans")
    plt.title("Protein Penetration Depth", fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")

    # Membrane thickness annotation (original smGROANA presentation)
    if thickness is not None:
        upper_phosphate = thickness / 2
        lower_phosphate = -thickness / 2
        upper_leaflet = thickness / 4
        lower_leaflet = -thickness / 4
        plt.axhline(upper_phosphate, linestyle="--", color="red", label="Phosphate Plane")
        plt.axhline(lower_phosphate, linestyle="--", color="red")
        plt.axhline(upper_leaflet, linestyle="--", color="gray", label="Membrane Leaflet")
        plt.axhline(lower_leaflet, linestyle="--", color="gray")

    plt.axhline(0, linestyle="-", color="#1f77b4", label="Membrane Midplane")
    #plt.legend(prop={'family': 'DejaVu Sans', 'size': 10}, loc='upper right', frameon=True, labelspacing=0.15)
    plt.tight_layout()

    plt.savefig(outfile, dpi=300)

    plt.close()


# ---------------------------------------------------------
# Write XVG
# ---------------------------------------------------------

def write_deltaz_xvg(data, outfile):

    time = data["time"]
    dz = data["deltaz"]

    dz_local = data.get("deltaz_local")

    with open(outfile, "w") as f:

        f.write('@ title "Protein Penetration Depth"\n')
        f.write('# deltaz: COM z minus bilayer midplane (primary, original '
                'convention)\n')
        if dz_local is not None:
            f.write('# deltaz_local: depth vs LOCAL phosphate plane of the '
                    'proximal leaflet (positive = inserted past headgroups; '
                    f'lateral radius {data.get("local_radius", "n/a")} A, '
                    'local plane used in '
                    f'{100*data.get("local_plane_fraction", 0):.0f}% of '
                    'frames)\n')
        f.write('@ xaxis label "Time (ns)"\n')
        f.write('@ yaxis label "Delta Z (Angstrom)"\n')
        if dz_local is not None:
            f.write('# time deltaz deltaz_local\n')
            for t, z, dl in zip(time, dz, dz_local):
                f.write(f"{t:10.4f} {z:10.4f} {dl:10.4f}\n")
        else:
            f.write('# time deltaz\n')
            for t, z in zip(time, dz):
                f.write(f"{t:10.4f} {z:10.4f}\n")


# ---------------------------------------------------------
# Per-residue insertion profile
# ---------------------------------------------------------

def write_deltaz_per_residue_csv(data, outfile):
    pr = data.get("per_residue")
    if pr is None:
        return
    with open(outfile, "w") as f:
        f.write("# mean residue COM z minus bilayer midplane (Angstrom)\n")
        f.write("resid,resname,mean_deltaz_A\n")
        for ri, rn, m in zip(pr["resids"], pr["resnames"], pr["mean"]):
            f.write(f"{ri},{rn},{m:.4f}\n")


def plot_deltaz_per_residue(data, outfile, thickness=None):
    """
    Mean per-residue ΔZ in the same midplane convention as the main
    plot, drawn as a profile line against the membrane reference lines
    so anchoring residues are read the same way. No error bars — frame
    scatter is visible in the time series; here only the mean position
    matters.
    """
    pr = data.get("per_residue")
    if pr is None:
        return
    n = len(pr["resids"])
    plt.figure(figsize=(max(6, 0.28 * n), 4))
    x = np.arange(n)
    plt.plot(x, pr["mean"], color="black", lw=1.2, marker="o", ms=3.5)

    if thickness is not None:
        plt.axhline(thickness / 2, linestyle="--", color="red", lw=1,
                    label="Phosphate Plane")
        plt.axhline(-thickness / 2, linestyle="--", color="red", lw=1)
        plt.axhline(thickness / 4, linestyle="--", color="gray", lw=1,
                    label="Membrane Leaflet")
        plt.axhline(-thickness / 4, linestyle="--", color="gray", lw=1)
    plt.axhline(0, linestyle="-", color="#1f77b4", lw=1,
                label="Membrane Midplane")

    labels = [f"{rn}{ri}" for rn, ri in zip(pr["resnames"], pr["resids"])]
    plt.xticks(x, labels, rotation=90, fontsize=7, fontfamily="DejaVu Sans")
    plt.ylabel(r"Mean $\Delta Z$ ($\AA$)", fontsize=12,
               fontfamily="DejaVu Sans")
    plt.title("Per-residue Membrane Position", fontsize=12,
              fontweight="bold", fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

# ---------------------------------------------------------
# Plot ΔZ density
# ---------------------------------------------------------

def plot_deltaz_density(data, outfile, surface_z=None):
    """
    Plots smoothed density with an optional horizontal line for the membrane surface.
    
    Parameters:
    - data: Dictionary/DataFrame containing 'time' and 'deltaz'
    - outfile: String path for the output image
    - surface_z: Float, the average Z position of the membrane surface
    """
    time = np.array(data["time"])
    dz = np.array(data["deltaz"])

    # Create bins - 200 provides a smoother gradient for the blur
    t_bins = 200
    z_bins = 200

    H, xedges, yedges = np.histogram2d(
        time,
        dz,
        bins=[t_bins, z_bins]
    )

    H = H.T  # Transpose for plotting

    # Apply Gaussian Blur (sigma=1.5 to 2.0 is usually ideal for this data)
    H_smoothed = gaussian_filter(H, sigma=2.0)
    #H_smoothed = H_smoothed / H_smoothed.sum()  # Normalize to get density

    plt.figure(figsize=(6, 4))
    # z_limit = max(abs(dz.min()), abs(dz.max()))
    # z_limit = np.ceil(z_limit / 5) * 5  # Round up to nearest 5 for cleaner limits

    im = plt.imshow(
        H_smoothed,
        extent=[time.min(), time.max(), dz.min(), dz.max()],
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )

    # Add the average membrane surface line
    if surface_z is not None:
        # Upper leaflet surface
        plt.axhline(y=surface_z, color='white', linestyle='--', linewidth=1.5, label='Membrane Surface')
       

        #plt.legend(prop={'family': 'DejaVu Sans', 'size': 10}, loc='upper right')

    cbar = plt.colorbar(im)
    cbar.set_label("Density", fontsize=12, fontfamily="DejaVu Sans")
    
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily("DejaVu Sans")

    # Labels and Title
    plt.xlabel("Time (ns)", fontsize=12, fontfamily="DejaVu Sans")
    plt.ylabel(r"$\Delta Z$ ($\AA$)", fontsize=12, fontfamily="DejaVu Sans")
    plt.title("Peptide Penetration Depth Density", fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
# def plot_deltaz_density(data, outfile):

#     time = np.array(data["time"])
#     dz = np.array(data["deltaz"])

#     # create bins
#     t_bins = 100
#     z_bins = 100

#     H, xedges, yedges = np.histogram2d(
#         time,
#         dz,
#         bins=[t_bins, z_bins]
#     )

#     H = H.T  # transpose for plotting

#     plt.figure(figsize=(10,4))

#     plt.imshow(
#         H,
#         extent=[time.min(), time.max(), dz.min(), dz.max()],
#         aspect='auto',
#         origin='lower',
#         cmap='jet'
#     )
#     cbar = plt.colorbar()
#     cbar.set_label("Density", fontsize=12, fontfamily="DejaVu Sans")

#     #plt.colorbar(label="Density", fontsize=12, fontfamily="DejaVu Sans")

#     plt.xlabel("Time (ns)", fontsize=12, fontfamily="DejaVu Sans")
#     plt.ylabel(r"$\Delta Z$ ($\AA$)", fontsize=12, fontfamily="DejaVu Sans")
#     plt.title("Peptide penetration depth density", fontsize=14, fontfamily="DejaVu Sans")

#     plt.tight_layout()
#     plt.savefig(outfile, dpi=300)
#     plt.close()