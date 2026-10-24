import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless environments (thread-safe backend)
import matplotlib.pyplot as plt


def calculate_apl_timeseries(universe, upper_leaflet, lower_leaflet, stride):
    """
    Compute APL for each frame of the trajectory.

    Assumes membrane lies in XY plane.
    Works with GROMACS-fitted trajectories.
    """

    times = []
    apl_upper = []
    apl_lower = []

    n_upper = len(upper_leaflet.residues)
    n_lower = len(lower_leaflet.residues)
    
    
    if n_upper == 0 or n_lower == 0:
        raise ValueError("Leaflet lipid count is zero.")

    for ts in universe.trajectory [::stride]:  # Analyze every nth (as per stride) frame for speed

        box = universe.dimensions

        # APL = (box_x * box_y) / n_lipids_per_leaflet
        # Units of APL will remain in Angstrom², which is standard for lipid membranes.
        box_x = box[0]     # Å
        box_y = box[1]     # Å

        area = box_x * box_y # Å²

        apl_u = area / n_upper
        apl_l = area / n_lower

        time_ns = ts.time / 1000.0  # Time is always in ns (ps → ns)

        times.append(time_ns)
        apl_upper.append(apl_u)
        apl_lower.append(apl_l)

    result = {}

    result["time"] = np.array(times)
    result["apl_upper"] = np.array(apl_upper)
    result["apl_lower"] = np.array(apl_lower)
    
    apl_upper = np.array(apl_upper)
    apl_lower = np.array(apl_lower)

    result["apl_upper_mean"] = apl_upper.mean()
    result["apl_lower_mean"] = apl_lower.mean()

    result["apl_upper_std"] = apl_upper.std()
    result["apl_lower_std"] = apl_lower.std()

    return result

# ----------------------------------------------------------------------
# Per-species APL (Voronoi tessellation)
#
# Box-area / lipid-count APL cannot be decomposed by species; the
# standard per-species measure assigns each lipid its Voronoi cell area
# in the leaflet plane (as in FATSLiM / MEMBPLUGIN). Periodicity is
# handled by tessellating the phosphates together with their 8 periodic
# images and reading areas for the central copies only. The whole
# membrane is always tessellated — per-species numbers are a
# DECOMPOSITION of the full leaflet, never a restricted calculation.
# ----------------------------------------------------------------------

def _voronoi_areas_periodic(xy, box_xy):
    from scipy.spatial import Voronoi
    n = len(xy)
    shifts = [(ix, iy) for ix in (-1, 0, 1) for iy in (-1, 0, 1)]
    tiles = np.concatenate([xy + np.array([ix * box_xy[0], iy * box_xy[1]])
                            for ix, iy in shifts])
    # central copy occupies the block for shift (0, 0)
    central0 = shifts.index((0, 0)) * n
    vor = Voronoi(tiles)
    areas = np.full(n, np.nan)
    for i in range(n):
        region = vor.regions[vor.point_region[central0 + i]]
        if -1 in region or len(region) < 3:
            continue
        poly = vor.vertices[region]
        x, y = poly[:, 0], poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1))
                             - np.dot(y, np.roll(x, 1)))
    return areas


def calculate_apl_per_species(universe, upper_leaflet, lower_leaflet,
                              stride=1):
    """
    Mean Voronoi APL per lipid species (upper+lower combined), with the
    per-leaflet Voronoi mean as a consistency check against the
    box-area APL. Returns None when scipy is unavailable.
    """
    try:
        from scipy.spatial import Voronoi  # noqa: F401
    except Exception:
        return None

    species_acc = {}
    leaflet_means = {"upper": [], "lower": []}
    times = []

    leaflets = (("upper", upper_leaflet), ("lower", lower_leaflet))
    resnames = {tag: ag.resnames for tag, ag in leaflets}

    for ts in universe.trajectory[::stride]:
        times.append(ts.time / 1000.0)
        box_xy = ts.dimensions[:2]
        frame_vals = {}
        for tag, ag in leaflets:
            areas = _voronoi_areas_periodic(ag.positions[:, :2], box_xy)
            ok = np.isfinite(areas)
            if ok.any():
                leaflet_means[tag].append(float(np.nanmean(areas)))
            for rn in np.unique(resnames[tag]):
                sel = ok & (resnames[tag] == rn)
                if sel.any():
                    frame_vals.setdefault(rn, []).extend(areas[sel].tolist())
        # one entry per frame per species (both leaflets pooled)
        for rn, vals in frame_vals.items():
            species_acc.setdefault(rn, []).append(float(np.mean(vals)))

    return {
        "time": np.array(times),
        "species_mean": {r: float(np.mean(v))
                         for r, v in species_acc.items()},
        "species_std": {r: float(np.std(v))
                        for r, v in species_acc.items()},
        "species_series": {r: np.array(v) for r, v in species_acc.items()},
        "voronoi_leaflet_mean": {t: float(np.mean(v)) if v else float("nan")
                                 for t, v in leaflet_means.items()},
    }


def write_apl_per_species_csv(data, outfile):
    with open(outfile, "w") as f:
        f.write("# Voronoi per-species APL (whole-membrane tessellation)\n")
        f.write("species,mean_apl_A2,std_A2\n")
        for r in sorted(data["species_mean"]):
            f.write(f"{r},{data['species_mean'][r]:.4f},"
                    f"{data['species_std'][r]:.4f}\n")
        f.write(f"# Voronoi leaflet means (consistency check): "
                f"upper={data['voronoi_leaflet_mean']['upper']:.3f}, "
                f"lower={data['voronoi_leaflet_mean']['lower']:.3f}\n")


def plot_apl_per_species(data, outfile):
    species = sorted(data["species_series"])
    if not species:
        return
    plt.figure(figsize=(6, 4))
    try:
        from analysis.curvature import species_colors
        colors = species_colors(species)
    except Exception:
        shades = ["black", "gray", "#b0b0b0", "#707070", "#d0d0d0"]
        colors = {r: shades[i % len(shades)] for i, r in enumerate(species)}
    for i, r in enumerate(species):
        n = min(len(data["time"]), len(data["species_series"][r]))
        plt.plot(data["time"][:n], data["species_series"][r][:n],
                 lw=0.9, color=colors[r],
                 label=f"{r} ({data['species_mean'][r]:.1f} "
                       + r"$\AA^2$)")
    plt.xlim(0, data["time"].max() if len(data["time"]) else 1)
    plt.xlabel("Time (ns)", fontsize=12, fontfamily="DejaVu Sans")
    plt.ylabel(r"APL ($\AA^2$)", fontsize=12,
               fontfamily="DejaVu Sans")
    plt.title("Area Per Lipid by Species", fontsize=12, fontweight="bold",
              fontfamily="DejaVu Sans")
    plt.legend(loc="upper right", prop={"family": "DejaVu Sans", "size": 9},
               frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def write_apl_xvg(output_file, data):
    """
    Write APL timeseries to XVG format.
    """

    with open(output_file, "w") as f:

        f.write("# smGROANA APL analysis\n")
        f.write("@    title \"Area Per Lipid\"\n")
        f.write("@    xaxis label \"Time (ns)\"\n")
        f.write("@    yaxis label \"APL (Å^2)\"\n")

        f.write("@TYPE xy\n")

        for t, u, l in zip(data["time"], data["apl_upper"], data["apl_lower"]):
            f.write(f"{t:.4f} {u:.5f} {l:.5f}\n")




def plot_apl(output_png, data):

    plt.figure()

    plt.plot(data["time"], data["apl_upper"], label="Upper leaflet", color="black", alpha=0.9, linewidth=0.9)
    plt.plot(data["time"], data["apl_lower"], label="Lower leaflet", color="gray", alpha=0.9, linewidth=0.9)

    # Calculating limits to add 10 units on both sides
    y_min = min(data["apl_lower"].min(), data["apl_upper"].min()) - 5
    y_max = max(data["apl_lower"].max(), data["apl_upper"].max()) + 5
    plt.ylim(y_min, y_max)
    plt.xlim(0, data["time"].max())

    plt.xlabel("Time (ns)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.ylabel(r"APL ($\AA^2$)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.legend(loc='upper right', prop={"family": "DejaVu Sans", "size": 10}, frameon=True, facecolor='white', edgecolor='black')
    plt.title("Area Per Lipid", fontsize=12,fontweight="bold", fontfamily="DejaVu Sans")

    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.close()