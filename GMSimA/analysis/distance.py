import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array


# --------------------------------------------------------
# COMPUTE MINIMUM DISTANCE TIMESERIES
# --------------------------------------------------------
def calculate_distance_timeseries(u, selA, selB, stride):

    A = u.select_atoms(selA)
    B = u.select_atoms(selB)

    if len(A) == 0 or len(B) == 0:
        raise ValueError("Selection returned zero atoms.")

    times = []
    distances = []

    for ts in u.trajectory[::stride]:
        d = distance_array(A.positions, B.positions, box=ts.dimensions)
        distances.append(np.min(d))      # Distance Unit (Å)
        times.append(ts.time /1000 )     # Time unit (ns)

    return {
        "time": np.array(times),
        "distance": np.array(distances)
    }


# --------------------------------------------------------
# WRITE XVG
# --------------------------------------------------------
def write_distance_xvg(data, outfile):

    with open(outfile, "w") as f:
        f.write("# smGROANA Minimum Distance Analysis\n")
        f.write("@    title \"Minimum distance\"\n")
        f.write("@    xaxis  label \"Time (ns)\"\n")
        f.write("@    yaxis  label \"Distance (Å)\"\n")
        f.write("@TYPE xy\n")

        for t, d in zip(data["time"], data["distance"]):
            f.write(f"{t:.3f} {d:.4f}\n")


# --------------------------------------------------------
# PLOT DISTRIBUTION
# --------------------------------------------------------
def plot_distance_distribution(data, outfile, R_cut=3.8, nbins=60):

    d = data["distance"]

    if len(d) == 0:
        return None

    # Histogram (NO artificial cutoff)
    counts, edges = np.histogram(d, bins=nbins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # % below 3.8 Å (IMPORTANT)
    perc = np.mean(d <= R_cut) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(centers, counts, lw=1.2, color="black")
    # cutoff line at 3.8 Å
    ax.axvline(R_cut, color="blue", linestyle="--", lw=1)
    # clean ticks (same style as your reference)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30])
    y_max = counts.max()
    ax.set_ylim(0, y_max * 1.15)
    # percentage box
    ax.text( R_cut - 0.2, y_max * 1.05, f"{perc:.1f}%",
        ha="right", va="center", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="lightgrey", ec="none"))

    ax.set_xlabel(r'$\xi$ ($\AA$)', fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    ax.set_ylabel("Count", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    ax.set_title("Minimum-Distance Distribution", fontsize=12, fontweight="bold", color="black", fontfamily="DejaVu Sans")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

    return perc


# ---------------------------------------------------------
# MASTER WRAPPER (LIKE OTHER MODULES)
# ---------------------------------------------------------
def run_distance_analysis(u, output_prefix, stride=1,
                         selA="protein and not name H* and not backbone",
                         selB="resname POPS and name P*"):

    data = calculate_distance_timeseries(u, selA, selB, stride)

    xvg_file = f"{output_prefix}.xvg"
    png_file = f"{output_prefix}.png"

    write_distance_xvg(data, xvg_file)
    perc = plot_distance_distribution(data, png_file)

    return {
        "data": data,
        "contact_percent": perc
    }