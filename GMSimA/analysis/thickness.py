import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_membrane_normal(headgroup_positions):
    """
    Determine membrane normal using PCA.
    Returns the normal vector.
    """
    coords = headgroup_positions - np.mean(headgroup_positions, axis=0)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # smallest eigenvalue corresponds to membrane normal
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return normal / np.linalg.norm(normal)


def calculate_thickness_timeseries(universe, upper_leaflet, lower_leaflet,stride):

    times = []
    thickness_values = []

    # The leaflet AtomGroups from detect_leaflets() ARE the phosphorus
    # atoms (whatever the force-field naming: 'P' for CHARMM, 'P31' for
    # AMBER) — use them directly instead of re-selecting 'name P',
    # which would come back empty on AMBER topologies.
    upper_heads = upper_leaflet
    lower_heads = lower_leaflet

    for ts in universe.trajectory[::stride]:

        # Because the trajectory is pre-aligned (rot+trans fit), the membrane normal
        # is assumed to be the Z-axis. Therefore thickness can be computed directly
        # as the difference between the mean Z positions of the two leaflets.

        z_upper = upper_heads.positions[:,2].mean()
        z_lower = lower_heads.positions[:,2].mean()

        thickness = abs(z_upper - z_lower)

        time_ns = ts.time / 1000.0

        times.append(time_ns)
        thickness_values.append(thickness)

    thickness_values = np.array(thickness_values)

    result = {
        "time": np.array(times),
        "thickness": thickness_values,
        "mean": thickness_values.mean(),
        "std": thickness_values.std()
    }

    return result
# ------------- XVG output -------------
def write_thickness_xvg(output_file, data):

    with open(output_file, "w") as f:

        f.write("# smGROANA membrane thickness\n")
        f.write("@    title \"Membrane Thickness\"\n")
        f.write("@    xaxis label \"Time (ns)\"\n")
        f.write("@    yaxis label \"Thickness (Å)\"\n")

        f.write("@TYPE xy\n")

        for t, th in zip(data["time"], data["thickness"]):
            f.write(f"{t:.4f} {th:.4f}\n")

# ------------- Plotting -------------
def plot_thickness(output_png, data):

    plt.figure()

    plt.plot(data["time"], data["thickness"], color="black", alpha=0.9, linewidth=0.9 )
    # Calculating limits to add 10 units on both sides
    y_min = min(data["thickness"]) - 5
    y_max = max(data["thickness"]) + 5
    plt.ylim(y_min, y_max)
    plt.xlim(0, data["time"].max())

    plt.xlabel("Time (ns)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.ylabel(r"Thickness ($\AA$)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.title("Membrane-Thickness", fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")

    plt.tight_layout()

    plt.savefig(output_png, dpi=300)

    plt.close()