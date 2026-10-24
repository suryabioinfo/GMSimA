import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------
#    MEMBRANE CURVATURE ANALYSIS (Individual Leaflet)
# --------------------------------------------------------

def calculate_curvature_timeseries(u, upper, lower, stride=1, grid_size=30):
    """
    Two curvature radii per leaflet per frame:

    upper/lower (PRIMARY, original smGROANA metric): radius from the raw
        interpolated phosphate surface. At this resolution the surface
        tracks individual lipid protrusions, so the value (~15-20 Å for
        a fluid bilayer) is a MOLECULAR-SCALE roughness/packing metric —
        sensitive to peptide binding, comparable across systems, and
        continuous with earlier smGROANA results.

    upper_meso/lower_meso (secondary): same Laplacian after Gaussian-
        smoothing the surface with sigma of one lipid-lipid spacing,
        which removes the protrusion signal and reports the MESOSCALE
        undulation curvature (hundreds of Å for a flat patch).
    """
    times = []
    R_upper, R_lower = [], []
    Rm_upper, Rm_lower = [], []

    def compute_leaflet_curvature(positions):

        xy = positions[:, :2]
        z  = positions[:, 2]

        # grid
        xi = np.linspace(xy[:,0].min(), xy[:,0].max(), grid_size)
        yi = np.linspace(xy[:,1].min(), xy[:,1].max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata(xy, z, (xi, yi), method='cubic')
        # mask invalid regions
        mask = np.isnan(zi)
        # fill using non-zero data
        zi_nn=griddata(xy, z, (xi, yi), method = 'nearest')
        zi[mask] = zi_nn[mask]

        dx = xi[0, 1] - xi[0, 0]
        dy = yi[1, 0] - yi[0, 0]
        if dx <= 0 or dy <= 0:
            return np.inf, np.inf

        def radius_from(surface):
            # Monge small-slope mean curvature H = 0.5*(z_xx + z_yy),
            # PHYSICAL Å spacing; one-cell border trimmed (one-sided
            # stencils + nearest-fill edge artifacts).
            d2x = np.gradient(np.gradient(surface, dx, axis=1), dx, axis=1)
            d2y = np.gradient(np.gradient(surface, dy, axis=0), dy, axis=0)
            H = 0.5 * (d2x + d2y)
            mean_curv = np.mean(np.abs(H[1:-1, 1:-1]))
            if mean_curv == 0:
                return np.inf
            return 1.0 / (mean_curv + 1e-12)   # Å

        r_rough = radius_from(zi)

        # mesoscale: smooth to one lipid-lipid spacing first
        area = ((xy[:, 0].max() - xy[:, 0].min())
                * (xy[:, 1].max() - xy[:, 1].min()))
        lipid_spacing = np.sqrt(area / max(len(xy), 1))    # Å
        sigma_grid = (max(lipid_spacing / dy, 0.0),
                      max(lipid_spacing / dx, 0.0))
        r_meso = radius_from(gaussian_filter(zi, sigma=sigma_grid,
                                             mode="nearest"))
        return r_rough, r_meso

    for ts in u.trajectory[::stride]:
        ru, rmu = compute_leaflet_curvature(upper.positions)
        rl, rml = compute_leaflet_curvature(lower.positions)
        times.append(ts.time / 1000.0)
        R_upper.append(ru)
        R_lower.append(rl)
        Rm_upper.append(rmu)
        Rm_lower.append(rml)

    return {
        "time": np.array(times),
        "upper": np.array(R_upper),
        "lower": np.array(R_lower),
        "upper_meso": np.array(Rm_upper),
        "lower_meso": np.array(Rm_lower)
    }


# ---------------------------------------------------------
#               WRITE CURVATURE DATA XVG
# ---------------------------------------------------------
def write_curvature_xvg(data, outfile):

    has_meso = "upper_meso" in data

    with open(outfile, "w") as f:
        f.write("@ title \"Membrane Curvature\"\n")
        f.write("# upper/lower: molecular-scale roughness radius "
                "(original smGROANA metric)\n")
        if has_meso:
            f.write("# upper_meso/lower_meso: mesoscale undulation radius "
                    "(lipid-spacing smoothed surface)\n")
        f.write("@ xaxis label \"Time (ns)\"\n")
        f.write("@ yaxis label \"Radius (Å)\"\n")
        f.write("@TYPE xy\n")

        if has_meso:
            f.write("# time upper lower upper_meso lower_meso\n")
            for t, u, l, um, lm in zip(data["time"], data["upper"],
                                       data["lower"], data["upper_meso"],
                                       data["lower_meso"]):
                f.write(f"{t:.3f} {u:.3f} {l:.3f} {um:.3f} {lm:.3f}\n")
        else:
            for t, u, l in zip(data["time"], data["upper"], data["lower"]):
                f.write(f"{t:.3f} {u:.3f} {l:.3f} \n")



# ---------------------------------------------------------
#          CURVATURE PLOTTING
# ---------------------------------------------------------
def plot_curvature(data, outfile, keys=("upper", "lower"),
                   title="Membrane Curvature"):

    up, lo = data[keys[0]], data[keys[1]]

    plt.figure(figsize=(6,4))

    plt.plot(data["time"], up, label="Upper leaflet", color="black", alpha=0.9, linewidth=0.9)
    plt.plot(data["time"], lo, label="Lower leaflet", color="gray", alpha=0.9, linewidth=0.9)

    # Fix axis limits
    y_min = min(lo.min(), up.min()) - 1
    y_max = max(lo.max(), up.max()) + 2
    plt.ylim(y_min, y_max)
    plt.xlim(0, data["time"].max())

    plt.xlabel("Time (ns)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.ylabel(r"Radius of Curvature ($\AA$)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.title(title)
    plt.legend(loc='upper right', prop={"family": "DejaVu Sans", "size": 10}, frameon=True, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# -------------------------------------------------------
# LATERAL DIFFUSION (MSD)
# -------------------------------------------------------


def unwrap_xy_trajectory(coords, boxes):
    """
    Unwrap XY coordinates across periodic boundaries.

    Wrapped ('-pbc mol') trajectories still make a lipid jump a full box
    length when it crosses the boundary, which invalidates MSD. Here each
    frame-to-frame displacement is minimum-imaged and accumulated, which
    reconstructs the continuous path (valid as long as no atom moves more
    than half a box length between analyzed frames).

    Parameters
    ----------
    coords : (nframes, natoms, 2) wrapped XY positions (Å)
    boxes  : (nframes, 2) box XY lengths per frame (Å)
    """
    unwrapped = np.array(coords, dtype=float, copy=True)
    for t in range(1, len(coords)):
        delta = coords[t] - coords[t - 1]
        L = boxes[t]
        delta -= np.round(delta / L) * L         # minimum image
        unwrapped[t] = unwrapped[t - 1] + delta
    return unwrapped


def _msd_fft(coords):
    """
    All-lag MSD via the FFT autocorrelation identity (Kneller/nMOLDYN
    algorithm): O(N log N) per atom instead of the O(N^2) displacement
    double loop, which was the dominant cost for long trajectories
    (a 10,000-frame run drops from minutes to well under a second).

    coords : (nframes, natoms, ndim) unwrapped positions.
    Returns msd(t) averaged over atoms, length nframes.
    """
    N, natoms, ndim = coords.shape
    r = np.asarray(coords, dtype=float)

    # S2(m) = (1/(N-m)) sum_k r_k . r_{k+m}   via FFT, per atom
    F = np.fft.fft(r, n=2 * N, axis=0)
    acf = np.fft.ifft(F * F.conj(), axis=0)[:N].real   # (N, natoms, ndim)
    S2 = acf.sum(axis=2)                               # dot over dims
    S2 /= (N - np.arange(N))[:, None]

    # S1(m) = (1/(N-m)) sum_k (|r_k|^2 + |r_{k+m}|^2)
    D = np.square(r).sum(axis=2)                       # (N, natoms)
    Q = 2.0 * D.sum(axis=0)
    S1 = np.empty((N, natoms))
    for m in range(N):
        if m > 0:
            Q -= D[m - 1] + D[N - m]
        S1[m] = Q / (N - m)

    msd_per_atom = S1 - 2.0 * S2
    msd = msd_per_atom.mean(axis=1)
    msd[0] = 0.0
    return msd


def calculate_lateral_diffusion(u, sel="name P*", stride=1):
    atoms = u.select_atoms(sel)
    coords = []
    boxes = []
    for ts in u.trajectory[::stride]:
        coords.append(atoms.positions[:, :2].copy())  # XY
        boxes.append(ts.dimensions[:2].copy())
    coords = np.array(coords)
    boxes = np.array(boxes)
    nframes = len(coords)

    # --- unwrap PBC jumps, then remove collective (COM) drift ---
    coords = unwrap_xy_trajectory(coords, boxes)
    coords -= coords.mean(axis=1, keepdims=True)

    msd = _msd_fft(coords)

    # ---- time in seconds ----
    dt_ps = u.trajectory.dt * stride
    time = np.arange(nframes) * dt_ps * 1e-12  # sec
    # ------------------------------
    # ROBUST FIT REGION (LINEAR MSD)
    # ------------------------------
    # Use middle region (avoid ballistic + noisy tail)
    start = int(nframes * 0.3)
    end   = int(nframes * 0.8)
    if end <= start + 5:
        raise RuntimeError("Trajectory too short for diffusion calculation")
    slope, _ = np.polyfit(time[start:end], msd[start:end], 1)

    # A non-positive slope means the sampled window carries no diffusive
    # signal; report it honestly rather than manufacturing a value.
    if slope <= 0:
        D_cm2 = float("nan")
    else:
        D = slope / 4.0          # Å²/s (2D)
        D_cm2 = D * 1e-16

    return {
        "time": time,
        "msd": msd,
        "D": D_cm2
    }


# ---------------------------------------------------------
#           WRITE DIFFUSION
# ---------------------------------------------------------
def write_diffusion_txt(data, outfile):
    with open(outfile, "w") as f:
        f.write("smGROANA lipid lateral diffusion analysis\n")
        f.write("Study carried on P atoms\n\n")
        f.write(f"Mean lateral diffusion rate of lipids is: {data['D']:.4e} cm^2/sec\n\n")
        f.write("# Time(s) MSD(Å^2)\n")

        for t, m in zip(data["time"], data["msd"]):
            f.write(f"{t:.6e} {m:.6f}\n")


# ---------------------------------------------------------
#                   DENSITY PROFILE
# ---------------------------------------------------------

def calculate_density_profile(u, selections, stride=1, bins=100,
                              center=True):
    """
    Mass density profiles along z, in kg/m³.

    `selections` values may be MDAnalysis selection strings or
    AtomGroups. With center=True (default) every frame's z-coordinates
    are taken RELATIVE TO THE BILAYER MIDPLANE of that frame (mean
    phosphate z), minimum-imaged into [-Lz/2, +Lz/2): the axis then
    reads -.../0/+... with 0 at the bilayer center, and membrane drift
    along z no longer smears the profiles. All selections are histogram-
    med in a single trajectory pass.
    """
    groups = {}
    for key, sel in selections.items():
        groups[key] = u.select_atoms(sel) if isinstance(sel, str) else sel

    midref = None
    if center:
        try:
            from core.lipids import phosphate_selection
            midref = u.select_atoms(phosphate_selection(u))
            if midref.n_atoms == 0:
                midref = None
        except Exception:
            midref = None

    lz0 = float(u.trajectory[0].dimensions[2])
    do_center = center and midref is not None
    rng = (-lz0 / 2.0, lz0 / 2.0) if do_center else (0.0, lz0)

    hists = {k: np.zeros(bins) for k in groups}
    edges = np.linspace(rng[0], rng[1], bins + 1)
    vol_sum = 0.0     # summed per-bin slab volume over frames (Å³)
    for ts in u.trajectory[::stride]:
        lz = float(ts.dimensions[2])
        z0 = midref.positions[:, 2].mean() if do_center else 0.0
        # Per-frame BOX-RELATIVE binning: fold into this frame's own
        # box, then scale onto the fixed reference range. Without the
        # scaling, NPT fluctuations of Lz leave the outermost bins
        # covered only by the larger-box frames, so bulk water appears
        # to decay to zero at the plot edges — a binning artifact, not
        # physics.
        scale = lz0 / lz if lz > 0 else 1.0
        for key, ag in groups.items():
            if ag.n_atoms == 0:
                continue
            z = ag.positions[:, 2] - z0
            if do_center:
                z -= lz * np.round(z / lz)     # minimum image about midplane
            h, _ = np.histogram(z * scale, bins=bins, range=rng,
                                weights=ag.masses)
            hists[key] += h
        vol_sum += (float(ts.dimensions[0]) * float(ts.dimensions[1])
                    * lz / bins)

    centers = 0.5 * (edges[:-1] + edges[1:])
    results = {}
    for key in groups:
        density = hists[key] / max(vol_sum, 1e-12)           # amu/Å³
        results[key] = (centers, density * 1.66054e3)        # kg/m³
    return results


# Water resnames across common force-field ports (TIP3/TIP4/OPC/SPC
# families). Virtual sites (e.g. OPC's EPW) carry zero mass and simply
# do not contribute to the mass histogram.
WATER_RESNAMES = ("SOL WAT HOH OPC OPC3 TIP3 TIP3P TIP4 TIP4P TIP4PEW "
                  "TIP4PD TIP5P SPC SPCE T3P T4P T4PEW")

# Fixed, colorblind-safe species colors (Okabe–Ito) keyed by headgroup
# class; unmatched species fall back to the cycle below.
_SPECIES_COLOR_RULES = (
    ("PC",   "#0072B2"),   # blue
    ("PE",   "#E69F00"),   # orange
    ("PS",   "#009E73"),   # green
    ("PG",   "#56B4E9"),   # sky blue
    ("PI",   "#CC79A7"),   # purple-pink
    ("SM",   "#AA4499"),   # magenta
    ("SPM",  "#AA4499"),
    ("CHL",  "#D55E00"),   # vermillion (also matches CHL1)
    ("CHOL", "#D55E00"),
    ("ERG",  "#D55E00"),
    ("PA",   "#F0E442"),   # yellow
)
_SPECIES_COLOR_CYCLE = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                        "#CC79A7", "#56B4E9", "#F0E442", "#999999")


def species_colors(species):
    """Deterministic color per lipid species name (Okabe–Ito palette)."""
    colors, used = {}, 0
    for rn in species:
        up = rn.upper()
        for tag, col in _SPECIES_COLOR_RULES:
            if tag in up:
                colors[rn] = col
                break
        else:
            colors[rn] = _SPECIES_COLOR_CYCLE[used % len(_SPECIES_COLOR_CYCLE)]
            used += 1
    return colors


def acyl_chain_atoms(u, memb_sel=""):
    """
    Heavy atoms + hydrogens of the ACYL CHAINS only — sterols and all
    headgroup/glycerol atoms excluded (family-aware).

    AMBER Lipid14/17/21: the chain residues (PA, OL, ...) *are* the acyl
    chains, so the whole residues are used. CHARMM36: chain carbons
    follow the C2x (sn-2) / C3x (sn-1) naming; the glycerol C1/C2/C3
    are excluded by exact name and chain hydrogens are added through
    bond connectivity when bonds are available.
    """
    from core.lipids import (detect_ff_family, AMBER_CHAINS)
    try:
        fam = detect_ff_family(u)
    except Exception:
        fam = "unknown"
    if fam == "amber":
        return u.select_atoms("resname " + " ".join(sorted(AMBER_CHAINS)))
    scope = f"({memb_sel}) and " if memb_sel else ""
    carbons = u.select_atoms(
        f"{scope}name C2* C3* and not name C1 C2 C3 "
        "and not resname CHL1 CHOL CHL ERG")
    try:
        hydrogens = u.select_atoms("mass 0.5:1.5 and bonded group ch",
                                   ch=carbons)
        return carbons | hydrogens
    except Exception:
        return carbons


# --------------------------------------------------------
# PLOT DENSITY
# --------------------------------------------------------
def plot_density_profile(data, outfile, custom_colors=None):

    plt.figure(figsize=(6,5))

    for key, (z, rho) in data.items():
        color_choice = custom_colors.get(key, "aqua") if custom_colors else None
        plt.plot(z, rho, label=key, color=color_choice)

    plt.xlabel(r"Z − bilayer center ($\AA$)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.axvline(0.0, color="#888888", linewidth=0.8, linestyle=":")
    plt.ylabel(r"Density ($kg \cdot m^{-3}$)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.title("Membrane Density Profile", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.legend(loc='upper right', prop={"family": "DejaVu Sans", "size": 8}, frameon=True, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# --------------------------------------------------------
# MASTER WRAPPER
# --------------------------------------------------------
def run_membrane_module(u, upper, lower, output_prefix, stride=1):

    # ---- Curvature ----
    curv = calculate_curvature_timeseries(u, upper, lower, stride)

    write_curvature_xvg(curv, f"{output_prefix}_curvature.xvg")
    # primary plot: original molecular-scale roughness radius
    plot_curvature(curv, f"{output_prefix}_curvature.png")
    # secondary: mesoscale undulation radius (smoothed surface)
    if "upper_meso" in curv:
        plot_curvature(curv, f"{output_prefix}_curvature_mesoscale.png",
                       keys=("upper_meso", "lower_meso"),
                       title="Membrane Curvature (mesoscale undulation)")

    # ---- Diffusion ----
    diff = calculate_lateral_diffusion(u, stride=stride)
    write_diffusion_txt(diff, f"{output_prefix}_diffusion.txt")

    # ---- Density ----
    # Component selections are restricted to the membrane lipids:
    # unrestricted 'name C*' / 'name P*' would silently count protein
    # atoms into the "Chains"/"Glycerol"/"Headgroup" curves.
    try:
        from core.lipids import membrane_selection
        memb = membrane_selection(u)
    except Exception:
        memb = ""
    scope = f"({memb}) and " if memb else ""
    selections = {
        "Headgroup": f"{scope}name P*",
        "Glycerol": f"{scope}name C1 C2 O*",
        "Acyl chains": acyl_chain_atoms(u, memb),
        "Water": f"resname {WATER_RESNAMES}",
    }

    dens = calculate_density_profile(u, selections, stride)
    my_colors = {"Headgroup": "gold", "Glycerol": "red",
                 "Acyl chains": "black", "Water": "blue"}
    plot_density_profile(dens, f"{output_prefix}_density.png", custom_colors=my_colors)

    # ---- Per-species density (whole-lipid mass density per resname) ----
    dens_species = None
    try:
        from core.lipids import detect_lipids, detect_ff_family
        species = sorted(detect_lipids(u))
        if len(species) > 1:
            # AMBER species are identified by the head resname; the whole
            # 3-residue lipid is captured through the bonded fragment.
            if detect_ff_family(u) == "amber":
                sp_selections = {rn: f"same fragment as resname {rn}"
                                 for rn in species}
            else:
                sp_selections = {rn: f"resname {rn}" for rn in species}
            dens_species = calculate_density_profile(u, sp_selections, stride)
            sp_colors = species_colors(species)
            plot_density_profile(dens_species,
                                 f"{output_prefix}_density_species.png",
                                 custom_colors=sp_colors)
    except Exception as exc:
        print(f"Per-species density skipped: {exc}")

    return {
        "curvature": curv,
        "diffusion": diff,
        "density": dens,
        "density_species": dens_species
    }