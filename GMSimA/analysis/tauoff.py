"""
Residence-time (τ-off) analysis of peptide–membrane binding.

Statistics
----------
Contacts are defined between the user-composed ANCHOR atoms (any
preset or custom selection, restricted to the protein group) and the
user-composed MEMBRANE PARTNER atoms (V3 interaction matrix: whole
lipids, phosphates, headgroup+glycerol, acyl chains or sterol
hydroxyls, optionally species-filtered), PBC-aware, dual-cutoff
CONTACT_CUTOFF / CONTACT_CUTOFF_END. The historical default —
Lys/Arg nitrogens on phosphates — is one composition among equals.

Gap tolerance and the minimum contact duration are expressed in
NANOSECONDS (not frames), so results are independent of the analysis
stride.

The primary estimate of τ-off comes from the survival function S(t) of
individual contact-event lifetimes: S(t) is fit with a mono-exponential
(maximum likelihood: τ = mean lifetime) and, when enough events exist,
a bi-exponential; the model is chosen by AIC and the SLOW component is
reported as τ-off, with a bootstrap confidence interval. The legacy
cumulative binding-ratio curve is retained as a secondary output.

A per-residue residence table (mean contact lifetime and bound fraction
for each basic residue) identifies the anchoring residues.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from MDAnalysis.lib.distances import capped_distance


CONTACT_CUTOFF = 4.5       # Å — a contact STARTS below this distance
CONTACT_CUTOFF_END = 7.0   # Å — and only ENDS above this (dual-cutoff /
                           # hysteresis scheme, as in PyLipID: suppresses
                           # rattling in and out of a single cutoff)
GAP_TOLERANCE_NS = 0.2     # bridge unbound gaps shorter than this
MIN_CONTACT_NS = 0.3       # discard contacts shorter than this
N_BOOTSTRAP = 200
_BASIC_SEL = "(resname LYS and name NZ) or (resname ARG and (name NH1 NH2))"

# ----------------------------------------------------------------------
# Anchor presets: WHAT part of the peptide is considered the binding
# moiety. 'basic' reproduces the original cationic-peptide behaviour;
# the others generalize the residence analysis to hydrophobic and
# aromatic anchoring, or to the whole peptide (PyLipID-style).
# The matching membrane PARTNER is chosen by the caller: phosphate
# head atoms for electrostatic anchors, all heavy lipid atoms for
# hydrophobic/aromatic/all (chain insertion is invisible to phosphates).
# ----------------------------------------------------------------------
ANCHOR_PRESETS = {
    "basic": _BASIC_SEL,
    "hydrophobic": ("resname LEU ILE VAL ALA MET PHE TRP PRO "
                    "and not backbone and not name H*"),
    "aromatic": ("resname PHE TYR TRP HIS HSD HSE HSP and name CG CD1 CD2 "
                 "CE1 CE2 CE3 CZ CZ2 CZ3 CH2 NE1 ND1 NE2 OH"),
    "all": "not name H*",
}


# -----------------------------------
# Series post-processing (frame-count helpers driven by ns parameters)
# -----------------------------------

def fill_gaps(series, gap):
    """
    Bridge unbound runs of length <= gap frames that lie strictly
    between two bound runs. (Run-length based: the previous windowed
    version could bridge gaps up to ~2x the tolerance because each fill
    extended the window of the next.)
    """
    series = series.copy()
    n = len(series)
    i = 0
    while i < n:
        if not series[i]:
            j = i
            while j < n and not series[j]:
                j += 1
            # gap [i, j) is bounded by contacts on both sides?
            if i > 0 and j < n and (j - i) <= gap:
                series[i:j] = True
            i = j
        else:
            i += 1
    return series


def remove_short_contacts(series, min_frames):
    clean = series.copy()
    start = None
    for i, val in enumerate(series):
        if val and start is None:
            start = i
        if not val and start is not None:
            if (i - start) < min_frames:
                clean[start:i] = False
            start = None
    # trailing run shorter than min_frames is kept: it is right-censored,
    # we cannot know how long it would have lasted.
    return clean


def extract_events(times, series):
    """
    Returns (lifetimes, censored): censored is True when the final event
    runs to the end of the trajectory (right-censored observation).
    """
    events = []
    start = None
    censored = False
    for i, val in enumerate(series):
        if val and start is None:
            start = times[i]
        if not val and start is not None:
            events.append(times[i] - start)
            start = None
    if start is not None:
        events.append(times[-1] - start)
        censored = True
    return np.array(events), censored


# -----------------------------------
# Contact series (global + per-residue in ONE trajectory pass)
# -----------------------------------

def _lipid_units(lipid_heads):
    """
    Group partner atoms into lipid units and label each unit's species.

    Returns (unit_of_atom, species): unit_of_atom[i] is the unit index
    of partner atom i; species[k] the species name of unit k. Units are
    bonded fragments (so AMBER 3-residue lipids count once), falling
    back to residues when no bond information exists. The species label
    is the head/sterol resname of the unit's fragment when identifiable
    (AMBER), else the unit's own resname (CHARMM one-residue lipids).
    """
    if len(lipid_heads) == 0:
        return np.array([], dtype=int), []
    try:
        gidx = lipid_heads.fragindices
    except Exception:
        gidx = lipid_heads.resindices
    uniq, inv = np.unique(gidx, return_inverse=True)

    try:
        from core.lipids import AMBER_HEADS, AMBER_STEROLS
        head_names = set(AMBER_HEADS) | set(AMBER_STEROLS)
    except Exception:
        head_names = set()

    species = []
    for k in range(len(uniq)):
        atoms = lipid_heads[inv == k]
        label = None
        try:
            frag_resnames = set(atoms[0].fragment.residues.resnames)
            for rn in frag_resnames:
                if rn in head_names:
                    label = rn
                    break
            if label is None:
                label = sorted(set(atoms.resnames))[0]
        except Exception:
            label = sorted(set(atoms.resnames))[0]
        species.append(str(label))
    return inv.astype(int), species


def compute_contact_series(universe, stride=1, protein_sel=None,
                           heads_sel=None, anchor_sel=None):
    """
    Dual-cutoff (hysteresis) contact series at three levels, computed in
    ONE trajectory pass:

      - global   : peptide bound/unbound to the membrane
      - per_res  : one series per anchor residue (anchoring analysis)
      - per_lip  : one series per LIPID MOLECULE — each lipid is its
                   own ligand, PyLipID-style, so residence events can be
                   pooled over hundreds of lipids for tight statistics

    A contact starts when the relevant minimum distance drops below
    CONTACT_CUTOFF and only ends once it exceeds CONTACT_CUTOFF_END,
    which suppresses single-cutoff rattling artifacts.

    Returns (times, global_series, per_res, labels, per_lip,
    lipid_species) — lipid_species[j] names the species of per_lip
    column j, enabling per-species pooling.

    anchor_sel: MDAnalysis selection (or ANCHOR_PRESETS key) defining
    the peptide's binding moiety; default 'basic' (Lys NZ / Arg NH*).
    heads_sel: the membrane partner atoms (any partner kind of the V3
    interaction matrix).
    """
    anchor = ANCHOR_PRESETS.get(anchor_sel or "basic",
                                anchor_sel or _BASIC_SEL)
    if protein_sel:
        pos_res = universe.select_atoms(f"({protein_sel}) and ({anchor})")
    else:
        pos_res = universe.select_atoms(anchor)

    lipid_heads = universe.select_atoms(heads_sel or "name P or name PO4")

    residues = pos_res.residues
    n_res = len(residues)
    res_index = {r.resindex: i for i, r in enumerate(residues)}
    atom_to_res = np.array([res_index[a.resindex] for a in pos_res.atoms],
                           dtype=int) if len(pos_res) else np.array([], int)

    # V3: partner atoms are grouped into LIPID UNITS (bonded fragment,
    # residue fallback), so 'per lipid' truly means one ligand per lipid
    # molecule even when the partner has many atoms per lipid (whole
    # lipids, acyl chains). With one partner atom per lipid (phosphates)
    # this reduces exactly to the previous behaviour.
    unit_of_atom, lipid_species = _lipid_units(lipid_heads)
    n_lip = len(lipid_species)

    times = []
    series, per_res, per_lip = [], [], []
    bound_g = False
    bound_r = np.zeros(n_res, dtype=bool)
    bound_l = np.zeros(n_lip, dtype=bool)

    for ts in universe.trajectory[::stride]:
        times.append(ts.time / 1000)
        min_r = np.full(n_res, np.inf)
        min_l = np.full(n_lip, np.inf)
        min_g = np.inf
        if len(pos_res) and n_lip:
            # box= is REQUIRED for periodicity-correct contacts; the
            # loose cutoff is queried so hysteresis can be evaluated.
            pairs, dists = capped_distance(pos_res.positions,
                                           lipid_heads.positions,
                                           max_cutoff=CONTACT_CUTOFF_END,
                                           box=ts.dimensions,
                                           return_distances=True)
            if len(pairs):
                np.minimum.at(min_r, atom_to_res[pairs[:, 0]], dists)
                np.minimum.at(min_l, unit_of_atom[pairs[:, 1]], dists)
                min_g = dists.min()

        # hysteresis update: enter below START, leave above END
        bound_g = (min_g <= CONTACT_CUTOFF_END) if bound_g \
            else (min_g <= CONTACT_CUTOFF)
        bound_r = np.where(bound_r, min_r <= CONTACT_CUTOFF_END,
                           min_r <= CONTACT_CUTOFF)
        bound_l = np.where(bound_l, min_l <= CONTACT_CUTOFF_END,
                           min_l <= CONTACT_CUTOFF)

        series.append(bound_g)
        per_res.append(bound_r.copy())
        per_lip.append(bound_l.copy())

    labels = [f"{rn}{ri}" for rn, ri in zip(residues.resnames, residues.resids)]
    return (np.array(times), np.array(series),
            np.array(per_res, dtype=bool), labels,
            np.array(per_lip, dtype=bool), lipid_species)


def _ns_to_frames(times, ns):
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    return max(1, int(round(ns / max(dt, 1e-12))))


def _clean_series(times, series, gap_ns, min_contact_ns):
    gap = _ns_to_frames(times, gap_ns)
    min_frames = _ns_to_frames(times, min_contact_ns)
    return remove_short_contacts(fill_gaps(series, gap), min_frames)


# -----------------------------------
# Survival function and exponential fits
# -----------------------------------

def survival_function(events):
    t = np.sort(events)
    n = len(t)
    S = 1.0 - np.arange(1, n + 1) / n
    return t, S


def _fit_mono(events):
    tau = float(np.mean(events))
    n = len(events)
    loglik = -n * np.log(tau) - np.sum(events) / tau
    return {"model": "mono", "tau_off": tau, "params": {"tau": tau},
            "loglik": loglik, "aic": 2 * 1 - 2 * loglik}


def _fit_bi(events):
    ev = np.asarray(events, dtype=float)

    def nll(p):
        w = 1.0 / (1.0 + np.exp(-p[0]))          # logistic -> (0,1)
        t1, t2 = np.exp(p[1]), np.exp(p[2])
        pdf = w / t1 * np.exp(-ev / t1) + (1 - w) / t2 * np.exp(-ev / t2)
        return -np.sum(np.log(np.maximum(pdf, 1e-300)))

    tau0 = max(np.mean(ev), 1e-6)
    best = None
    for split in (0.3, 3.0):
        x0 = [0.0, np.log(tau0 / split), np.log(tau0 * split)]
        try:
            r = minimize(nll, x0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-8,
                                  "fatol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    if best is None or not np.isfinite(best.fun):
        return None
    w = 1.0 / (1.0 + np.exp(-best.x[0]))
    t1, t2 = np.exp(best.x[1]), np.exp(best.x[2])
    if t1 > t2:
        t1, t2, w = t2, t1, 1.0 - w
    loglik = -best.fun
    return {"model": "bi", "tau_off": float(t2),
            "params": {"w_fast": float(w), "tau_fast": float(t1),
                       "tau_slow": float(t2)},
            "loglik": loglik, "aic": 2 * 3 - 2 * loglik}


def fit_residence_time(events, n_bootstrap=N_BOOTSTRAP, seed=0):
    """
    MLE exponential fits with AIC model selection and a bootstrap CI on
    tau_off (the slow component). Needs at least 1 event.
    """
    events = np.asarray(events, dtype=float)
    events = events[events > 0]
    if len(events) == 0:
        return None

    mono = _fit_mono(events)
    fit = mono
    if len(events) >= 10:
        bi = _fit_bi(events)
        if bi is not None and bi["aic"] < mono["aic"] - 2:
            fit = bi

    # bootstrap CI on tau_off, refitting the SELECTED model
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(events, size=len(events), replace=True)
        if fit["model"] == "bi":
            b = _fit_bi(sample) or _fit_mono(sample)
        else:
            b = _fit_mono(sample)
        boots.append(b["tau_off"])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    fit = dict(fit)
    fit["tau_off_ci"] = (float(lo), float(hi))
    fit["n_events"] = int(len(events))
    return fit


# -----------------------------------
# Legacy binding-ratio curve (secondary output)
# -----------------------------------

def generate_tau_grid(max_tau):
    return np.linspace(1, max_tau, 200)


def compute_tauoff_curve(events, total_time, max_tau):
    tau = generate_tau_grid(max_tau)
    ratios = []
    for t in tau:
        rev = events[events <= t]      # inclusive: an event equal to the
        ratios.append(rev.sum() / total_time * 100)   # grid end counts
    return tau, np.array(ratios)


# -----------------------------------
# Plots / writers
# -----------------------------------

def plot_survival(events, fit, outfile, censored=False,
                  title="Contact Lifetime Survival"):
    t, S = survival_function(events)
    plt.figure(figsize=(6, 4))
    plt.step(t, S, where="post", color="black", lw=1.2, label="S(t) observed")
    if fit is not None:
        tg = np.linspace(0, max(t.max(), 1e-6), 200)
        if fit["model"] == "bi":
            p = fit["params"]
            Sf = (p["w_fast"] * np.exp(-tg / p["tau_fast"])
                  + (1 - p["w_fast"]) * np.exp(-tg / p["tau_slow"]))
            lbl = (rf"fit: $\tau_{{off}}$={fit['tau_off']:.1f} ns")
        else:
            Sf = np.exp(-tg / fit["params"]["tau"])
            lbl = rf"fit: $\tau_{{off}}$={fit['tau_off']:.1f} ns"
            
        plt.plot(tg, Sf, color="gray", ls="--", lw=1.2, label=lbl)
    plt.yscale("log")
    plt.ylim(max(S[S > 0].min() * 0.5, 1e-3) if np.any(S > 0) else 1e-3, 1.05)
    plt.xlabel("Contact lifetime (ns)", fontsize=12, fontfamily="DejaVu Sans")
    plt.ylabel("Survival probability S(t)", fontsize=12,
               fontfamily="DejaVu Sans")
    if censored:
        title += " (censored events present)"
    plt.title(title, fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
    plt.legend(loc="upper right", prop={"family": "DejaVu Sans", "size": 9},
               frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def _write_fit_block(f, name, fit, events, censored_note,
                     always_bound=False):
    f.write(f"--- {name} ---\n")
    if fit is None:
        f.write("No contact events detected.\n\n")
        return
    f.write(f"Number of contact events: {fit['n_events']}\n")
    if censored_note:
        f.write(f"NOTE: {censored_note}\n")
    if always_bound:
        # A single event spanning the trajectory carries no kinetic
        # information beyond a LOWER BOUND: do not print a fitted value
        # or a (degenerate) bootstrap interval.
        f.write("The binder never unbound within the trajectory: the "
                "residence time is a lower bound,\n"
                f"tau_off >= {float(np.max(events)):.1f} ns "
                "(no fit, no confidence interval possible from a single "
                "censored event).\n\n")
        return
    f.write(f"Selected model: {fit['model']}-exponential (by AIC)\n")
    for k, v in fit["params"].items():
        f.write(f"  {k}: {v:.4f}\n")
    lo, hi = fit["tau_off_ci"]
    f.write(f"tau_off = {fit['tau_off']:.2f} ns "
            f"(95% bootstrap CI: {lo:.2f} - {hi:.2f} ns)\n")
    f.write(f"Mean contact lifetime: {np.mean(events):.2f} ns\n\n")


def write_residence_txt(fit, events, censored, outfile,
                        gap_ns=GAP_TOLERANCE_NS, min_contact_ns=MIN_CONTACT_NS,
                        lipid_fit=None, lipid_events=None,
                        n_lipid_censored=0, n_lipids=0):
    with open(outfile, "w") as f:
        f.write("smGROANA residence-time (tau-off) analysis\n")
        f.write(f"Dual-cutoff contacts (PyLipID-style): start < "
                f"{CONTACT_CUTOFF} A, end > {CONTACT_CUTOFF_END} A | "
                f"gap tolerance: {gap_ns} ns | min contact: "
                f"{min_contact_ns} ns\n\n")
        always_bound = (fit is not None and fit.get("n_events") == 1
                        and censored)
        _write_fit_block(
            f, "Peptide-membrane residence (whole-peptide bound/unbound)",
            fit, events if events is not None else [],
            "the final event runs to the trajectory end (right-censored); "
            "tau_off may be underestimated." if censored else "",
            always_bound=always_bound)
        if lipid_fit is not None or lipid_events is not None:
            note = ""
            if n_lipid_censored:
                note = (f"{n_lipid_censored} event(s) run to the trajectory "
                        "end (right-censored).")
            _write_fit_block(
                f, f"Per-lipid residence (events pooled over {n_lipids} "
                   "lipid head groups — individual-lipid dwell time at "
                   "the peptide, PyLipID convention)",
                lipid_fit, lipid_events if lipid_events is not None else [],
                note)


def write_per_residue_csv(rows, outfile):
    pd.DataFrame(rows, columns=["residue", "n_events",
                                "mean_lifetime_ns", "bound_fraction"]
                 ).to_csv(outfile, index=False)


def plot_per_residue_residence(rows, outfile):
    """
    Per-residue residence statistics for the basic residues: mean
    contact lifetime (bars, left axis) with the bound fraction overlaid
    as markers (right axis) — together they show which residues anchor
    (long lifetimes) versus touch transiently.
    """
    if not rows:
        return
    labels = [r[0] for r in rows]
    lifetimes = [r[2] for r in rows]
    bound = [r[3] for r in rows]
    n = len(rows)

    fig, ax1 = plt.subplots(figsize=(max(6, 0.5 * n), 4))
    x = np.arange(n)
    ax1.bar(x, lifetimes, color="gray", edgecolor="black", linewidth=0.5,
            label="Mean lifetime")
    ax1.set_ylabel("Mean contact lifetime (ns)", fontsize=12,
                   fontfamily="DejaVu Sans")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9,
                        fontfamily="DejaVu Sans")

    ax2 = ax1.twinx()
    ax2.plot(x, bound, "o", color="black", ms=5, label="Bound fraction")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Bound fraction", fontsize=12, fontfamily="DejaVu Sans")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               prop={"family": "DejaVu Sans", "size": 9}, frameon=True,
               facecolor="white", edgecolor="black")

    plt.title("Per-residue Binding Residence", fontsize=12,
              fontweight="bold", fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_tauoff(tau, ratios, tauoff, outfile, label="system", note=None):

    plt.figure(figsize=(6, 4))

    plt.plot(tau, ratios, lw=1.25, label=label, color="black")

    plt.xlim(0, max(tau))
    plt.ylim(0, 110)
    plt.yticks(np.arange(0, 110, 25))

    if note:
        # always-bound (or otherwise degenerate) regime: one clear
        # message inside the axes instead of a cramped corner annotation
        plt.text(0.5, 0.55, note, fontsize=10, fontfamily="DejaVu Sans",
                 ha="center", va="center", transform=plt.gca().transAxes,
                 bbox=dict(facecolor="#f2f2f2", edgecolor="gray",
                           boxstyle="round,pad=0.6"))
    elif np.isfinite(tauoff):
        plt.axvline(tauoff, ls="--", color="gray", lw=0.9)
        binding_ratio = ratios[np.argmin(np.abs(tau - tauoff))]
        plt.annotate(fr"$\tau_{{off}}$ = {tauoff:.1f} ns"
                     f"  ({binding_ratio:.1f}%)",
                     xy=(tauoff, binding_ratio),
                     xytext=(0.03, 0.95), textcoords="axes fraction",
                     fontsize=9, fontfamily="DejaVu Sans",
                     ha="left", va="top",
                     bbox=dict(facecolor="white", alpha=0.8,
                               edgecolor="gray", pad=2.5))

    plt.xlabel("Time (ns)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.ylabel("Binding Ratio (%)", fontsize=12, fontweight="normal", color="black", fontfamily="DejaVu Sans")
    plt.title("Residence Time Analysis", fontsize=12, fontweight="bold", color="black", fontfamily="DejaVu Sans")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def write_tauoff_csv(tau, ratios, tauoff, outfile):

    df = pd.DataFrame({
        "tau_ns": tau,
        "binding_ratio_percent": ratios
    })

    df["tauoff_detected"] = tauoff

    df.to_csv(outfile, index=False)


# -----------------------------------
# Master
# -----------------------------------

def run_tauoff_analysis(universe, stride=1, protein_sel=None, heads_sel=None,
                        anchor_sel=None,
                        gap_ns=GAP_TOLERANCE_NS,
                        min_contact_ns=MIN_CONTACT_NS,
                        output_prefix=None):
    """
    Full residence-time analysis. When output_prefix is given, writes the
    survival plot/txt and per-residue CSV; the returned dict always keeps
    the legacy keys (tau, ratios, tauoff) so existing callers still work
    — 'tauoff' is now the survival-fit estimate when available.
    """
    times, series, per_res, labels, per_lip, lipid_species = \
        compute_contact_series(
            universe, stride=stride, protein_sel=protein_sel,
            heads_sel=heads_sel, anchor_sel=anchor_sel)

    if len(times) < 2 or not len(series):
        return {"status": "skipped",
                "reason": "trajectory too short or no basic residues",
                "tau": np.array([1.0]), "ratios": np.array([0.0]),
                "tauoff": float("nan")}

    clean = _clean_series(times, series, gap_ns, min_contact_ns)
    events, censored = extract_events(times, clean)
    total_time = times[-1] - times[0]

    # ---- survival fit (primary peptide-level tau_off) ----
    fit = fit_residence_time(events) if len(events) else None

    # ---- per-lipid residence: pool events over every lipid ligand ----
    # (each lipid head is its own ligand; hundreds of ligands give far
    # more events than the single whole-peptide series -> tight CIs)
    lipid_events_list = []
    n_lipid_censored = 0
    # V3: the same events, additionally bucketed by lipid SPECIES, so
    # the tool can say whether the peptide holds one species longer
    # than another (per-species tau-off with the same fit machinery).
    sp_events, sp_censored, sp_nlip = {}, {}, {}
    for j in range(per_lip.shape[1]):
        col = per_lip[:, j]
        if not col.any():
            continue
        lseries = _clean_series(times, col, gap_ns, min_contact_ns)
        levs, lcens = extract_events(times, lseries)
        lipid_events_list.append(levs)
        if lcens:
            n_lipid_censored += 1
        sp = (lipid_species[j] if j < len(lipid_species) else "?")
        sp_events.setdefault(sp, []).append(levs)
        sp_censored[sp] = sp_censored.get(sp, 0) + (1 if lcens else 0)
        sp_nlip[sp] = sp_nlip.get(sp, 0) + 1
    lipid_events = (np.concatenate(lipid_events_list)
                    if lipid_events_list else np.array([]))
    lipid_fit = (fit_residence_time(lipid_events)
                 if len(lipid_events) else None)

    species_fits = {}
    for sp in sorted(sp_events):
        ev = (np.concatenate(sp_events[sp]) if sp_events[sp]
              else np.array([]))
        species_fits[sp] = {
            "n_lipids": sp_nlip.get(sp, 0),
            "n_events": int(len(ev)),
            "n_censored": sp_censored.get(sp, 0),
            # a fit needs enough events to mean anything; below the
            # floor only the counts are reported
            "fit": fit_residence_time(ev) if len(ev) >= 5 else None,
        }

    # ---- legacy binding-ratio curve (secondary) ----
    # Grid runs to the FULL trajectory length: tau_off comes from the
    # survival fit, so the curve is descriptive and truncating it would
    # hide long (censored) events.
    traj_len = times[-1]
    max_tau = max(int(traj_len), 2)
    tau, ratios = compute_tauoff_curve(events, total_time, max_tau)
    tauoff = fit["tau_off"] if fit else float("nan")

    # ---- per-residue residence table ----
    per_residue_rows = []
    for j, lab in enumerate(labels):
        rseries = _clean_series(times, per_res[:, j], gap_ns, min_contact_ns)
        revents, _ = extract_events(times, rseries)
        per_residue_rows.append([
            lab, len(revents),
            float(np.mean(revents)) if len(revents) else 0.0,
            float(np.mean(rseries)),
        ])

    if output_prefix:
        if len(events):
            plot_survival(events, fit, f"{output_prefix}_survival.png",
                          censored=censored,
                          title="Peptide Residence Survival")
        if len(lipid_events):
            plot_survival(lipid_events, lipid_fit,
                          f"{output_prefix}_lipid_survival.png",
                          censored=n_lipid_censored > 0,
                          title="Per-lipid Residence Survival (pooled)")
        write_residence_txt(fit, events, censored,
                            f"{output_prefix}_residence.txt",
                            gap_ns=gap_ns, min_contact_ns=min_contact_ns,
                            lipid_fit=lipid_fit, lipid_events=lipid_events,
                            n_lipid_censored=n_lipid_censored,
                            n_lipids=per_lip.shape[1])
        write_per_residue_csv(per_residue_rows,
                              f"{output_prefix}_per_residue.csv")
        plot_per_residue_residence(per_residue_rows,
                                   f"{output_prefix}_per_residue.png")
        if len(species_fits) > 1:
            write_species_csv(species_fits, f"{output_prefix}_species.csv")
            plot_species_residence(species_fits,
                                   f"{output_prefix}_species.png")

    return {
        "tau": tau,
        "ratios": ratios,
        "tauoff": tauoff,
        "fit": fit,
        "events": events,
        "censored": censored,
        # single censored event spanning the run: residence is a LOWER
        # BOUND, not a fitted value (callers should report it as such)
        "always_bound": bool(fit is not None and fit.get("n_events") == 1
                             and censored),
        "lipid_fit": lipid_fit,
        "lipid_events": lipid_events,
        "n_lipid_censored": n_lipid_censored,
        "species_fits": species_fits,
        "per_residue": per_residue_rows,
        "bound_fraction": float(np.mean(clean)),
    }


# -----------------------------------
# Per-species residence outputs (V3)
# -----------------------------------

def write_species_csv(species_fits, outfile):
    with open(outfile, "w") as f:
        f.write("species,n_lipids_contacted,n_events,n_censored,"
                "tau_off_ns,ci_low_ns,ci_high_ns,model\n")
        for sp, d in sorted(species_fits.items()):
            fit = d["fit"]
            if fit:
                ci = fit.get("tau_off_ci", (float("nan"), float("nan")))
                f.write(f"{sp},{d['n_lipids']},{d['n_events']},"
                        f"{d['n_censored']},{fit['tau_off']:.4f},"
                        f"{ci[0]:.4f},{ci[1]:.4f},{fit['model']}\n")
            else:
                f.write(f"{sp},{d['n_lipids']},{d['n_events']},"
                        f"{d['n_censored']},,,,too_few_events\n")


def plot_species_residence(species_fits, outfile):
    """Per-species pooled tau-off, one bar per species with the 95%
    bootstrap interval; species with too few events are annotated."""
    fitted = [(sp, d) for sp, d in sorted(species_fits.items()) if d["fit"]]
    if not fitted:
        return
    try:
        from analysis.curvature import species_colors
        colors = species_colors([sp for sp, _ in fitted])
    except Exception:
        colors = {sp: "gray" for sp, _ in fitted}

    plt.figure(figsize=(6, 4))
    xs = np.arange(len(fitted))
    ymax = max((f[1]["fit"].get("tau_off_ci",
                                (0, f[1]["fit"]["tau_off"]))[1]
                if np.isfinite(f[1]["fit"].get(
                    "tau_off_ci", (np.nan, np.nan))[1])
                else f[1]["fit"]["tau_off"])
               for f in fitted)
    for x, (sp, d) in zip(xs, fitted):
        fit = d["fit"]
        tau = fit["tau_off"]
        ci = fit.get("tau_off_ci", (np.nan, np.nan))
        has_ci = np.isfinite(ci[0]) and np.isfinite(ci[1])
        err = ([[max(tau - ci[0], 0.0)], [max(ci[1] - tau, 0.0)]]
               if has_ci else None)
        plt.bar(x, tau, width=0.6, color=colors[sp],
                edgecolor="black", linewidth=0.6)
        if err is not None:
            plt.errorbar(x, tau, yerr=err, fmt="none", ecolor="black",
                         capsize=4, lw=1.1)
        # numbers ON the figure so the whiskers are self-explanatory:
        # tau + the 95% CI bounds above the upper cap, n at the base
        if has_ci:
            plt.text(x, ci[1] + 0.03 * ymax,
                     f"{tau:.1f} ns\n[{ci[0]:.1f}–{ci[1]:.1f}]",
                     ha="center", va="bottom", fontsize=8,
                     fontfamily="DejaVu Sans")
        else:
            plt.text(x, tau + 0.03 * ymax, f"{tau:.1f} ns",
                     ha="center", va="bottom", fontsize=8,
                     fontfamily="DejaVu Sans")
        plt.text(x, 0.02 * ymax,
                 f"n={d['n_events']}", ha="center", va="bottom",
                 fontsize=8, fontfamily="DejaVu Sans")
    plt.ylim(0, ymax * 1.22)
    plt.xticks(xs, [sp for sp, _ in fitted], fontfamily="DejaVu Sans")
    plt.ylabel(r"$\tau_{off}$ (ns)", fontsize=12,
               fontfamily="DejaVu Sans")
    plt.title("Per-species Residence Time (pooled per-lipid events)",
              fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
