"""
GMSimA V3.0 — analysis JOB functions for multi-core execution.

Each job is a TOP-LEVEL, picklable function taking one plain-dict
context (ctx) and returning a plain-dict result:

    {"summary": {...}, "captions": {...}, "logs": [...],
     "error": None | "message", "extra": {...}}

The functions contain exactly the module glue that used to live inline
in AnalysisWorker.run(): the same analysis calls, the same file names,
the same summary keys and figure captions. The worker now executes the
SAME job functions either sequentially in-process (single-core mode,
where it passes its already-open Universe) or in a multiprocessing
pool (each process opens its own Universe from the tpr/xtc paths).

Multiprocessing notes: jobs never touch Qt; matplotlib is forced to
Agg and the central plot style applied in every worker process; the
'spawn' start context is used for PyInstaller/Windows safety.
"""

import os


# ----------------------------------------------------------------------
# Per-process setup + helpers
# ----------------------------------------------------------------------

def _setup():
    import matplotlib
    matplotlib.use("Agg", force=True)
    try:
        from utils.plotstyle import apply_default_style
        apply_default_style()
    except Exception:
        pass


def _universe(ctx, u=None):
    if u is not None:
        return u
    _setup()
    import warnings
    from core.universe_loader import load_universe
    with warnings.catch_warnings():          # parent already logged it
        warnings.simplefilter("ignore", RuntimeWarning)
        return load_universe(ctx["tpr"], ctx["xtc"])


def _res():
    return {"summary": {}, "captions": {}, "logs": [], "error": None,
            "extra": {}}


def _op(ctx, base):
    root, ext = os.path.splitext(base)
    return os.path.join(ctx["output_dir"], f"{root}{ctx.get('sfx', '')}{ext}")


def _skey(ctx, name):
    lab = ctx.get("pep_label", "")
    return f"{lab}: {name}" if (ctx.get("multi") and lab) else name


def _tag(ctx):
    lab = ctx.get("pep_label", "")
    return f" [{lab}]" if (ctx.get("multi") and lab) else ""


def _leaflets(u):
    from core.leaflet_detection import detect_leaflets
    r = detect_leaflets(u)
    return r["upper_leaflet"], r["lower_leaflet"]


# ----------------------------------------------------------------------
# Intrinsic membrane jobs
# ----------------------------------------------------------------------

def job_apl(ctx, u=None):
    out = _res()
    try:
        u = _universe(ctx, u)
        import numpy as _np  # noqa: F401
        from analysis.apl import (calculate_apl_timeseries, write_apl_xvg,
                                  plot_apl, calculate_apl_per_species,
                                  write_apl_per_species_csv,
                                  plot_apl_per_species)
        upper, lower = _leaflets(u)
        apl = calculate_apl_timeseries(u, upper, lower, ctx["stride"])
        write_apl_xvg(os.path.join(ctx["output_dir"], "apl.xvg"), apl)
        plot_apl(os.path.join(ctx["output_dir"], "apl.png"), apl)
        out["summary"]["APL upper (Å²)"] = (
            f"{apl['apl_upper_mean']:.2f} ± {apl['apl_upper_std']:.2f}")
        out["summary"]["APL lower (Å²)"] = (
            f"{apl['apl_lower_mean']:.2f} ± {apl['apl_lower_std']:.2f}")
        out["captions"]["apl.png"] = (
            f"Area per lipid: upper {apl['apl_upper_mean']:.2f} ± "
            f"{apl['apl_upper_std']:.2f} Å² "
            f"[{apl['apl_upper'].min():.1f}–{apl['apl_upper'].max():.1f}]"
            f", lower {apl['apl_lower_mean']:.2f} ± "
            f"{apl['apl_lower_std']:.2f} Å² "
            f"[{apl['apl_lower'].min():.1f}–{apl['apl_lower'].max():.1f}].")
        try:
            sp = calculate_apl_per_species(u, upper, lower, ctx["stride"])
            if sp and len(sp["species_mean"]) > 1:
                write_apl_per_species_csv(
                    sp, os.path.join(ctx["output_dir"],
                                     "apl_per_species.csv"))
                plot_apl_per_species(
                    sp, os.path.join(ctx["output_dir"],
                                     "apl_per_species.png"))
                for rn, m in sorted(sp["species_mean"].items()):
                    out["summary"][f"APL {rn} (Å², Voronoi)"] = (
                        f"{m:.2f} ± {sp['species_std'][rn]:.2f}")
                out["captions"]["apl_per_species.png"] = (
                    "Voronoi APL by species (whole-membrane "
                    "tessellation): " + ", ".join(
                        f"{rn} {m:.2f} ± {sp['species_std'][rn]:.2f} Å²"
                        for rn, m in sorted(sp["species_mean"].items()))
                    + ".")
                out["logs"].append("Per-species Voronoi APL completed")
        except Exception as e:
            out["logs"].append(f"WARNING: per-species APL skipped: {e}")
        out["logs"].append("APL analysis completed")
    except Exception as e:
        out["error"] = f"APL analysis failed: {e}"
    return out


def job_thickness(ctx, u=None):
    out = _res()
    try:
        u = _universe(ctx, u)
        import numpy as _np
        from analysis.thickness import (calculate_thickness_timeseries,
                                        write_thickness_xvg, plot_thickness)
        upper, lower = _leaflets(u)
        thickness = calculate_thickness_timeseries(u, upper, lower,
                                                   ctx["stride"])
        mean_thickness = thickness["mean"]
        out["extra"]["mean_thickness"] = float(mean_thickness)
        out["summary"]["Membrane thickness (Å)"] = f"{mean_thickness:.2f}"
        _tv = _np.asarray(thickness["thickness"])
        out["captions"]["thickness.png"] = (
            f"P–P membrane thickness: mean {mean_thickness:.2f} ± "
            f"{_tv.std():.2f} Å, range [{_tv.min():.1f}–{_tv.max():.1f}] Å.")
        write_thickness_xvg(os.path.join(ctx["output_dir"],
                                         "thickness.xvg"), thickness)
        plot_thickness(os.path.join(ctx["output_dir"], "thickness.png"),
                       thickness)
        out["logs"].append("Membrane thickness analysis completed")
    except Exception as e:
        out["error"] = f"Thickness analysis failed: {e}"
    return out


def job_scd(ctx, u=None):
    out = _res()
    try:
        u = _universe(ctx, u)
        import numpy as _np
        from analysis.scd import (calculate_scd, write_scd_xvg, plot_scd,
                                  write_scd_per_species_csv,
                                  plot_scd_per_species)
        scd = calculate_scd(u, ctx["stride"])
        write_scd_xvg(scd, os.path.join(ctx["output_dir"], "scd.xvg"))
        plot_scd(scd, os.path.join(ctx["output_dir"], "scd.png"))
        method = ("C–H bond vectors (NMR convention)"
                  if scd.get("method") == "C-H"
                  else "chord approximation (~2× |S_CD|)")
        out["summary"]["S_CD method"] = method
        _s1 = (-_np.asarray(scd["sn1_scd"]) if len(scd["sn1_scd"])
               else _np.array([0.0]))
        _s2 = (-_np.asarray(scd["sn2_scd"]) if len(scd["sn2_scd"])
               else _np.array([0.0]))
        out["captions"]["scd.png"] = (
            f"Order parameter ({method}): "
            f"plateau −S_CD ≈ {max(_s1.max(), _s2.max()):.3f}; "
            f"sn-1 mean {_s1.mean():.3f}, sn-2 mean {_s2.mean():.3f}.")
        out["captions"]["scd_per_species.png"] = (
            "Order parameter decomposed by lipid species "
            "(pair-count-weighted combination equals the global curve).")
        if scd.get("per_species"):
            write_scd_per_species_csv(
                scd, os.path.join(ctx["output_dir"], "scd_per_species.csv"))
            plot_scd_per_species(
                scd, os.path.join(ctx["output_dir"], "scd_per_species.png"))
            out["logs"].append("Per-species SCD written")
        out["logs"].append("SCD analysis completed")
    except Exception as e:
        out["error"] = f"SCD analysis failed: {e}"
    return out


def job_membrane(ctx, u=None):
    out = _res()
    try:
        u = _universe(ctx, u)
        import numpy as _np
        from analysis.curvature import run_membrane_module
        upper, lower = _leaflets(u)
        prefix = os.path.join(ctx["output_dir"], "membrane")
        result = run_membrane_module(u, upper, lower, output_prefix=prefix,
                                     stride=ctx["stride"])
        if result is None or "curvature" not in result:
            raise RuntimeError("Curvature module returned no data")
        curv = result["curvature"]
        for key in ("time", "upper", "lower"):
            if key not in curv:
                raise RuntimeError(f"Missing key in curvature output: {key}")
        if len(curv["time"]) == 0:
            raise RuntimeError("Empty curvature timeseries")
        out["logs"].append(
            f"Curvature computed: {len(curv['time'])} frames "
            "(Upper/Lower leaflets)")
        out["summary"]["Curvature radius upper/lower (Å)"] = (
            f"{_np.nanmean(curv['upper']):.1f} / "
            f"{_np.nanmean(curv['lower']):.1f}")
        if "upper_meso" in curv:
            out["summary"]["Mesoscale undulation radius (Å)"] = (
                f"{_np.nanmean(curv['upper_meso']):.0f} / "
                f"{_np.nanmean(curv['lower_meso']):.0f}")
        D = result.get("diffusion", {}).get("D")
        if D is not None:
            out["summary"]["Lateral diffusion D (cm²/s)"] = f"{D:.3e}"
        out["captions"]["membrane_curvature.png"] = (
            "Molecular-scale roughness radius: upper "
            f"{_np.nanmean(curv['upper']):.1f} ± "
            f"{_np.nanstd(curv['upper']):.1f} Å, lower "
            f"{_np.nanmean(curv['lower']):.1f} ± "
            f"{_np.nanstd(curv['lower']):.1f} Å (packing/roughness "
            "metric, not a literal bending radius).")
        if "upper_meso" in curv:
            out["captions"]["membrane_curvature_mesoscale.png"] = (
                "Mesoscale undulation radius (lipid-spacing-smoothed "
                f"surface): upper {_np.nanmean(curv['upper_meso']):.0f} Å, "
                f"lower {_np.nanmean(curv['lower_meso']):.0f} Å.")
        if D is not None:
            out["captions"]["membrane_diffusion.txt"] = ""
            out["captions"]["membrane_density.png"] = (
                "Component mass density along the membrane normal; "
                "z = 0 at the bilayer midplane (per-frame centered). "
                "Acyl-chain curve excludes sterols; selections "
                "restricted to membrane lipids.")
            out["captions"]["membrane_density_species.png"] = (
                "Whole-lipid mass density per species; z = 0 at the "
                "bilayer midplane (per-frame centered).")
        out["logs"].append("Membrane module completed")
    except Exception as e:
        out["error"] = f"Membrane module failed: {e}"
    return out


# ----------------------------------------------------------------------
# Interaction jobs (per peptide)
# ----------------------------------------------------------------------

def job_distance(ctx, u=None):
    out = _res()
    tag = _tag(ctx)
    try:
        u = _universe(ctx, u)
        from analysis.distance import run_distance_analysis
        target = ctx["distance_target"]
        if len(u.select_atoms(target)) == 0:
            out["logs"].append(
                f"WARNING: Distance analysis skipped: target selection "
                f"'{target}' matches no atoms in this system.")
            return out
        selA = ctx["distance_selA"]
        dist_result = run_distance_analysis(
            u, output_prefix=_op(ctx, "distance"), stride=ctx["stride"],
            selA=selA, selB=target)
        _d = dist_result["data"]["distance"]
        _pc = dist_result.get("contact_percent")
        out["captions"][os.path.basename(_op(ctx, "distance.png"))] = (
            f"Minimum peptide–target distance ξ: mean "
            f"{_d.mean():.2f} ± {_d.std():.2f} Å, min {_d.min():.2f} Å"
            + (f"; {_pc:.1f}% of frames below 3.8 Å."
               if _pc is not None else "."))
        out["logs"].append(f"Distance analysis completed{tag}")
    except Exception as e:
        out["error"] = f"Distance analysis failed{tag}: {e}"
    return out


def job_deltaz_tilt(ctx, u=None):
    """
    ΔZ and (optionally) tilt/FEL/states as ONE job: the FEL needs the
    ΔZ series, so they are coupled and always run in the same process.
    """
    out = _res()
    tag = _tag(ctx)
    try:
        u = _universe(ctx, u)
        import numpy as _np
        deltaz = None
        if ctx["do_deltaz"]:
            from analysis.deltaz import (calculate_deltaz, write_deltaz_xvg,
                                         plot_deltaz,
                                         write_deltaz_per_residue_csv,
                                         plot_deltaz_per_residue)
            deltaz = calculate_deltaz(u, ctx["stride"],
                                      protein_sel=ctx["pep_sel"],
                                      phosphate_sel=ctx["phosphate_sel"])
            mean_thickness = ctx.get("mean_thickness")
            write_deltaz_xvg(deltaz, _op(ctx, "deltaz.xvg"))
            plot_deltaz(deltaz, _op(ctx, "deltaz.png"), mean_thickness)
            write_deltaz_per_residue_csv(
                deltaz, _op(ctx, "deltaz_per_residue.csv"))
            plot_deltaz_per_residue(deltaz,
                                    _op(ctx, "deltaz_per_residue.png"),
                                    thickness=mean_thickness)
            out["summary"][_skey(ctx, "Mean ΔZ vs midplane (Å)")] = (
                f"{_np.mean(deltaz['deltaz']):.2f}")
            out["summary"][_skey(ctx, "Mean local insertion depth (Å)")] = (
                f"{_np.mean(deltaz['deltaz_local']):.2f} "
                "(+ = past phosphate plane)")
            _dz = deltaz["deltaz"]
            out["captions"][os.path.basename(_op(ctx, "deltaz.png"))] = (
                f"ΔZ vs bilayer midplane: mean {_np.mean(_dz):.2f} ± "
                f"{_np.std(_dz):.2f} Å, range [{_np.min(_dz):.1f}–"
                f"{_np.max(_dz):.1f}] Å; mean depth vs local phosphate "
                f"plane {_np.mean(deltaz['deltaz_local']):.2f} Å "
                "(+ = inserted).")
            _pr = deltaz.get("per_residue")
            if _pr is not None and len(_pr["mean"]):
                _lo = int(_np.argmin(_pr["mean"]))
                _hi = int(_np.argmax(_pr["mean"]))
                out["captions"][os.path.basename(
                    _op(ctx, "deltaz_per_residue.png"))] = (
                    "Per-residue membrane position: deepest "
                    f"{_pr['resnames'][_lo]}{_pr['resids'][_lo]} "
                    f"({_pr['mean'][_lo]:.1f} Å), highest "
                    f"{_pr['resnames'][_hi]}{_pr['resids'][_hi]} "
                    f"({_pr['mean'][_hi]:.1f} Å).")
            out["logs"].append(f"Delta-Z analysis completed{tag}")

        if ctx["do_tilt"]:
            from analysis.tilt import (compute_tilt_timeseries,
                                       write_tilt_xvg, plot_tilt_timeseries,
                                       plot_tilt_distribution,
                                       plot_tilt_deltaz_fel,
                                       analyze_states)
            biomolecule = u.select_atoms(ctx["pep_sel"])
            if len(biomolecule) == 0 or len(biomolecule.residues) < 2:
                out["logs"].append(
                    f"WARNING: Tilt analysis skipped{tag}: peptide group "
                    "has fewer than 2 residues.")
                return out
            # v3.3: pass the membrane group so transmembrane segments can
            # be identified (and so normal="global"/"local" has
            # phosphates to fit); axis_mode/tm_segments come from the GUI.
            _memb = None
            try:
                _memb = u.select_atoms(ctx["membrane_sel"])
                if len(_memb) == 0:
                    _memb = None
            except Exception:
                _memb = None
            tilt_result = compute_tilt_timeseries(
                u, biomolecule, ctx["stride"], memb_sel=_memb,
                axis_mode=ctx.get("axis_mode", "auto"),
                segments=ctx.get("tm_segments"))
            for _w in tilt_result.get("warnings", []):
                out["logs"].append(f"Tilt{tag}: {_w}")
            _tv = tilt_result["tilt"]
            out["captions"][os.path.basename(
                _op(ctx, "tilt_timeseries.png"))] = (
                f"Peptide tilt vs membrane normal: mean "
                f"{_np.nanmean(_tv):.1f} ± {_np.nanstd(_tv):.1f}°, range "
                f"[{_np.nanmin(_tv):.0f}–{_np.nanmax(_tv):.0f}]°.")
            out["captions"][os.path.basename(
                _op(ctx, "tilt_distribution.png"))] = (
                f"Tilt-angle distribution (mean {_np.nanmean(_tv):.1f}°, "
                f"σ {_np.nanstd(_tv):.1f}°).")
            write_tilt_xvg(tilt_result, _op(ctx, "tilt_timeseries.xvg"))
            plot_tilt_timeseries(tilt_result, _op(ctx, "tilt_timeseries.png"))
            plot_tilt_distribution(tilt_result,
                                   _op(ctx, "tilt_distribution.png"))

            # v3.3: per-segment tilt series + GALA tilt-rotation map.
            from analysis import orientation as _ORI
            out["summary"][_skey(ctx, "Tilt axis definition")] = (
                f"{tilt_result['topology']} ({tilt_result['mode']})")
            if tilt_result.get("segments"):
                _ORI.write_orientation_csv(
                    tilt_result, _op(ctx, "tilt_orientation.csv"))
                _ORI.write_orientation_summary(
                    tilt_result, _op(ctx, "tilt_orientation.txt"))
                _wheel = _ORI.plot_tilt_rotation(
                    tilt_result, _op(ctx, "tilt_wheel.png"))
                if _wheel:
                    out["captions"][os.path.basename(
                        _op(ctx, "tilt_wheel.png"))] = (
                        "Tilt–rotation map (GALA convention, Strandberg "
                        "et al., Biophys. J. 86:3709, 2004): radius = "
                        "tilt θ, polar angle = rotation ρ about the helix "
                        "axis, with ρ = 0 pointing at the bilayer (the "
                        "down-tilt side), colour = time. A tight cluster "
                        "means a well-defined orientation; a full ring "
                        "means the helix rotates freely about its own "
                        "axis. One panel per segment. Because the radius "
                        "carries θ and the colour carries time, this "
                        "panel also shows the tilt's time evolution. The "
                        "tilt is written θ, not the paper's τ, because "
                        "τ denotes the binding residence time here.")
            if deltaz is not None:
                states = plot_tilt_deltaz_fel(
                    tilt_result, deltaz, _op(ctx, "tilt_deltaZ_FEL.png"),
                    return_states=True)
                if states:
                    # export_state_structures(
                    #     u, tilt_result["tilt"], deltaz["deltaz"], states,
                    #     biomolecule, output_prefix=_op(ctx, "struct"),
                    #     stride=ctx["stride"])
                    # V3.0: KDE-state structures are NOT auto-exported
                    # (projection-dependent states over-assert as PDB
                    # files); tilt_states.csv keeps each state's
                    # representative frame for extraction on demand.
                    rows = analyze_states(
                        tilt_result["tilt"], deltaz["deltaz"], states,
                        stride=ctx["stride"],
                        output_prefix=_op(ctx, "tilt_states"))
                    if rows:
                        pops = ", ".join(
                            f"state {r['state']}: "
                            f"{r['population_percent']:.1f}%" for r in rows)
                        out["logs"].append(f"FEL states{tag} — {pops}")
                        out["summary"][
                            _skey(ctx, "FEL states (population)")] = pops
                        out["captions"][os.path.basename(
                            _op(ctx, "tilt_deltaZ_FEL.png"))] = (
                            f"Tilt–ΔZ free-energy landscape: "
                            f"{len(rows)} state(s) — {pops}."
                            "State structures are deliberately not "
                            "exported (KDE states are projection-"
                            "dependent); tilt_states.csv lists each "
                            "state's representative trajectory frame "
                            "for extraction with gmx trjconv if "
                            "needed.")
            else:
                out["logs"].append(
                    f"Tilt–ΔZ FEL/state analysis skipped{tag}: enable ΔZ "
                    "together with tilt.")
            out["logs"].append(f"Tilt analysis completed{tag}")
    except Exception as e:
        out["error"] = f"ΔZ/Tilt analysis failed{tag}: {e}"
    return out


def job_tauoff(ctx, u=None):
    out = _res()
    tag = _tag(ctx)
    try:
        u = _universe(ctx, u)
        import numpy as _np
        from analysis.tauoff import (run_tauoff_analysis, write_tauoff_csv,
                                     plot_tauoff)
        result = run_tauoff_analysis(
            u, stride=ctx["stride"], protein_sel=ctx["pep_sel"],
            heads_sel=ctx["partner_sel"], anchor_sel=ctx["anchor_sel"],
            output_prefix=_op(ctx, "tauoff"))
        if result.get("status") == "skipped":
            out["logs"].append(
                f"WARNING: tau-off skipped{tag}: {result.get('reason')}.")
            return out
        write_tauoff_csv(result["tau"], result["ratios"], result["tauoff"],
                         _op(ctx, "tauoff.csv"))
        always = result.get("always_bound", False)
        note = None
        if always:
            note = (f"Peptide never unbound within the trajectory:\n"
                    f"residence ≥ {result['tauoff']:.0f} ns (lower bound, "
                    "1 censored event).\nSee the per-lipid and "
                    "site-resolved τ-off for kinetics.")
        plot_tauoff(result["tau"], result["ratios"], result["tauoff"],
                    _op(ctx, "tauoff.png"), note=note)
        fit = result.get("fit")
        if fit and always:
            out["summary"][_skey(ctx, "τ-off peptide (ns)")] = (
                f"≥ {result['tauoff']:.0f} (never unbound — lower bound; "
                "see per-lipid/site τ-off)")
        elif fit:
            lo, hi = fit["tau_off_ci"]
            out["summary"][_skey(ctx, "τ-off peptide (ns)")] = (
                f"{fit['tau_off']:.2f} (95% CI {lo:.2f}–{hi:.2f}, "
                f"{fit['model']}-exp, {fit['n_events']} events"
                + (", last event censored"
                   if result.get("censored") else "") + ")")
        lfit = result.get("lipid_fit")
        if lfit:
            lo, hi = lfit["tau_off_ci"]
            out["summary"][_skey(ctx, "τ-off per-lipid (ns)")] = (
                f"{lfit['tau_off']:.2f} (95% CI {lo:.2f}–{hi:.2f}, "
                f"{lfit['model']}-exp, {lfit['n_events']} pooled events)")
        out["summary"][_skey(ctx, "Bound fraction")] = (
            f"{result.get('bound_fraction', 0):.3f}")
        if always:
            out["captions"][os.path.basename(_op(ctx, "tauoff.png"))] = (
                "Binding-ratio curve: the peptide never unbound within "
                f"the trajectory — residence ≥ {result['tauoff']:.0f} ns "
                "(lower bound); kinetics come from the per-lipid and "
                "site-resolved analyses.")
            out["captions"][os.path.basename(
                _op(ctx, "tauoff_survival.png"))] = (
                "Whole-peptide survival: a single censored event "
                "(never unbound); not fitted.")
        elif fit:
            lo, hi = fit["tau_off_ci"]
            out["captions"][os.path.basename(_op(ctx, "tauoff.png"))] = (
                f"Binding-ratio curve; survival-fit τ-off = "
                f"{fit['tau_off']:.1f} ns ({fit['n_events']} events), "
                f"bound fraction {result.get('bound_fraction', 0):.2f}.")
            out["captions"][os.path.basename(
                _op(ctx, "tauoff_survival.png"))] = (
                f"Whole-peptide contact-lifetime survival: "
                f"{fit['model']}-exp fit, τ-off {fit['tau_off']:.1f} ns "
                f"(95% CI {lo:.1f}–{hi:.1f}).")
        if lfit:
            llo, lhi = lfit["tau_off_ci"]
            out["captions"][os.path.basename(
                _op(ctx, "tauoff_lipid_survival.png"))] = (
                f"Per-lipid residence (pooled over all lipids): "
                f"{lfit['n_events']} events, {lfit['model']}-exp "
                f"τ-off = {lfit['tau_off']:.2f} ns "
                f"(95% CI {llo:.2f}–{lhi:.2f}).")
        _prr = result.get("per_residue") or []
        if _prr:
            _top = max(_prr, key=lambda r: r[2])
            out["captions"][os.path.basename(
                _op(ctx, "tauoff_per_residue.png"))] = (
                "Per-residue residence: longest mean lifetime "
                f"{_top[0]} ({_top[2]:.1f} ns, bound fraction "
                f"{_top[3]:.2f}).")
        _spf = result.get("species_fits") or {}
        _sp_parts = []
        for _sp, _d in sorted(_spf.items()):
            if _d["fit"]:
                _sp_parts.append(f"{_sp}: {_d['fit']['tau_off']:.2f} ns "
                                 f"({_d['n_events']} ev)")
            else:
                _sp_parts.append(f"{_sp}: n/a ({_d['n_events']} ev)")
        if len(_spf) > 1:
            out["summary"][_skey(ctx, "τ-off by species (ns)")] = (
                "; ".join(_sp_parts))
            out["captions"][os.path.basename(
                _op(ctx, "tauoff_species.png"))] = (
                "Per-species pooled residence time (same event "
                "statistics as the per-lipid fit, split by lipid "
                "species): " + "; ".join(_sp_parts) + ".")
        out["logs"].append(f"tau-off analysis completed{tag}")
    except Exception as e:
        out["error"] = f"tau-off analysis failed{tag}: {e}"
    return out


def job_enrichment(ctx, u=None):
    out = _res()
    tag = _tag(ctx)
    try:
        u = _universe(ctx, u)
        import numpy as _np
        from analysis.enrichment import run_enrichment_analysis
        res = run_enrichment_analysis(
            u, protein_sel=ctx["pep_sel"], membrane_sel=ctx["membrane_sel"],
            stride=ctx["stride"], output_prefix=_op(ctx, "enrich"),
            anchor_sel=ctx["anchor_sel"], anchor_label=ctx["anchor_name"])
        if res.get("status") == "skipped":
            out["logs"].append(
                f"WARNING: Enrichment skipped{tag}: {res.get('reason')}.")
            return out
        enr = res.get("enrichment")
        if enr:
            for rn, v in sorted(enr["series"].items()):
                out["summary"][_skey(ctx, f"Enrichment E({rn})")] = (
                    f"{_np.nanmean(v):.2f}")
            _gap = int(_np.sum(_np.isnan(
                next(iter(enr["series"].values())))))
            _gap_note = (f" Gaps ({_gap} frames): no lipid within the "
                         "shell — peptide unbound; E is undefined there, "
                         "not zero." if _gap else "")
            out["captions"][os.path.basename(
                _op(ctx, "enrich_lipid_enrichment.png"))] = (
                "Lipid enrichment around the peptide (E > 1 = enriched): "
                + ", ".join(f"E({rn}) = {_np.nanmean(v):.2f}"
                            for rn, v in sorted(enr["series"].items()))
                + "." + _gap_note)
        _ct = res.get("contacts")
        if _ct is not None and len(_ct["contact_fraction"]):
            _order = _np.argsort(_ct["contact_fraction"])[::-1][:3]
            out["captions"][os.path.basename(
                _op(ctx, "enrich_residue_contacts.png"))] = (
                "Per-residue membrane contacts; top: " + ", ".join(
                    f"{_ct['resnames'][i]}{_ct['resids'][i]} "
                    f"({_ct['contact_fraction'][i]:.2f})"
                    for i in _order) + ".")
        out["logs"].append(f"Enrichment completed{tag}")
    except Exception as e:
        out["error"] = f"Enrichment failed{tag}: {e}"
    return out


def job_sites(ctx, u=None):
    out = _res()
    tag = _tag(ctx)
    try:
        u = _universe(ctx, u)
        from analysis.binding_sites import run_binding_site_analysis
        res = run_binding_site_analysis(
            u, protein_sel=ctx["pep_sel"],
            membrane_sel=ctx["sites_membrane_sel"] or ctx["membrane_sel"],
            anchor_sel=ctx["anchor_sel"], stride=ctx["stride"],
            output_prefix=_op(ctx, "sites"))
        if res.get("status") == "skipped":
            out["logs"].append(
                f"WARNING: Binding-site analysis skipped{tag}: "
                f"{res.get('reason')}.")
            return out
        sites = res["sites"]
        out["summary"][_skey(ctx, "Binding sites")] = (
            f"{len(sites)} site(s): " + "; ".join(
                f"site {s['site']} [{s['residues']}] occ "
                f"{100*s['occupancy']:.0f}%, τ {s['tau_off_ns']:.1f} ns"
                for s in sites))
        out["logs"].append(
            f"Binding sites{tag}: " + "; ".join(
                f"site {s['site']}: {s['residues']}" for s in sites))
        out["captions"][os.path.basename(
            _op(ctx, "sites_cocontact.png"))] = (
            f"Residue co-contact matrix; {len(sites)} site(s) detected "
            "(red blocks): " + "; ".join(
                f"site {s['site']} = {s['residues']}" for s in sites) + ".")
        for s in sites:
            out["captions"][os.path.basename(
                _op(ctx, f"sites_site{s['site']}_survival.png"))] = (
                f"Site {s['site']} ({s['residues']}): occupancy "
                f"{100*s['occupancy']:.0f}%, τ-off "
                f"{s['tau_off_ns']:.1f} ns "
                f"(CI {s['tau_off_ci_low']:.1f}–"
                f"{s['tau_off_ci_high']:.1f}), "
                f"composition {s['lipid_composition']}.")
        out["logs"].append(f"Binding-site analysis completed{tag}")
    except Exception as e:
        out["error"] = f"Binding-site analysis failed{tag}: {e}"
    return out


# Name -> function map used by the pool (functions must be importable
# by name in the spawned process).
JOB_FUNCS = {
    "apl": job_apl,
    "thickness": job_thickness,
    "scd": job_scd,
    "membrane": job_membrane,
    "distance": job_distance,
    "deltaz_tilt": job_deltaz_tilt,
    "tauoff": job_tauoff,
    "enrichment": job_enrichment,
    "sites": job_sites,
}


def run_job(name, ctx):
    """Pool entry point: dispatch by job name (picklable)."""
    _setup()
    return name, JOB_FUNCS[name](ctx)
