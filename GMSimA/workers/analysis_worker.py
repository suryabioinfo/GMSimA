from PySide6.QtCore import QThread, Signal
import os, warnings

# Safety net: the analysis modules below import matplotlib.pyplot and
# this worker runs them inside a QThread. Plotting MUST use the non-GUI
# "Agg" backend — the Qt backend called from a worker thread corrupts
# Qt state and segfaults on exit. (main.py sets this too; keeping it
# here protects any headless/scripted use of the worker.)
import matplotlib
matplotlib.use("Agg", force=True)
from utils.plotstyle import apply_default_style
apply_default_style()

import MDAnalysis as mda
from core.universe_loader import load_universe, UnsupportedTPRError
from datetime import datetime
from analysis.apl import (calculate_apl_timeseries, write_apl_xvg, plot_apl,
                          calculate_apl_per_species,
                          write_apl_per_species_csv, plot_apl_per_species)
from analysis.thickness import calculate_thickness_timeseries, write_thickness_xvg, plot_thickness
from analysis.scd import (calculate_scd, write_scd_xvg, plot_scd,
                          write_scd_per_species_csv, plot_scd_per_species)
from core.leaflet_detection import detect_leaflets
from analysis.deltaz import (calculate_deltaz, write_deltaz_xvg, plot_deltaz,
                             plot_deltaz_density,
                             write_deltaz_per_residue_csv,
                             plot_deltaz_per_residue)
from analysis.tilt import (compute_tilt_timeseries, write_tilt_xvg, plot_tilt_timeseries,
                           plot_tilt_distribution, plot_tilt_deltaz_fel,
                           export_state_structures, analyze_states)
from analysis.tauoff import run_tauoff_analysis,write_tauoff_csv,plot_tauoff
from analysis.binding_sites import run_binding_site_analysis
from analysis.distance import run_distance_analysis
from analysis.curvature import run_membrane_module
from analysis.enrichment import run_enrichment_analysis
from core.trajectory_check import check_trajectory_preprocessing
from core.lipids import (membrane_selection, phosphate_selection,
                         anionic_selection)
from utils.logger import SmgroanaLogger
from utils.report import generate_report

# Suppress all the md analysis warnings that arise during structure writing.

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.coordinates.PDB")


# LEGACY TEST CODE
# apl = calculate_apl_timeseries(u, upper_leaflet, lower_leaflet)
# write_apl_xvg("apl.xvg", apl)
# plot_apl("apl.png", apl)


class AnalysisWorker(QThread):
   
    
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    



    def __init__(self, tpr_file, xtc_file, output_dir, analysis_options,
                 stride, protein_sel="", membrane_sel="",
                 anchor_choice="Basic (Lys/Arg)",
                 partner_choice="Auto", species_choice="All species",
                 parallel=True, n_cores=0,
                 axis_mode="auto", tm_segments=""):

        super().__init__()

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")       # Timestamp for New directory everytime.
        
        self.tpr = tpr_file
        self.xtc = xtc_file
        #self.output_dir = output_dir         # Reactivate to get old behavior
        self.output_dir = os.path.join(output_dir, f"GMSimA_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.last_output_dir = self.output_dir
        
        self.analysis = analysis_options
        self.stride = stride
        self._is_cancelled = False

        # GUI group selections ("auto"/"" -> resolved at run time from
        # the topology; these drive the INTERACTION analyses only —
        # intrinsic membrane modules keep their internal selections).
        self.protein_sel_raw = (protein_sel or "").strip()
        self.membrane_sel_raw = (membrane_sel or "").strip()
        self.anchor_choice = (anchor_choice or "Basic (Lys/Arg)").strip()
        # V3 interaction matrix: membrane-side partner kind and species
        # filter, composed independently of the anchor.
        self.partner_choice = (partner_choice or "Auto").strip()
        self.species_choice = (species_choice or "All species").strip()
        # V3.1 multi-core execution: parallel=False forces the
        # sequential in-process path; n_cores=0 means auto
        # (min(#jobs, cpu_count - 1)).
        self.parallel = bool(parallel)
        self.n_cores = int(n_cores or 0)
        # v3.3 tilt: how the biomolecule's axis is defined (see
        # analysis/orientation.py). "" / None for tm_segments means the
        # transmembrane segments are auto-detected from the backbone.
        self.axis_mode = (axis_mode or "auto").strip()
        self.tm_segments = (tm_segments or "").strip() or None
        self.biomol_selection = "protein"          # resolved in run()
        # Optional explicit override for the distance-module target;
        # None -> derived from the membrane group (anionic phosphates,
        # else all lipid phosphates).
        self.distance_target = None
        
        self.logger = SmgroanaLogger(self.output_dir)
        # Key numbers registered by each module for the HTML report.
        self.summary = {}
        # Per-figure captions ({png basename: text}) for the report;
        # figures without an entry get a generic fallback description.
        self.captions = {}

    def _cap(self, path_or_name, text):
        """Register the caption shown under one figure in report.html."""
        self.captions[os.path.basename(path_or_name)] = text

    # ------------- V2 helpers -------------

    def _peptide_units(self, u):
        """
        Split the resolved protein group into bonded fragments so every
        peptide is analyzed on its own. Returns [(label, selection)];
        with a single fragment, [( "", biomol_selection )] so the V1
        behaviour and file names are unchanged.
        """
        import numpy as _np
        try:
            prot = u.select_atoms(self.biomol_selection)
            frags = prot.fragments
        except Exception:
            return [("", self.biomol_selection)]
        if len(frags) <= 1:
            return [("", self.biomol_selection)]

        # labels: unique non-blank segids where possible, else pepN
        segids = []
        for frag in frags:
            s = {str(x).strip() for x in frag.segments.segids}
            segids.append(s.pop() if len(s) == 1 else "")
        use_segids = (all(segids)
                      and len(set(segids)) == len(segids))

        units = []
        for i, frag in enumerate(frags):
            label = segids[i] if use_segids else f"pep{i+1}"
            ix = frag.indices
            if len(ix) > 1 and (_np.diff(ix) == 1).all():
                sel = f"index {ix[0]}:{ix[-1]}"
            else:
                sel = "index " + " ".join(map(str, ix))
            units.append((label, sel))
        return units

    def _resolve_anchor(self, u):
        """
        V3 interaction matrix: compose (anchor selection, membrane
        partner selection, description) from the three independent GUI
        axes — anchor preset/custom selection, membrane partner kind
        (Auto picks a sensible partner per preset), and lipid-species
        filter. No chemistry is privileged: cationic-on-phosphates is
        just one composition of the matrix. Presets are defined in the
        editable config/interactions.py.
        """
        from config.interactions import resolve_anchor, auto_partner_for
        from core.lipids import partner_selection

        anchor, anchor_name = resolve_anchor(self.anchor_choice)

        kind = self.partner_choice
        if kind.strip().lower() == "auto":
            kind = auto_partner_for(anchor_name)

        species = self.species_choice
        if species.strip().lower() in ("all species", "all", ""):
            species = None

        try:
            partner, pdesc = partner_selection(u, kind, species)
        except ValueError:
            # unknown kind typed into the editable combo -> use it
            # verbatim as a membrane-side selection string
            partner = f"({kind}) and not name H*"
            pdesc = f"custom partner ({kind})"
        # empty partner (e.g. sterol hydroxyls without sterols): warn
        # and fall back to whole lipids so the run still completes
        try:
            if u.select_atoms(partner).n_atoms == 0:
                self.logger.write(
                    f"WARNING: membrane partner '{pdesc}' matches no "
                    "atoms in this system — falling back to whole "
                    "lipids (heavy atoms).")
                partner, pdesc = partner_selection(u, "whole", species)
        except Exception:
            pass

        desc = f"{anchor_name} vs {pdesc}"
        if species:
            desc += f" [species: {species}]"
        self._pairing = {"anchor_name": anchor_name, "partner_desc": pdesc,
                         "species": species or "all"}
        # binding sites treat WHOLE lipids as ligands (the site's lipid
        # composition needs the full molecule); only the species filter
        # carries over from the matrix, never the partner kind.
        try:
            self._sites_membrane_sel, _ = partner_selection(
                u, "whole", species)
        except Exception:
            self._sites_membrane_sel = None
        return anchor, partner, desc

    def _n_workers(self, n_jobs):
        """Worker-process count for the job pool (1 = sequential)."""
        if not self.parallel or n_jobs <= 1:
            return 1
        auto = max(1, (os.cpu_count() or 2) - 1)
        n = min(n_jobs, auto)
        if self.n_cores > 0:
            n = min(n, self.n_cores)
        return max(1, n)

    # ------------- Main Analysis Logics -------------
    def run(self):

        self.status.emit("Loading trajectory")
        self.logger.write("Loading trajectory")

        try:
            with warnings.catch_warnings(record=True) as _w:
                warnings.simplefilter("always", RuntimeWarning)
                u = load_universe(self.tpr, self.xtc)
            for _m in _w:
                if "cannot read" in str(_m.message) or "no guessable mass" in str(_m.message):
                    self.logger.write(f"WARNING: {_m.message}")

            from core.universe_loader import tpr_backend, BACKEND_LABELS
            self.topology_label = BACKEND_LABELS.get(tpr_backend(u), "?")
            self.logger.write("Topology read as: " + self.topology_label)

        except UnsupportedTPRError as e:
            # TPR newer than MDAnalysis and no fallback installed
            self.logger.write(str(e))
            self.error.emit(str(e))
            return

        except Exception as e:

            message = (
                "Could not load topology + XTC.\n\n"
                f"{e.__class__.__name__}: {e}\n\n"
                "Common reasons:\n"
                "• Topology and XTC come from different simulations\n"
                "• Wrong topology file (e.g. full vs. dry/stripped system —\n"
                "  the topology must have exactly the XTC's atoms)\n"
                "• Trajectory stripped incorrectly\n"
            )

            self.logger.write(message)
            self.error.emit(message)

            return

        self.progress.emit(10)

        # ------------------------------------------------------------
        # Warn-only preprocessing sanity checks.
        #
        # GMSimA expects '-pbc mol -center -ur compact' (optionally
        # rot+trans fitted) trajectories. Preprocessing is owned by an
        # upstream stage of the suite, so anomalies are REPORTED but
        # never block the analysis.
        # ------------------------------------------------------------
        try:
            preprocessing_warnings = check_trajectory_preprocessing(u)
        except Exception as exc:
            preprocessing_warnings = []
            self.logger.write(f"Preprocessing check skipped: {exc}")
        for warning_msg in preprocessing_warnings:
            self.logger.write(f"WARNING: {warning_msg}")
            self.status.emit(f"WARNING: {warning_msg}")

        # ------------------------------------------------------------
        # Resolve the GUI group selections (warn-only, never blocking).
        # "auto"/blank -> topology-derived defaults; an invalid or empty
        # custom selection falls back to auto with a logged warning.
        # ------------------------------------------------------------
        def _resolve(raw, auto_value, label):
            if not raw or raw.lower() == "auto":
                return auto_value
            try:
                if len(u.select_atoms(raw)) > 0:
                    return raw
                msg = (f"{label} selection '{raw}' matches no atoms; "
                       f"falling back to '{auto_value}'.")
            except Exception as exc:
                msg = (f"{label} selection '{raw}' is invalid ({exc}); "
                       f"falling back to '{auto_value}'.")
            self.logger.write(f"WARNING: {msg}")
            self.status.emit(f"WARNING: {msg}")
            return auto_value

        auto_membrane = membrane_selection(u) or "name P"
        self.biomol_selection = _resolve(self.protein_sel_raw, "protein",
                                         "Protein group")
        self.membrane_selection = _resolve(self.membrane_sel_raw,
                                           auto_membrane, "Membrane group")
        self.logger.write(f"Protein group: {self.biomol_selection}")
        self.logger.write(f"Membrane group: {self.membrane_selection}")

        # Phosphate atoms of the chosen membrane group (fallback: name P)
        try:
            self.phosphate_sel = phosphate_selection(
                u, base_selection=self.membrane_selection)
            if len(u.select_atoms(self.phosphate_sel)) == 0:
                self.phosphate_sel = "name P"
        except Exception:
            self.phosphate_sel = "name P"

        # Detect leaflets
        self.status.emit("Detecting membrane leaflets")
        self.logger.write("Detecting membrane leaflets")

        leaflets = detect_leaflets(u)

        upper = leaflets["upper_leaflet"]
        lower = leaflets["lower_leaflet"]

        #self.progress.emit(30)            # Reactivate if needed

        active_modules = [k for k, v in self.analysis.items() if v]
        n_modules = len(active_modules)

        base_progress = 30
        remaining = 70

        step = remaining / max(n_modules, 1)
        current_progress = base_progress

        mean_thickness = None # Initialize mean thickness variable for later use in ΔZ plot
        surface_z = None # Initialize surface_z variable for later use in ΔZ density plot

        ###################################################################
        # V3.1: job-based execution (single-core or multi-core).
        #
        # Every module now lives in workers/jobs.py as a top-level job
        # function returning {summary, captions, logs, error}. The SAME
        # jobs run either sequentially in this thread (reusing the open
        # Universe) or in a multiprocessing pool ('spawn' context; each
        # process opens its own Universe) — so parallel and sequential
        # runs produce identical outputs by construction.
        ###################################################################

        import numpy as _np
        import time as _time
        from workers import jobs as J

        peptide_units = self._peptide_units(u)
        multi = len(peptide_units) > 1
        if multi:
            names = ", ".join(l for l, _ in peptide_units)
            self.logger.write(f"Multiple peptides detected: {names}")
            self.summary["Peptides analyzed"] = names

        # V3 interaction matrix: anchor x membrane partner x species,
        # used by every residence/site analysis below
        anchor_sel, partner_sel, anchor_desc = self._resolve_anchor(u)
        self.logger.write(f"Interaction pairing: {anchor_desc}")
        self.summary["Interaction pairing"] = anchor_desc

        base_ctx = {
            "tpr": self.tpr, "xtc": self.xtc, "stride": self.stride,
            "output_dir": self.output_dir,
            "membrane_sel": self.membrane_selection,
            "phosphate_sel": self.phosphate_sel,
            "anchor_sel": anchor_sel, "partner_sel": partner_sel,
            "sites_membrane_sel": getattr(self, "_sites_membrane_sel",
                                          None),
            "anchor_name": (self._pairing.get("anchor_name")
                            if hasattr(self, "_pairing") else None),
            "multi": multi, "pep_label": "",
            "pep_sel": self.biomol_selection, "sfx": "",
            "mean_thickness": None,
            "do_deltaz": bool(self.analysis.get("deltaz")),
            "do_tilt": bool(self.analysis.get("tilt")),
            "axis_mode": self.axis_mode,
            "tm_segments": self.tm_segments,
        }

        # ---- build the job list in the CANONICAL analysis order ----
        # APL -> Thickness -> Order parameter -> Curvature/Density/
        # Diffusion -> per peptide: Distance -> tau-off -> ΔZ/Tilt/FEL
        # -> Enrichment/Contacts -> Binding sites. Deselecting modules
        # never reorders the rest: the enabled subset keeps this
        # sequence in the status flow, the log and the merged results.
        job_list = []          # (label, job_name, ctx)
        if self.analysis.get("apl"):
            job_list.append(("APL", "apl", dict(base_ctx)))
        if self.analysis.get("thickness"):
            job_list.append(("Thickness", "thickness", dict(base_ctx)))
        if self.analysis.get("order"):
            job_list.append(("SCD", "scd", dict(base_ctx)))
        if self.analysis.get("curvature"):
            job_list.append(("Membrane module", "membrane", dict(base_ctx)))

        for pep_label, pep_sel in peptide_units:
            tag = f" [{pep_label}]" if multi else ""
            pctx = dict(base_ctx)
            pctx["pep_label"] = pep_label
            pctx["pep_sel"] = pep_sel
            pctx["sfx"] = f"_{pep_label}" if multi else ""

            if self.analysis.get("distance"):
                dctx = dict(pctx)
                target = self.distance_target
                if not target:
                    target = anionic_selection(
                        u, base_selection=self.membrane_selection)
                    if not target:
                        target = self.phosphate_sel
                        self.logger.write(
                            "Distance target: no anionic lipids in the "
                            "membrane group — using all lipid phosphates.")
                self.logger.write(f"Distance target selection: {target}")
                dctx["distance_target"] = target
                selA = f"({pep_sel}) and not name H* and not backbone"
                try:
                    if len(u.select_atoms(selA)) == 0:
                        selA = f"({pep_sel}) and not name H*"
                except Exception:
                    selA = f"({pep_sel}) and not name H*"
                dctx["distance_selA"] = selA
                job_list.append((f"Distance{tag}", "distance", dctx))

            if self.analysis.get("tauoff"):
                job_list.append((f"tau-off{tag}", "tauoff", dict(pctx)))
            if self.analysis.get("deltaz") or self.analysis.get("tilt"):
                job_list.append((f"ΔZ/Tilt{tag}", "deltaz_tilt",
                                 dict(pctx)))
            if self.analysis.get("enrichment"):
                job_list.append((f"Enrichment{tag}", "enrichment",
                                 dict(pctx)))
            if self.analysis.get("binding_sites"):
                job_list.append((f"Binding sites{tag}", "sites",
                                 dict(pctx)))

        n_total = len(job_list)
        base_progress = 30
        step = 70.0 / max(n_total, 1)
        current_progress = float(base_progress)

        def _merge(label, res):
            self.summary.update(res.get("summary", {}))
            self.captions.update(res.get("captions", {}))
            for line in res.get("logs", []):
                self.logger.write(line)
                if line.startswith("WARNING"):
                    self.status.emit(line)
            if res.get("error"):
                self.logger.write(res["error"])
                self.error.emit(res["error"])

        def _propagate_thickness(res, remaining):
            """ΔZ reference lines need the mean thickness: hand the
            thickness job's result to every not-yet-run/submitted ctx."""
            mt = res.get("extra", {}).get("mean_thickness")
            if mt is not None:
                for _, _, c in remaining:
                    c["mean_thickness"] = mt

        n_workers = self._n_workers(len(job_list))
        if n_workers > 1 and len(job_list) > 1:
            # ---------------- multi-core execution ----------------
            # Thickness (if enabled) completes BEFORE the pool starts,
            # because the ΔZ/Tilt job needs its mean for the reference
            # lines — but it is merged in canonical order (after APL),
            # so the log/report sequence matches a sequential run.
            import multiprocessing as _mp
            results = {}
            pool_jobs = list(job_list)
            for i, (label, name, c) in enumerate(pool_jobs):
                if name == "thickness":
                    self.status.emit("Computing membrane thickness")
                    tres = J.JOB_FUNCS[name](c, u)
                    results[label] = tres
                    _propagate_thickness(tres, pool_jobs)
                    pool_jobs.pop(i)
                    current_progress += step
                    self.progress.emit(int(current_progress))
                    if self._is_cancelled:
                        _merge(label, tres)
                        self._stopped_by_user("Thickness")
                        return
                    break

            self.status.emit(
                f"Running {len(pool_jobs)} modules on {n_workers} cores")
            self.logger.write(
                f"Parallel execution: {len(pool_jobs)} jobs on "
                f"{n_workers} processes (spawn)")
            mpctx = _mp.get_context("spawn")
            pool = mpctx.Pool(processes=n_workers)
            try:
                pending = [(label, name,
                            pool.apply_async(J.run_job, (name, c)))
                           for label, name, c in pool_jobs]
                pool.close()
                n_done = 0
                while n_done < len(pending):
                    if self._is_cancelled:
                        pool.terminate()
                        pool.join()
                        # merge whatever finished, in canonical order
                        for label, name, c in job_list:
                            if label in results:
                                _merge(label, results[label])
                        self._stopped_by_user(
                            "current batch (unfinished parallel jobs "
                            "were discarded)")
                        return
                    for label, name, ar in pending:
                        if label in results or not ar.ready():
                            continue
                        try:
                            _, r = ar.get()
                        except Exception as e:
                            r = {"summary": {}, "captions": {},
                                 "logs": [], "extra": {},
                                 "error": f"{label} crashed: {e}"}
                        results[label] = r
                        n_done += 1
                        current_progress += step
                        self.progress.emit(int(current_progress))
                        self.status.emit(
                            f"{label} finished "
                            f"({n_done}/{len(pending)})")
                    _time.sleep(0.25)
                pool.join()
            finally:
                try:
                    pool.terminate()
                except Exception:
                    pass
            # merge in CANONICAL order, not completion order
            for label, name, c in job_list:
                _merge(label, results.get(label, {}))
        else:
            # ---------------- sequential execution ----------------
            # runs strictly in the canonical order above
            for i, (label, name, c) in enumerate(job_list):
                self.status.emit(f"Running {label}")
                self.logger.write(f"Running {label}")
                res = J.JOB_FUNCS[name](c, u)
                _merge(label, res)
                if name == "thickness":
                    _propagate_thickness(res, job_list[i + 1:])
                current_progress += step
                self.progress.emit(int(current_progress))
                self.status.emit(f"{label} completed")
                if self._is_cancelled:
                    self._stopped_by_user(label)
                    return

        # ---------------- HTML run report ----------------
        try:
            metadata = {
                "Topology": os.path.basename(self.tpr),
                "Topology read as": getattr(self, "topology_label", "?"),
                "XTC": os.path.basename(self.xtc),
                "Stride": self.stride,
                "Protein group": self.biomol_selection,
                "Membrane group": self.membrane_selection,
                "Frames analyzed": len(u.trajectory[::self.stride]),
            }
            path = generate_report(self.output_dir, metadata=metadata,
                                   summary=self.summary,
                                   captions=self.captions)
            if path:
                self.logger.write(f"HTML report written: "
                                  f"{os.path.basename(path)}")
        except Exception as e:
            self.logger.write(f"Report generation failed: {e}")

        self.progress.emit(100)

        self.finished.emit(self.output_dir)

    # ------------- Job Cancellation -------------
    def cancel(self):
        """Called from the GUI Stop button: acknowledge immediately;
        the run stops at the next module boundary."""
        self._is_cancelled = True
        self.status.emit("Stop requested — finishing the current module, "
                         "then stopping...")
        self.logger.write("Stop requested by user")

    def _stopped_by_user(self, module):
        """
        Called at every cancellation checkpoint. Reports clearly that
        the USER stopped the analysis (not an error), notes which
        modules completed, and emits finished so the partial results
        stay accessible via the Results button.
        """
        msg = (f"Analysis stopped by user after the {module} module. "
               "Completed results (and the report) are in "
               f"'{os.path.basename(self.output_dir)}'.")
        self.logger.write(msg)
        try:
            metadata = {"Topology": os.path.basename(self.tpr),
                        "XTC": os.path.basename(self.xtc),
                        "Stride": self.stride,
                        "Status": f"STOPPED BY USER after {module}"}
            generate_report(self.output_dir, metadata=metadata,
                            summary=self.summary,
                            captions=self.captions)
        except Exception:
            pass
        self.finished.emit(self.output_dir)
        self.status.emit(msg)
