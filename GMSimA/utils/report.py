"""
Self-contained HTML run report for GMSimA.

generate_report() scans the run's output folder, embeds every plot as
base64 (the report is a single portable file), tabulates the key
numbers the worker registered, and lists every warning from the log.
"""

import os
import base64
import html
from datetime import datetime

# Section ordering: (title, list of filename prefixes)
_SECTIONS = [
    ("Area Per Lipid", ["apl"]),
    ("Membrane Thickness", ["thickness"]),
    ("Order Parameter", ["scd"]),
    ("Curvature / Density / Diffusion", ["membrane"]),
    ("Distance", ["distance"]),
    ("Insertion Depth (ΔZ)", ["deltaz"]),
    ("Residence Time (τ-off)", ["tauoff"]),
    ("Tilt / FEL / States", ["tilt", "struct"]),
    ("Enrichment / Contacts", ["enrich"]),
]


def _img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# Generic per-figure descriptions used when the worker did not register
# a run-specific caption (keyed by filename prefix, first match wins).
_FALLBACK_CAPTIONS = [
    ("apl_per_species", "Per-species Voronoi area per lipid (whole-membrane tessellation)."),
    ("apl", "Per-leaflet area per lipid from the lateral box area and leaflet lipid counts."),
    ("thickness", "Phosphate-to-phosphate membrane thickness over time."),
    ("scd_per_species", "C–H deuterium order parameter resolved by lipid species (sn-1 solid, sn-2 dashed)."),
    ("scd", "C–H deuterium order parameter per chain carbon (NMR convention)."),
    ("membrane_curvature_mesoscale", "Mesoscale undulation radius from the lipid-spacing-smoothed phosphate surface."),
    ("membrane_curvature", "Molecular-scale surface roughness radius of each leaflet (original GMSimA metric)."),
    ("membrane_density_species", "Whole-lipid mass density per species; z = 0 at the bilayer midplane."),
    ("membrane_density", "Component mass density profile (headgroup, glycerol, acyl chains, water); z = 0 at the bilayer midplane."),
    ("distance", "Minimum peptide–target distance ξ distribution; annotation gives the contact percentage below 3.8 Å."),
    ("deltaz_per_residue", "Mean per-residue position relative to the bilayer midplane (red: phosphate planes)."),
    ("deltaz", "Peptide center-of-mass ΔZ relative to the bilayer midplane with membrane reference lines."),
    ("tauoff_lipid_survival", "Survival of individual-lipid residence events pooled over all lipids, with exponential fit."),
    ("tauoff_survival", "Survival function of whole-peptide contact lifetimes with the selected exponential fit."),
    ("tauoff_species", "Per-species pooled residence time with 95% bootstrap intervals (bar = tau-off; n = pooled events)."),
    ("tauoff_per_residue", "Mean contact lifetime (bars) and bound fraction (dots) per anchor residue."),
    ("tauoff", "Cumulative binding-ratio curve of contact events (descriptive; τ-off comes from the survival fit)."),
    ("tilt_deltaZ_FEL", "Tilt–ΔZ free-energy landscape; stars mark detected states."),
    ("tilt_distribution", "Distribution of the peptide tilt angle."),
    ("tilt_timeseries", "Peptide tilt angle against the membrane normal over time."),
    ("enrich_lipid", "Lipid-species enrichment factor around the peptide (E > 1 = local enrichment)."),
    ("enrich_residue", "Fraction of frames each residue contacts the membrane (< 4.5 Å; basic residues in black)."),
    ("sites_cocontact", "Residue co-contact matrix; red blocks delimit the detected binding sites."),
    ("sites_site", "Residence survival of one detected binding site (events pooled over lipids)."),
    ("struct", "Representative structure export of an FEL state."),
]


def _caption_for(png_name, captions):
    if captions and png_name in captions:
        return captions[png_name]
    stem = png_name[:-4]  # drop .png
    # strip a per-peptide suffix like _pep1 for fallback matching
    for pre, text in _FALLBACK_CAPTIONS:
        if stem.startswith(pre) or stem.replace("_pep", "#").startswith(pre):
            return text
    return ""


def _collect_warnings(log_path):
    warnings = []
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "WARNING:" in line:
                    warnings.append(line.split("WARNING:", 1)[1].strip())
    except Exception:
        pass
    return warnings


def generate_report(output_dir, metadata=None, summary=None,
                    log_name="GMSimA.log", outfile="report.html",
                    captions=None):
    """
    metadata : dict of run facts (files, stride, groups, ...)
    summary  : dict of key numbers registered by the worker
    captions : {png_basename: short description} written under each
               figure; figures without an entry get a generic fallback
    Returns the report path, or None on failure (never raises).
    """
    try:
        metadata = metadata or {}
        summary = summary or {}
        pngs = sorted(p for p in os.listdir(output_dir)
                      if p.lower().endswith(".png"))
        warnings = _collect_warnings(os.path.join(output_dir, log_name))

        parts = []
        parts.append(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>GMSimA Report</title><style>"
            "body{font-family:sans-serif;max-width:1000px;margin:24px auto;"
            "padding:0 16px;color:#222;background:#fafafa;}"
            "h1{border-bottom:2px solid #444;padding-bottom:6px;}"
            "h2{margin-top:34px;border-bottom:1px solid #bbb;"
            "padding-bottom:4px;}"
            "table{border-collapse:collapse;margin:10px 0;}"
            "td,th{border:1px solid #ccc;padding:5px 12px;text-align:left;"
            "font-size:14px;}th{background:#eee;}"
            "img{max-width:100%;border:1px solid #ddd;background:#fff;"
            "margin:8px 0;}"
            ".warn{background:#fff3cd;border:1px solid #ffe08a;"
            "border-radius:4px;padding:8px 12px;margin:6px 0;font-size:14px;}"
            ".meta{color:#666;font-size:13px;}"
            ".cap{color:#333;font-size:14px;background:#eef3f8;"
            "border-left:3px solid #2E5395;border-radius:0 4px 4px 0;"
            "padding:7px 12px;margin:0 0 18px 0;}"
            "</style></head><body>")
        parts.append("<h1>GMSimA Analysis Report</h1>")
        parts.append(f"<p class='meta'>Generated "
                     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        if metadata:
            parts.append("<h2>Run</h2><table>")
            for k, v in metadata.items():
                parts.append(f"<tr><th>{html.escape(str(k))}</th>"
                             f"<td>{html.escape(str(v))}</td></tr>")
            parts.append("</table>")

        if warnings:
            parts.append(f"<h2>Warnings ({len(warnings)})</h2>")
            for w in warnings:
                parts.append(f"<div class='warn'>{html.escape(w)}</div>")

        if summary:
            parts.append("<h2>Key Results</h2><table>")
            for k, v in summary.items():
                parts.append(f"<tr><th>{html.escape(str(k))}</th>"
                             f"<td>{html.escape(str(v))}</td></tr>")
            parts.append("</table>")

        used = set()
        for title, prefixes in _SECTIONS:
            imgs = [p for p in pngs
                    if any(p.startswith(pre) for pre in prefixes)
                    and p not in used]
            if not imgs:
                continue
            used.update(imgs)
            parts.append(f"<h2>{html.escape(title)}</h2>")
            for p in imgs:
                try:
                    b64 = _img_b64(os.path.join(output_dir, p))
                except Exception:
                    continue
                parts.append(f"<p class='meta'>{html.escape(p)}</p>"
                             f"<img src='data:image/png;base64,{b64}'>")
                cap = _caption_for(p, captions)
                if cap:
                    parts.append(f"<div class='cap'>{html.escape(cap)}"
                                 "</div>")

        leftovers = [p for p in pngs if p not in used]
        if leftovers:
            parts.append("<h2>Other Plots</h2>")
            for p in leftovers:
                try:
                    b64 = _img_b64(os.path.join(output_dir, p))
                except Exception:
                    continue
                parts.append(f"<p class='meta'>{html.escape(p)}</p>"
                             f"<img src='data:image/png;base64,{b64}'>")
                cap = _caption_for(p, captions)
                if cap:
                    parts.append(f"<div class='cap'>{html.escape(cap)}"
                                 "</div>")

        parts.append("</body></html>")

        out_path = os.path.join(output_dir, outfile)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(parts))
        return out_path
    except Exception:
        return None
