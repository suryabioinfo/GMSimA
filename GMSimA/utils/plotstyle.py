"""
Central matplotlib style for every smGROANA figure.

One rcParams profile applied before any analysis plotting keeps text
sizes consistent across all modules (titles, axis labels, tick labels,
legends) so no figure ends up with oversized ticks or clashing fonts.
Explicit per-call font sizes in the analysis modules still win where
they are set deliberately.
"""

import matplotlib


def apply_default_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round",
    })
