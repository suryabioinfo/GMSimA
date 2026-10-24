"""
GMSimA — About tab content.

Edit THIS file to update the About tab; nothing in main_window.py needs
to change. The tab renders ABOUT_HTML in a scrollable, link-enabled
QTextBrowser. Update the constants below (version, contact, citation)
or the HTML body itself as the tool evolves.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

# ----------------------------------------------------------------------
# Editable metadata
# ----------------------------------------------------------------------
APP_NAME = "GMSimA"
APP_VERSION = "3.0.0"
APP_TAGLINE = "Protein / Peptide–Membrane GROmacs ANAlysis Suite"
DEVELOPER = "Surya Pratap Singh; PhD"
DEV_EMAIL = "ssurya@uratpx.com"
DEV_PROFILE_URL = "https://github.com/yourusername"  # will be linked in the in near future
CITATION = ("Singh, SP et al. GMSimA: an integrated GUI for protein/peptide–membrane "
    "trajectory analysis (manuscript in preparation).")

# ----------------------------------------------------------------------
# About page (rich HTML rendered by QTextBrowser)
# ----------------------------------------------------------------------
ABOUT_HTML = f"""
<html>
<body style="font-family: sans-serif; font-size: 11pt; margin: 14px;">

<h1 style="margin-bottom: 2px; color: #000;">{APP_NAME} <span style="font-size: 11pt;
font-weight: normal; color: #000;">v{APP_VERSION}</span></h1>
<p style="margin-top: 0; color: #000;"><i>{APP_TAGLINE}</i></p>

<hr>

<h2>What it does</h2>
<p>
{APP_NAME} is a graphical analysis suite for GROMACS membrane simulations,
built for studying how proteins and peptides interact with lipid bilayers.
It reads a portable run input file (<b>.tpr</b>) plus trajectory
(<b>.xtc</b>) and produces publication-ready plots
together with the underlying data in open formats
(<b>.xvg</b>/<b>.csv</b>/<b>.txt</b>) for replotting in Grace, gnuplot,
or Python.
</p>

<h2>Analyses</h2>

<h3>Intrinsic membrane properties</h3>
<ul>
<li><b>Area per lipid (APL)</b> — per-leaflet time series from the lateral
box area and leaflet lipid counts.</li>
<li><b>Membrane thickness</b> — phosphate-to-phosphate distance between
leaflet mean planes over time.</li>
<li><b>Deuterium order parameter (S<sub>CD</sub>)</b> — per-carbon chain
order for the sn-1 and sn-2 acyl chains, with automatic detection of the
unsaturated chain from C=C bond lengths.</li>
<li><b>Membrane curvature</b> — per-leaflet surface roughness/curvature
descriptor from a Monge-patch Laplacian of the interpolated phosphate
surface.</li>
<li><b>Density profile</b> — mass density of headgroups, glycerol
backbone, acyl chains, and water along the membrane normal.</li>
<li><b>Lateral diffusion</b> — lipid MSD with periodic-boundary
unwrapping and leaflet-COM drift removal; D from the linear MSD regime.</li>
</ul>

<h3>Protein/peptide–membrane interactions</h3>
<ul>
<li><b>Distance (&xi;)</b> — biomolecule–membrane separation over time.</li>
<li><b>Binding residence (&tau;-off)</b> — dual-cutoff (4.5/7.0&nbsp;Å)
contact events with survival-function fits. V3 interaction matrix: the
anchor (basic, acidic, aromatic, hydrophobic, polar, all, or custom),
the membrane partner (whole lipids, phosphates, headgroup+glycerol,
acyl chains, sterol hydroxyls) and a lipid-species filter are composed
independently — no chemistry is privileged. Whole-peptide, pooled
per-lipid AND per-species residence times with bootstrap confidence
intervals; presets editable in config/interactions.py.</li>
<li><b>Binding sites</b> — residue-community clustering of the
residue–lipid co-contact network (Louvain); per-site occupancy,
lipid composition, site-resolved &tau;-off, and representative bound
poses.</li>
<li><b>Penetration (&Delta;z)</b> — insertion depth of the biomolecule
relative to the phosphate plane.</li>
<li><b>Tilt / FEL </b> — helix tilt angle time series; 2D
tilt–&Delta;z free-energy landscape; detection of well-separated
FEL states with occupancy statistics, and export of a representative
is recorded in tilt_states.csv for each state, which can be extracted with 
gmx trjconv -dump if needed.</li>
</ul>

<h2>Recommended trajectory preprocessing</h2>
<p>
For reliable results, analyze a trajectory in which molecules are whole
and the membrane is centered:
</p>
<p style="font-family: monospace; background-color: #2b2b2b; color: #d8d8d8;
padding: 8px; border-radius: 4px;">
gmx trjconv -s md.tpr -f md.xtc -pbc mol -center -ur compact -o traj.xtc
</p>
<p>
Additionally rotational+translational fit
(<span style="font-family: monospace;">-fit rot+trans</span> on the
membrane) is recommended if the membrane drifts. {APP_NAME} checks the
loaded trajectory for common preprocessing problems (broken molecules,
tilted or off-center membranes) and reports <b>warnings</b> in the log —
analyses always proceed, so please review warnings before interpreting
results.
</p>

<h2>Outputs</h2>
<p>
All results are written to the chosen output folder: plots as
<b>.png</b>, data as <b>.xvg</b>, <b>.csv</b>, and
<b>.txt</b>, and representative structures as <b>.pdb</b>. A full run
log (<span style="font-family: monospace;">GMSimA.log</span>)
records every step and warning.
</p>

<h2>Performance</h2>
<p>
{APP_NAME} runs the enabled analysis modules concurrently on multiple CPU
cores. Choose the number of processes with the &quot;CPU cores&quot;
spin box (max: machine cores &minus; 1; set 1 for a
sequential run). It is recommended to use maximum 4 cores on a machine with 8 GiB memory (Choose cores carefully!).
</p>

<h2>Requirements</h2>
<p>
Python&nbsp;3.9+, MDAnalysis, NumPy, SciPy, pandas, matplotlib,
NetworkX, PySide6. Currently supports only all atom force-field (CHARMM36 and AMBER Lipid14/17/21 family);
multiple peptides in one system are analyzed individually
(auto-detected bonded fragments). 
Tested for trajectories from GROMACS 2022 and later versions.
</p>

<h2>If GMSimA contributes to your work, please cite:</h2>
<p>{CITATION}</p>

<h2>Developer</h2>
<p style="margin-bottom: 4px;">
  <b>{DEVELOPER}</b>
</p>
<p style="font-size: 10pt;">
  <!-- Email link -->
  <b>Email:</b> <a href="mailto:{DEV_EMAIL}" style="color: #7aa2f7;">{DEV_EMAIL}</a><br>
  <!-- Web/ORCID link -->
  <b>Profile:</b> <a href="{DEV_PROFILE_URL}" style="color: #7aa2f7;">Research Profile / GitHub</a>
</p>

<hr>
<p style="color: #000000; font-size: 9pt;">
&copy; 2026 {APP_NAME} developers. Provided as-is, without warranty;
always validate analysis output against your system setup.
</p>

</body>
</html>
"""


def build_about_widget(parent=None):
    """Return the widget displayed in the About tab."""
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)

    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)

    # Set widget background, font color, and remove default border
    browser.setStyleSheet("""
        QTextBrowser {
            background-color: #757575;
            color: #FFFFFF;
            border: none;
        }
    """)

    browser.setHtml(ABOUT_HTML)

    layout.addWidget(browser)
    return widget
