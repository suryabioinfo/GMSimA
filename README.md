# GMSimA

**s**mall-**m**olecule / peptide–membrane **GRO**macs **ANA**lysis — a PySide6
GUI suite for analyzing protein/peptide–lipid bilayer simulations from
GROMACS (`.tpr`/`.gro`/`.pdb` + `.xtc`), built on MDAnalysis.

GMSimA produces publication-quality figures (300 dpi PNG), the underlying
data in open formats (`.xvg`/`.csv`/`.txt`), representative structures
(`.pdb`), a full run log, and a self-contained `report.html` per run.

## Available Analyses

**Intrinsic membrane properties** (always computed on the whole membrane;
species-resolved results are decompositions, never restricted calculations):

- **Area per lipid** — per-leaflet box-area APL, plus per-species Voronoi APL
- **Membrane thickness** — phosphate-to-phosphate, per frame
- **Order parameter (S_CD)** — true C–H bond-vector NMR convention
  (directly comparable to experiment and `gmx order`), per chain and per
  species; chord-vector fallback for united-atom topologies
- **Curvature** — molecular-scale roughness radius (primary) and
  lipid-spacing-smoothed mesoscale undulation radius (secondary)
- **Density profile** — headgroup/glycerol/chains/water, plus per species
- **Lateral diffusion** — PBC-unwrapped, COM-corrected, FFT-based MSD with
  the 2D Einstein relation

**Protein/peptide–membrane interactions** (driven by the auto-detected,
user-overridable Protein and Membrane groups):

- **Minimum distance (ξ)** with contact-percentage distribution
- **Penetration depth (ΔZ)** vs the bilayer midplane, with membrane
  reference lines and a per-residue membrane-position profile
- **Binding residence (τ-off)** — dual-cutoff (4.5/7.0 Å hysteresis)
  contacts, survival-function analysis with mono/bi-exponential maximum
  likelihood fits, AIC model selection and bootstrap confidence intervals;
  both whole-peptide residence and per-lipid pooled residence
  (PyLipID-style), plus a per-residue residence table and plot. The
  anchor moiety is selectable (basic Lys/Arg, hydrophobic, aromatic,
  all heavy atoms, or a custom selection) with the membrane partner
  switching automatically between phosphates and whole-lipid atoms
- **Binding sites (V2)** — residue-community clustering of the
  residue–lipid co-contact network (Louvain) into distinct binding
  sites, each with occupancy, lipid composition, site-resolved τ-off
  with confidence interval, co-contact heat map, and representative
  bound-pose PDBs
- **Tilt / FEL / States** — tilt time series, tilt–ΔZ free-energy landscape,
  detection of well-separated states with populations, and export of one
  representative PDB structure per state
- **Lipid enrichment & contacts** — shell/bulk enrichment index per lipid
  species and per-residue membrane contact fractions

## Required trajectory preprocessing

Analyze trajectories in which molecules are whole and the membrane is
centered:

```bash
gmx trjconv -s a.tpr -f a.xtc -pbc mol -center -ur compact -o b.xtc
```

(optionally followed by a rot+trans fit for tilt/FEL work). GMSimA screens
the first, middle and last frames for broken molecules, unwrapped
("nojump"-style) coordinates, membrane misalignment and off-center systems —
warnings are reported in the log, status bar and HTML report, and the
analysis always proceeds.

## Installation (from source)

```bash
python -m venv venv && source venv/bin/activate
pip install .
GMSimA            # or: python main.py
```

Requires Python ≥ 3.9. Dependencies (installed automatically): numpy, scipy,
pandas, matplotlib, MDAnalysis ≥ 2.4, PySide6.

## Usage

1. Load a topology and the trajectory (`.xtc`). The topology must contain
   exactly the atoms in the XTC (e.g. the dry system for a dry XTC):
   * `.tpr` — preferred; exact bonds, masses and charges.
   * `.gro` / `.pdb` — any GROMACS version, no GROMACS install needed.
     Bonds are guessed from distances and cleaned (no cross-residue H
     bonds, only consecutive-residue links, disulfides added); masses are
     guessed from atom names; charges are unavailable (no current module
     needs them). Use this for `.tpr` files newer than the installed
     MDAnalysis (e.g. GROMACS 2026 with MDAnalysis ≤ 2.10):
     `gmx editconf -f system.tpr -o system.gro`.
     Optionally `pip install TprParser` lets such `.tpr` files load directly.
2. The Protein/Membrane group selectors auto-populate from the topology;
   any MDAnalysis selection string can be typed instead.
3. Pick a stride, choose analyses, choose an output folder, **Run Analysis**.
4. Results land in a timestamped subfolder together with `GMSimA.log` and
   `report.html`. The **Stop** button halts the run at the next module
   boundary and keeps everything already computed.

Supporting a new lipid species is one dictionary entry in `core/lipids.py`
(CHARMM36 naming: C2x = sn-2 chain, C3x = sn-1 chain).

**Force fields (V2):** CHARMM36(m) one-residue lipids and the AMBER
Lipid14/17/21 family (three-residue lipids: sn-1 chain, head, sn-2 chain)
are both supported and auto-detected; AMBER acyl chains are identified by
bond connectivity, so no atom-name tables are needed.

**Multiple peptides (V2):** if the protein group contains several bonded
molecules, every interaction analysis runs once per peptide with suffixed
outputs (`deltaz_pep1.xvg`, `tauoff_pep2_residence.txt`, ...).
