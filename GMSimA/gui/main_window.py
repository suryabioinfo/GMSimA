from PySide6.QtWidgets import (
    QLabel, QPushButton, QLineEdit, QToolBar, QTextEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox,
    QGroupBox, QCheckBox, QProgressBar, QMainWindow, QMessageBox,
    QTabWidget, QComboBox   # <------- NEWDEF
)
from PySide6.QtGui import QIcon, QAction, Qt
from PySide6.QtCore import QSize
from workers.analysis_worker import AnalysisWorker
from core.trajectory_loader import load_trajectory_metadata
import os, subprocess, platform


class TrajectoryInfoDialog(QMessageBox):
    def __init__(self, info_text):
        super().__init__()
        self.setWindowTitle("System / Trajectory Information")
        self.setText(info_text)
        self.setIcon(QMessageBox.Information)
        self.setStandardButtons(QMessageBox.Ok)
        self.setMinimumWidth(750)


class MainWindow(QMainWindow):

    # v3.3 tilt: GUI label -> analysis/orientation.py mode name.
    _AXIS_MODES = {
        "Auto (from topology)": "auto",
        "Helix axis": "helix",
        "TM segments": "segments",
        "Macrocycle normal": "ring",
        "Principal axis": "principal",
        "End-to-end (legacy)": "endtoend",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GMSimA")
        self.setMinimumWidth(650)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        icon_path = os.path.join(base_dir, "assets", "icon.ico")
        self.setWindowIcon(QIcon(icon_path))

        self.worker = None
        self.last_output_dir = None   # <------- NEWDEF

        # -------- Toolbar (UNCHANGED) --------
        self.create_toolbar()

        # ===========================
        # NEW: HEADER + TABS SYSTEM
        # ===========================
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # -------- Header Buttons --------
        header = QWidget()
        header.setObjectName("HeaderBar")   # <------- NEWDEF

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)

        self.btn_membrane = QPushButton("Membrane Analysis")
        self.btn_about = QPushButton("About")

        for b in [self.btn_membrane, self.btn_about]:
            b.setCheckable(True)
            b.setObjectName("TabButton")   # <------- NEWDEF

        header_layout.addWidget(self.btn_membrane)
        header_layout.addWidget(self.btn_about)
        header_layout.addStretch()

        # -------- Tabs Container --------
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()

        self.tabs.addTab(self.create_membrane_tab(), "")
        self.tabs.addTab(self.create_about_tab(), "")

        root_layout.addWidget(header)
        root_layout.addWidget(self.tabs)

        # -------- Button Logic --------
        self.btn_membrane.clicked.connect(lambda: self.switch_tab(0))
        self.btn_about.clicked.connect(lambda: self.switch_tab(1))

        self.btn_membrane.setChecked(True)
        self.tabs.setCurrentIndex(0)

        # -------- Load QSS --------
        self.load_styles()   # <------- NEWDEF

    # ===========================
    # STYLE LOADER
    # ===========================
    def load_styles(self):   # <------- NEWDEF
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            qss_path = os.path.join(base_dir, "gui", "styles.qss")
            if os.path.exists(qss_path):
                with open(qss_path, "r") as f:
                     qss = f.read()
                     assets = os.path.join(base_dir, "assets").replace("\\", "/")
                     self.setStyleSheet(qss.replace("@ASSETS@", assets))
        except Exception as e:
            print("Style load failed:", e)

    # ===========================
    # TAB SWITCH
    # ===========================
    def switch_tab(self, index):   # <------- NEWDEF
        self.tabs.setCurrentIndex(index)
        self.btn_membrane.setChecked(index == 0)
        self.btn_about.setChecked(index == 1)

    # ===========================
    # MEMBRANE TAB (UNCHANGED LOGIC)
    # ===========================
    def create_membrane_tab(self):   # <------- NEWDEF

        widget = QWidget()
        # V3.0.0: the stylesheet scopes the professional theme to THIS
        # tab only (QWidget#MembraneTab ... rules); the About tab keeps
        # the default appearance.
        widget.setObjectName("MembraneTab")
        layout = QVBoxLayout(widget)

        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        input_group = self.create_input_group()
        membrane_group = self.create_membrane_group()
        interaction_group = self.create_interaction_group()

        self.run_button = QPushButton("Run Analysis")
        self.run_button.setObjectName("runButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_analysis)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Ready")

        layout.addWidget(input_group)
        layout.addWidget(membrane_group)
        layout.addWidget(interaction_group)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        return widget

    # ===========================
    # ABOUT TAB
    # ===========================
    def create_about_tab(self):   # <------- NEWDEF
        # Content lives in gui/about.py — edit that file to update this tab.
        from gui.about import build_about_widget
        return build_about_widget()

    # ===========================
    # ORIGINAL FUNCTIONS BELOW (UNCHANGED)
    # ===========================

    def create_input_group(self):
        group = QGroupBox("Input/Output")
        layout = QVBoxLayout()

        tpr_layout = QHBoxLayout()
        self.tpr_edit = QLineEdit()
        self.tpr_edit.setPlaceholderText(".tpr (exact) or .gro / .pdb (bonds guessed)")
        tpr_button = QPushButton("Browse")
        tpr_button.clicked.connect(self.browse_tpr)
        tpr_layout.addWidget(QLabel("Load Topology"))
        tpr_layout.addWidget(self.tpr_edit)
        tpr_layout.addWidget(tpr_button)

        xtc_layout = QHBoxLayout()
        self.xtc_edit = QLineEdit()
        xtc_button = QPushButton("Browse")
        xtc_button.clicked.connect(self.browse_xtc)
        xtc_layout.addWidget(QLabel("Load XTC"))
        xtc_layout.addWidget(self.xtc_edit)
        xtc_layout.addWidget(xtc_button)

        stride_layout = QHBoxLayout()
        self.stride_spin = QSpinBox()
        self.stride_spin.setMinimum(1)
        self.stride_spin.setMaximum(1000)
        self.stride_spin.setValue(1)
        stride_layout.addWidget(QLabel("Trajectory Stride (frames)"))
        stride_layout.addWidget(self.stride_spin)

        # V3.1: multi-core execution (module-level multiprocessing).
        # The user picks HOW MANY processes to use; 1 = sequential.
        import os as _os
        _max_cores = max(1, (_os.cpu_count() or 2) - 1)
        self.cores_spin = QSpinBox()
        self.cores_spin.setMinimum(1)
        self.cores_spin.setMaximum(_max_cores)
        self.cores_spin.setValue(_max_cores)
        self.cores_spin.setToolTip(
            "Number of CPU for the analysis modules\n"
            f"(1–{_max_cores}; this machine has {_os.cpu_count()} cores.\n")
        stride_layout.addWidget(QLabel(f"CPU cores (max {_max_cores})"))
        stride_layout.addWidget(self.cores_spin)

        # ---- Group selectors (auto-populated from the topology) ----
        # Editable: any MDAnalysis selection string can be typed in.
        prot_layout = QHBoxLayout()
        self.protein_combo = QComboBox()
        self.protein_combo.setEditable(True)
        self.protein_combo.addItem("auto")
        self.protein_combo.setToolTip(
            "Protein/peptide group used by the interaction analyses.\n"
            "Auto-detected from the topology; any MDAnalysis selection\n"
            "string can also be typed (e.g. 'protein and segid PROA').")
        prot_layout.addWidget(QLabel("Protein Group"))
        prot_layout.addWidget(self.protein_combo, 1)

        memb_layout = QHBoxLayout()
        self.membrane_combo = QComboBox()
        self.membrane_combo.setEditable(True)
        self.membrane_combo.addItem("auto")
        self.membrane_combo.setToolTip(
            "Membrane group used by the interaction analyses.\n"
            "Auto-detected lipid species from the topology; any MDAnalysis\n"
            "selection string can also be typed (e.g. 'resname POPC POPS').")
        memb_layout.addWidget(QLabel("Membrane Group"))
        memb_layout.addWidget(self.membrane_combo, 1)

        out_layout = QHBoxLayout()
        self.out_edit = QLineEdit()
        out_button = QPushButton("Browse")
        out_button.clicked.connect(self.browse_output)
        out_layout.addWidget(QLabel("Output Location"))
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_button)

        layout.addLayout(tpr_layout)
        layout.addLayout(xtc_layout)
        layout.addLayout(stride_layout)
        layout.addLayout(prot_layout)
        layout.addLayout(memb_layout)
        layout.addLayout(out_layout)

        group.setLayout(layout)
        return group

    def validate_inputs(self):
        if self.tpr_edit.text() and self.xtc_edit.text():
            self.run_button.setEnabled(True)
        else:
            self.run_button.setEnabled(False)

    def create_membrane_group(self):
        group = QGroupBox("Intrinsic Membrane Properties")
        layout = QVBoxLayout()
        self.apl = QCheckBox("Area Per Lipid (APL)")
        self.thickness = QCheckBox("Membrane Thickness (d_PP)")
        self.order = QCheckBox("Order Parameter (S_CD)")
        self.curvature = QCheckBox("Curvature, Density \u0026 Diffusion (H, \u03c1, D)")
        layout.addWidget(self.apl)
        layout.addWidget(self.thickness)
        layout.addWidget(self.order)
        layout.addWidget(self.curvature)
        group.setLayout(layout)
        return group

    def create_interaction_group(self):
        group = QGroupBox("Interactions Properties")
        layout = QVBoxLayout()
        self.distance = QCheckBox("Protein\u2013Membrane Distance (\u03be)")
        self.tauoff = QCheckBox("Binding Residence (\u03c4-off)")
        self.deltaz = QCheckBox("Penetration Depth (\u0394z)")
        self.tilt = QCheckBox("Tilt && Free-Energy Landscape (\u03b8, FEL)")
        self.enrichment = QCheckBox("Lipid Enrichment && Contacts (E)")
        self.binding_sites = QCheckBox("Binding Sites (site-resolved \u03c4-off)")
        layout.addWidget(self.distance)
        layout.addWidget(self.tauoff)
        layout.addWidget(self.deltaz)
        layout.addWidget(self.tilt)

        # ---- v3.3: how the molecular axis for the tilt is defined ----
        # Sub-option of the tilt checkbox, so it is indented and greys
        # out when tilt is off. See analysis/orientation.py.
        axis_layout = QHBoxLayout()
        axis_layout.setContentsMargins(22, 0, 0, 0)
        self.axis_label = QLabel("Tilt axis")
        self.axis_mode = QComboBox()
        self.axis_mode.addItems(list(self._AXIS_MODES))
        self.axis_mode.setToolTip(
            "How the molecular axis used for the tilt angle is defined.\n"
            "Auto: helix fit for a helical peptide, macrocycle plane\n"
            "normal for a cyclic peptide, per-segment axes for a multi-\n"
            "pass protein. End-to-end reproduces v2/v3 numbers.")
        self.tm_segments = QLineEdit()
        self.tm_segments.setPlaceholderText(
            "TM segments by resid, e.g. 6-33, 41-68  (blank = auto-detect)")
        self.tm_segments.setToolTip(
            "Optional explicit transmembrane/helical segments as resid\n"
            "ranges. Leave blank to detect them from the backbone\n"
            "geometry. Used by the 'TM segments' mode.")
        axis_layout.addWidget(self.axis_label)
        axis_layout.addWidget(self.axis_mode, 1)
        axis_layout.addWidget(self.tm_segments, 2)
        layout.addLayout(axis_layout)

        # enable the axis row only while tilt is selected
        for _w in (self.axis_label, self.axis_mode, self.tm_segments):
            _w.setEnabled(self.tilt.isChecked())
            self.tilt.toggled.connect(_w.setEnabled)

        layout.addWidget(self.enrichment)
        layout.addWidget(self.binding_sites)

        # ---- V3 interaction matrix: anchor x partner x species ----
        # Presets come from the EDITABLE config/interactions.py; each
        # combo also accepts a typed MDAnalysis selection.
        try:
            from config.interactions import ANCHOR_PRESETS, PARTNER_KINDS
            anchor_items = list(ANCHOR_PRESETS)
            partner_items = list(PARTNER_KINDS)
        except Exception:
            anchor_items = ["Basic (Lys/Arg)", "All heavy atoms",
                            "Hydrophobic side chains", "Aromatic rings"]
            partner_items = ["Auto"]

        anchor_layout = QHBoxLayout()
        self.anchor_combo = QComboBox()
        self.anchor_combo.setEditable(True)
        self.anchor_combo.addItems(anchor_items)
        self.anchor_combo.setToolTip(
            "PEPTIDE side of the interaction pairing: which atoms of\n"
            "the peptide count as the binding moiety in the residence\n"
            "(τ-off) and binding-site analyses.")
        anchor_layout.addWidget(QLabel("Anchor atoms"))
        anchor_layout.addWidget(self.anchor_combo, 1)
        layout.addLayout(anchor_layout)

        partner_layout = QHBoxLayout()
        self.partner_combo = QComboBox()
        self.partner_combo.setEditable(True)
        self.partner_combo.addItems(partner_items)
        self.partner_combo.setToolTip(
            "MEMBRANE side of the pairing: which lipid atoms the anchor\n"
            "binds to. 'Auto' picks the natural partner for the chosen\n"
            "preset (phosphates for basic anchors, whole lipids\n"
            "otherwise).")
        partner_layout.addWidget(QLabel("Membrane partner"))
        partner_layout.addWidget(self.partner_combo, 1)
        layout.addLayout(partner_layout)

        species_layout = QHBoxLayout()
        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.addItem("All species")
        self.species_combo.setToolTip(
            "Restrict the membrane partner to one lipid species\n"
            "(populated from the loaded topology). 'All species' will\n" 
            "use every lipid.")
        species_layout.addWidget(QLabel("Lipid species"))
        species_layout.addWidget(self.species_combo, 1)
        layout.addLayout(species_layout)

        group.setLayout(layout)
        return group

    def browse_tpr(self):
        from core.universe_loader import TOPOLOGY_FILE_FILTER
        file, _ = QFileDialog.getOpenFileName(self, "Load Topology (TPR / GRO / PDB)", "", TOPOLOGY_FILE_FILTER)
        if file:
            self.tpr_edit.setText(file)
            self.populate_group_selectors(file)
            self.update_traj_info()
            self.validate_inputs()

    def populate_group_selectors(self, tpr_file):
        """
        Fill the Protein/Membrane group dropdowns from the topology.
        Never blocks the user: on any failure the combos stay editable
        with their current text.
        """
        try:
            from core.lipids import suggest_groups
            from core.universe_loader import load_universe, UnsupportedTPRError
            u = load_universe(tpr_file)
            groups = suggest_groups(u)
        except UnsupportedTPRError as e:
            QMessageBox.warning(self, "Unsupported TPR version", str(e))
            return
        except Exception as e:
            print(f"Group auto-detection failed: {e}")
            return

        self.protein_combo.clear()
        if groups["protein_options"]:
            self.protein_combo.addItems(groups["protein_options"])
        else:
            self.protein_combo.addItem("auto")
            self.protein_combo.setToolTip(
                "No protein detected in this topology — interaction analyses "
                "will be skipped with a warning.")

        self.membrane_combo.clear()
        if groups["membrane_options"]:
            self.membrane_combo.addItems(groups["membrane_options"])
        else:
            self.membrane_combo.addItem("auto")

        # V3: species filter options from the detected lipid inventory
        if hasattr(self, "species_combo"):
            self.species_combo.clear()
            self.species_combo.addItem("All species")
            for rn in sorted(groups.get("lipids", {})):
                self.species_combo.addItem(rn)

        if groups["unknown_lipids"]:
            QMessageBox.warning(
                self, "Unknown lipids",
                "These residues look lipid-like but are not in the lipid "
                f"registry: {', '.join(groups['unknown_lipids'])}.\n\n"
                "They are excluded from lipid-aware analyses; add them to "
                "core/lipids.py or type a custom selection to include them.")

    def browse_xtc(self):
        file, _ = QFileDialog.getOpenFileName(self, "Load XTC File", "", "Trajectory (*.xtc)")
        if file:
            self.xtc_edit.setText(file)
            self.update_traj_info()
            self.validate_inputs()

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if folder:
            self.out_edit.setText(folder)

    def run_analysis(self):
        tpr = self.tpr_edit.text()
        xtc = self.xtc_edit.text()
        stride = self.stride_spin.value()
        out = self.out_edit.text()

        if not tpr or not xtc:
            QMessageBox.critical(self, "Missing Input", "Provide a topology (TPR/GRO/PDB) and an XTC")
            return

        options = {
            "apl": self.apl.isChecked(),
            "thickness": self.thickness.isChecked(),
            "order": self.order.isChecked(),
            "curvature": self.curvature.isChecked(),
            "distance": self.distance.isChecked(),
            "tauoff": self.tauoff.isChecked(),
            "deltaz": self.deltaz.isChecked(),
            "tilt": self.tilt.isChecked(),
            "enrichment": self.enrichment.isChecked(),
            "binding_sites": self.binding_sites.isChecked()
        }

        if not any(options.values()):
            QMessageBox.warning(self, "No analysis", "Select at least one analysis")
            return

        # "auto" (or blank) lets the worker resolve the group at run time.
        protein_sel = self.protein_combo.currentText().strip()
        membrane_sel = self.membrane_combo.currentText().strip()

        self.worker = AnalysisWorker(tpr, xtc, out, options, stride,
                                     protein_sel=protein_sel,
                                     membrane_sel=membrane_sel,
                                     anchor_choice=self.anchor_combo
                                     .currentText().strip(),
                                     partner_choice=self.partner_combo
                                     .currentText().strip(),
                                     species_choice=self.species_combo
                                     .currentText().strip(),
                                     parallel=self.cores_spin
                                     .value() > 1,
                                     n_cores=self.cores_spin.value(),
                                     axis_mode=self._AXIS_MODES.get(
                                         self.axis_mode.currentText(),
                                         "auto"),
                                     tm_segments=self.tm_segments
                                     .text().strip())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.show_error_popup)
        self.worker.start()

    def on_finished(self, output_dir):
        self.last_output_dir = output_dir
        self.status_label.setText("Finished")

    # ----------- Toolbar with placeholders ----------
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24,24))
        # Text beside icon so actions stay visible even without icon files
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        icon_dir = os.path.join(base_dir, "assets", "icons")

        # Run analysis
        # run_icon = QIcon(os.path.join(icon_dir, "run.svg"))
        # run_action = QAction(run_icon, "Run", self)  
        # run_action.setToolTip("Run Analysis")
        # run_action.triggered.connect(self.run_analysis)
 
        # # Stop analysis
        # stop_icon = QIcon(os.path.join(icon_dir, "stop.svg"))   
        # stop_action = QAction(stop_icon, "Stop", self)  
        # stop_action.setToolTip("Stop Analysis")  
        # stop_action.triggered.connect(self.stop_analysis)

        # stop = QAction(QIcon(os.path.join(icon_dir, "stop.svg")), "Stop", self)
        # stop.triggered.connect(self.stop_analysis)

        # Stop analysis
        stop_icon_path = os.path.join(icon_dir, "stop.svg")
        stop_icon = QIcon(stop_icon_path) if os.path.exists(stop_icon_path) else QIcon()
        stop_action = QAction(stop_icon, "Stop", self)
        stop_action.setToolTip("Stop the running analysis "
                               "(takes effect at the next module boundary)")
        stop_action.triggered.connect(self.stop_analysis)

        # Export results
        exp_icon_path = os.path.join(icon_dir, "results.png")
        export_icon = QIcon(exp_icon_path) if os.path.exists(exp_icon_path) else QIcon()
        export_action = QAction(export_icon, "Results", self)
        export_action.setToolTip("Results")
        export_action.triggered.connect(self.export_results)

        # export = QAction(QIcon(os.path.join(icon_dir, "export.svg")), "Export", self)
        # export.triggered.connect(self.export_results)

        toolbar.addAction(stop_action)
        toolbar.addSeparator()
        toolbar.addAction(export_action)

        self.addToolBar(toolbar)

    def stop_analysis(self):
        if self.worker:
            self.worker.cancel()

    def export_results(self):
        path = getattr(self, "last_output_dir", None)
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "No Data", "No Completed Analysis")
            return

        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def show_error_popup(self, message):
        QMessageBox.critical(self, "Error", message)

    # --------------- Clean shutdown ---------------------
    def closeEvent(self, event):
        """
        Stop a running analysis thread before the window is destroyed.
        Destroying a QThread while it is still running is undefined
        behaviour in Qt and segfaults on exit.
        """
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(10000)   # up to 10 s for a graceful stop
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
        super().closeEvent(event)

   # --------------- Trjectory Details ---------------------
    def update_traj_info(self):
        tpr = self.tpr_edit.text()
        xtc = self.xtc_edit.text()
        if not tpr or not xtc:
            return
        text = "Trajectory Information\n\n"
        if not tpr or not xtc:
            if tpr:
                text += f"Topology: {os.path.basename(tpr)}\n"
            if xtc:
                text += f"XTC: {os.path.basename(xtc)}\n"
            #self.traj_info.setText(text)
            dialog = TrajectoryInfoDialog(text)
            dialog.exec()
            return
        # ---------------- GUI Panel: Load metadata ----------------
        try:
            metadata = load_trajectory_metadata(tpr, xtc)
        except Exception as e:

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Trajectory Error")
            msg.setText(f"Error reading trajectory:\n{str(e)}")
            msg.exec()
            #self.traj_info.setText(f"Error reading trajectory:\n{str(e)}")
            return

        text += f"Topology: {metadata['tpr']}\n"
        text += f"XTC: {metadata['xtc']}\n"
        text += f"Atoms: {metadata['atoms']}\n"
        from core.universe_loader import BACKEND_LABELS
        text += f"Topology read as: {BACKEND_LABELS.get(metadata.get('tpr_reader'), '?')}\n"
        text += "\n"
        
        text += f"Frames: {metadata['frames']}\n"
        text += f"Timestep: {metadata['timestep_ps']} ps\n"
        text += f"Simulation length: {metadata['sim_length_ns']} ns\n"
        box = metadata["box"]
        text += f"Box: {box[0]} x {box[1]} x {box[2]} nm\n\n"

        text += f"Forcefield: {metadata['forcefield']}\n"
        text += "\nSystem Details:\nLipids composition\n"
        for lipid,count in metadata["lipids"].items():
            text += f"    {lipid}: {count}\n"
        #text += "\n"
        text += f" Upper leaflet lipids: {metadata['upper_leaflet_lipids']}\n"
        text += f" Lower leaflet lipids: {metadata['lower_leaflet_lipids']}\n"
        text += " Cholesterol distribution:\n"
        text += f"    Upper leaflet CHOL: {metadata['upper_chol']}\n"
        text += f"    Lower leaflet CHOL: {metadata['lower_chol']}\n"

        if metadata["protein_present"]:
            text += f" Protein atoms: {metadata['protein_atoms']}\n"
        else:
            text += "No protein detected\n"
        #self.traj_info.setText(text)
        dialog = TrajectoryInfoDialog(text)
        dialog.exec()