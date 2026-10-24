import sys, os
import multiprocessing

# ------------------------------------------------------------------
# CRITICAL: force the non-GUI matplotlib backend BEFORE anything else
# imports pyplot. With PySide6 installed, matplotlib auto-selects its
# Qt backend; the analysis worker then creates figures from a non-GUI
# QThread, which corrupts Qt state and segfaults on application exit.
# This line must stay ABOVE the gui/worker imports below.
# ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg", force=True)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from gui.splash_screen import SplashScreen
from gui.main_window import MainWindow



def main():

    # ---------- High DPI Support ------------
    app = QApplication(sys.argv)
    # ---------- Set Global Font --------------
    # Portable font stack: Qt picks the first family available on the
    # host OS (Segoe UI on Windows, Helvetica Neue on macOS, DejaVu/Arial
    # on Linux), so the GUI no longer depends on a Windows-only font.
    font = QFont()
    font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial",
                      "DejaVu Sans", "Sans Serif"])
    font.setPointSize(11)
    app.setFont(font)
    load_stylesheet(app)           # Load stylesheet

    # ---------- Splash Screen Loader ------------
    splash = SplashScreen()
    splash.show()

    # ---------- Main Window -----------------
    window = MainWindow()

    # Delay GUI show for splash Window
    QTimer.singleShot(5000, lambda: show_main(window, splash))

    exit_code = app.exec()

    # ---------------------------------------------------------
    # Orderly teardown: destroy Qt widgets while the QApplication
    # still exists. Leaving this to interpreter shutdown destroys
    # objects in arbitrary order and is a classic PySide6 cause of
    # "Segmentation fault (core dumped)" AFTER the window closes.
    # ---------------------------------------------------------
    del window
    del splash
    del app

    sys.exit(exit_code)


def show_main(window, splash):
    window.show()
    splash.finish(window)

def load_stylesheet(app):
    base_dir = os.path.dirname(__file__)
    style_path = os.path.join(base_dir, "gui", "styles.qss")

    with open(style_path, "r") as f:
        app.setStyleSheet(f.read())

if __name__ == "__main__":
    # V3.1 multi-core execution: required for the frozen (PyInstaller)
    # binary — without it every spawned analysis process re-launches
    # the whole GUI. Must be the first statement under __main__.
    multiprocessing.freeze_support()
    main()