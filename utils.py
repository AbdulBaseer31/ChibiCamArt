import sys
import os

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    When bundled via PyInstaller, it sets sys._MEIPASS to the temp extraction dir.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
