"""RAG_OS desktop entry point.

The RAG pipeline runs in a separate process (backend/server.py); this launches
it if it isn't already up, then opens the Qt window as a thin HTTP client.
Nothing here imports torch — the UI process stays light and never freezes.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
from pathlib import Path

faulthandler.enable()   # dump a Python traceback on a hard crash (segfault etc.)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from rag import config  # noqa: E402


def _backend_up() -> bool:
    try:
        httpx.get(f"{config.BACKEND_URL}/health", timeout=1.5)
        return True
    except Exception:  # noqa: BLE001
        return False


def _spawn_backend() -> subprocess.Popen | None:
    """Start the backend and return immediately — the window shows right away
    and the controller polls /health for readiness."""
    if _backend_up():
        return None
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # + CREATE_NO_WINDOW
    return subprocess.Popen(
        [sys.executable, "-m", "backend.server"],
        cwd=str(REPO), env=dict(os.environ), creationflags=flags,
    )


def main():
    backend = _spawn_backend()

    from PySide6.QtWidgets import QApplication
    from desktop.main_window import MainWindow
    from desktop.controller import Controller

    app = QApplication(sys.argv)
    app.setApplicationName("RAG_OS")

    window = MainWindow()
    controller = Controller(window)

    def _cleanup():
        controller.shutdown()
        # leave the backend running so the next launch is instant; kill only
        # if we started it AND the env asks us to.
        if backend is not None and os.environ.get("RAG_KILL_BACKEND_ON_EXIT") == "1":
            backend.terminate()

    app.aboutToQuit.connect(_cleanup)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
