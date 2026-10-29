"""Execute the notebook using the project kernel and preserve cell outputs."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from emlab.lab2_config import DEFAULT_CONFIG, KERNEL_NAME, NOTEBOOK_PATH
from emlab.lab2_runtime import configure_matplotlib_cache

ROOT = Path(__file__).resolve().parents[1]


def report(cell, cell_index, **kwargs):
    print(f"Completed cell {cell_index}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-bootstrap",
        action="store_true",
        help="Use saved draws after verifying inputs and configuration",
    )
    arguments = parser.parse_args()
    configure_matplotlib_cache(ROOT)
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "tmp/jupyter")
    os.environ["IPYTHONDIR"] = str(ROOT / "tmp/ipython")
    os.environ["LAB2_REUSE_BOOTSTRAP"] = (
        "1" if arguments.reuse_bootstrap else "0"
    )
    Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    path = ROOT / NOTEBOOK_PATH
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(
            notebook,
            timeout=DEFAULT_CONFIG.notebook_timeout_seconds,
            resources={"metadata": {"path": str(ROOT)}},
            kernel_name=KERNEL_NAME,
            on_cell_executed=report,
        ).execute()
    finally:
        nbformat.write(notebook, path)
    print("Notebook executed successfully")


if __name__ == "__main__":
    main()
