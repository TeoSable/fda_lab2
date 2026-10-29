"""Project-local runtime directories configured before plotting imports."""

import os
from pathlib import Path

MATPLOTLIB_CACHE_PATH = Path("tmp") / "matplotlib"


def configure_matplotlib_cache(root):
    """Use a project directory for Matplotlib settings and font cache."""
    directory = (Path(root) / MATPLOTLIB_CACHE_PATH).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(directory)
    return directory
