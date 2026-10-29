"""Install the named Jupyter kernel inside the project virtual environment."""

from pathlib import Path

from ipykernel.kernelspec import install

from emlab.lab2_config import KERNEL_DISPLAY_NAME, KERNEL_NAME
from emlab.lab2_runtime import configure_matplotlib_cache

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = install(
        prefix=str(ROOT / ".venv"),
        kernel_name=KERNEL_NAME,
        display_name=KERNEL_DISPLAY_NAME,
        env={"MPLCONFIGDIR": str(configure_matplotlib_cache(ROOT))},
    )
    print(f"Installed {KERNEL_NAME}: {destination}")


if __name__ == "__main__":
    main()
