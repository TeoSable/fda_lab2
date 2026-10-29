"""Physical constants and configuration for laboratory 2."""

from dataclasses import dataclass

KERNEL_NAME = "emlab-lab2"
KERNEL_DISPLAY_NAME = "Python (EMLab — lab2)"
NOTEBOOK_PATH = "notebooks/lab2_interventions.ipynb"
OUTPUT_PATH = "results/lab2"
CACHE_VERSION = 2
L1_HENRY = 1e-6
L2_HENRY = 0.5e-6
R1_OHM = 2.0
FREQUENCY_HZ = 13.56e6
NANOHENRY_TO_HENRY = 1e-9
RLC_PARAMETER_NAMES = ("M [nH]", "R2 [ohm]")
RLC_INITIAL = (12.0, 6.0)
RLC_BOUNDS = ((0.1, 0.5), (100.0, 15.0))
ZERO_RESISTANCE_OHM = 1e-6
FIT_TOLERANCE = 1e-8
FIT_MAX_EVALUATIONS = 150
ROBUST_LOSS_SCALE = 1.0
BOUNDARY_ABSOLUTE_TOLERANCE = 1e-5
BOUNDARY_RELATIVE_TOLERANCE = 1e-5
MATERIAL_AMPLITUDE_BOUNDS = (0.0, 3.0)
OFFSET_BOUNDS = (-0.1, 0.1)
OFFSET_INITIAL = 0.0
OFFSET_TRUE = 0.0
OFFSET_PARAMETER_NAME = "c0"
MATERIAL_NOISE_SCALE = 2e-5
MATERIAL_SEED_OFFSET = 10
BOOTSTRAP_SEED_OFFSET = 100
INTERVENTION_SEED_OFFSET = 200
DECISION_LOSS = "linear"
MATERIAL_SPECS = (("silica", (0.25, 3.5)), ("water", (0.2, 1.1)))


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by data preparation, bootstrap, plots and decisions."""

    seed: int = 20261029
    bootstrap_replicates: int = 300
    intervention_repeats: int = 5
    levels: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    losses: tuple[str, ...] = ("linear", "soft_l1")
    example_alpha: float = 2.0
    example_repeat: int = 0
    confidence_quantiles: tuple[float, float] = (0.025, 0.975)
    material_train_points: int = 160
    material_validation_points: int = 81
    rlc_relative_tolerance: float = 0.05
    material_tolerance: float = 2e-4
    max_boundary_fraction: float = 0.05
    notebook_timeout_seconds: int = 7200

    def __post_init__(self):
        if self.bootstrap_replicates < 2 or self.intervention_repeats < 1:
            raise ValueError("Use at least two bootstrap draws and one repeat")
        if 0.0 not in self.levels or self.example_alpha not in self.levels:
            raise ValueError("Levels must include zero and example_alpha")
        if not 0 <= self.example_repeat < self.intervention_repeats:
            raise ValueError(
                "example_repeat is outside the configured repeats"
            )
        if any(level < 0 for level in self.levels):
            raise ValueError("Intervention levels must be nonnegative")
        if DECISION_LOSS not in self.losses:
            raise ValueError("The decision loss must be included in losses")


DEFAULT_CONFIG = ExperimentConfig()
