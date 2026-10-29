"""Residual bootstrap, conditional response noise and marginal W1."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import wasserstein_distance

from .lab2_config import (
    BOUNDARY_ABSOLUTE_TOLERANCE,
    BOUNDARY_RELATIVE_TOLERANCE,
    FIT_MAX_EVALUATIONS,
    FIT_TOLERANCE,
    FREQUENCY_HZ,
    L1_HENRY,
    L2_HENRY,
    NANOHENRY_TO_HENRY,
    R1_OHM,
    ROBUST_LOSS_SCALE,
)


def rlc_current(times, theta):
    """Solve the zero-state RLC IVP; theta is (M in nH, R2 in ohm)."""
    times = np.asarray(times, dtype=float)
    mutual_inductance = float(theta[0]) * NANOHENRY_TO_HENRY
    resistance = float(theta[1])
    if (
        mutual_inductance < 0
        or resistance <= 0
        or mutual_inductance**2 >= L1_HENRY * L2_HENRY
    ):
        raise ValueError("Non-passive parameters")

    omega = 2 * np.pi * FREQUENCY_HZ
    capacitances = 1 / (omega**2 * np.array([L1_HENRY, L2_HENRY]))
    inverse_inductance = np.linalg.inv(
        [[L1_HENRY, mutual_inductance], [mutual_inductance, L2_HENRY]]
    )
    matrix = np.zeros((4, 4))
    matrix[:2, 2:] = np.diag(1 / capacitances)
    matrix[2:, :2] = -inverse_inductance
    matrix[2:, 2:] = -inverse_inductance @ np.diag([R1_OHM, resistance])
    forcing = np.r_[0.0, 0.0, inverse_inductance[:, 0]]
    particular = np.linalg.solve(1j * omega * np.eye(4) - matrix, forcing)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    weights = np.linalg.solve(eigenvectors, -particular.real)
    transient = (eigenvectors[3] * weights) @ np.exp(
        eigenvalues[:, None] * times
    )
    return (particular[3] * np.exp(1j * omega * times) + transient).real


@dataclass
class Regression:
    """Fixed-design regression with optional coefficient ablation."""

    x: np.ndarray
    y: np.ndarray
    scale: np.ndarray
    names: tuple
    predict: Callable
    initial: np.ndarray
    bounds: tuple

    def fit(self, y=None, loss="linear", fixed=None):
        """Return fitted coefficients and a boundary flag."""
        observations = self.y if y is None else np.asarray(y)
        free = np.ones(len(self.initial), dtype=bool)
        initial = self.initial.copy()
        for index, value in (fixed or {}).items():
            free[index] = False
            initial[index] = value

        def expand(parameters):
            theta = initial.copy()
            theta[free] = parameters
            return theta

        lower, upper = np.asarray(self.bounds)
        result = least_squares(
            lambda parameters: (
                (self.predict(self.x, expand(parameters)) - observations)
                / self.scale
            ),
            initial[free],
            bounds=(lower[free], upper[free]),
            loss=loss,
            f_scale=ROBUST_LOSS_SCALE,
            x_scale="jac",
            max_nfev=FIT_MAX_EVALUATIONS,
            ftol=FIT_TOLERANCE,
            xtol=FIT_TOLERANCE,
            gtol=FIT_TOLERANCE,
        )
        if not result.success:
            raise RuntimeError(result.message)
        theta = expand(result.x)
        boundary = any(
            np.any(
                np.isclose(
                    theta[free],
                    bound[free],
                    atol=BOUNDARY_ABSOLUTE_TOLERANCE,
                    rtol=BOUNDARY_RELATIVE_TOLERANCE,
                )
            )
            for bound in (lower, upper)
        )
        return theta, bool(boundary)


def residual_bootstrap(model, y, indices, loss="linear"):
    """Resample centered standardized residuals and refit at fixed inputs."""
    theta, boundary = model.fit(y, loss)
    mean = model.predict(model.x, theta)
    residual = (y - mean) / model.scale
    residual -= residual.mean()
    draws = []
    boundary_hits = 0
    for index in indices:
        sample = mean + model.scale * residual[index]
        estimate, hit = model.fit(sample, loss)
        draws.append(estimate)
        boundary_hits += hit
    return (
        np.asarray(draws),
        theta,
        {
            "boundary_fraction": boundary_hits / len(indices),
            "point_boundary": boundary,
        },
    )


def conditional_noise(y, scale, alpha, z):
    """Add alpha * scale(X) * Z to the response while keeping inputs fixed."""
    if alpha < 0 or np.any(np.asarray(scale) <= 0):
        raise ValueError("alpha >= 0 and positive conditional scales required")
    return np.asarray(y) + alpha * np.asarray(scale) * np.asarray(z)


def marginal_w1(before, after):
    """Return marginal W1 and W1 divided by the baseline standard deviation."""
    before, after = np.asarray(before), np.asarray(after)
    if (
        before.ndim != 2
        or after.ndim != 2
        or before.shape[1] != after.shape[1]
    ):
        raise ValueError("Expected B x parameter arrays")
    raw = np.array(
        [wasserstein_distance(a, b) for a, b in zip(before.T, after.T)]
    )
    standard_deviation = before.std(axis=0, ddof=1)
    scaled = np.divide(
        raw,
        standard_deviation,
        out=np.full_like(raw, np.nan),
        where=standard_deviation > 0,
    )
    return raw, scaled
