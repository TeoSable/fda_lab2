"""Prepare RLC, dispersion and redundant-offset regression experiments."""

import json
from dataclasses import dataclass, replace

import numpy as np

from .data import load_material, refractive_index
from .intervention import Regression, rlc_current
from .lab2_config import (
    MATERIAL_AMPLITUDE_BOUNDS,
    MATERIAL_NOISE_SCALE,
    MATERIAL_SEED_OFFSET,
    MATERIAL_SPECS,
    OFFSET_BOUNDS,
    OFFSET_INITIAL,
    OFFSET_PARAMETER_NAME,
    OFFSET_TRUE,
    RLC_BOUNDS,
    RLC_INITIAL,
    RLC_PARAMETER_NAMES,
    ZERO_RESISTANCE_OHM,
)


@dataclass
class Dataset:
    """Regression, validation data and structural ablation values."""

    name: str
    model: Regression
    validation_x: np.ndarray
    validation_y: np.ndarray
    tolerance: float
    ablations: dict
    metadata: dict


@dataclass(frozen=True)
class SellmeierPredictor:
    """Published resonances with fitted amplitudes and an optional offset."""

    resonances: np.ndarray
    include_offset: bool = False

    def design(self, wavelengths):
        squared = np.asarray(wavelengths)[:, None] ** 2
        matrix = squared / (squared - self.resonances)
        if self.include_offset:
            matrix = np.column_stack([matrix, np.ones(len(wavelengths))])
        return matrix

    def __call__(self, wavelengths, theta):
        return self.design(wavelengths) @ theta


def _rlc_dataset(root, config):
    data = np.genfromtxt(
        root / "results/synthetic_transient.csv", delimiter=",", names=True
    )
    train = data["is_train"].astype(bool)
    metrics = json.loads((root / "results/metrics.json").read_text())
    sigma = metrics["coupling_fit"]["noise_std_A"]
    model = Regression(
        x=data["time_s"][train],
        y=data["noisy_i2_A"][train],
        scale=np.full(train.sum(), sigma),
        names=RLC_PARAMETER_NAMES,
        predict=rlc_current,
        initial=np.asarray(RLC_INITIAL),
        bounds=RLC_BOUNDS,
    )
    return Dataset(
        name="rlc",
        model=model,
        validation_x=data["time_s"][~train],
        validation_y=data["true_i2_A"][~train],
        tolerance=config.rlc_relative_tolerance
        * np.max(abs(data["true_i2_A"])),
        ablations={0: 0.0, 1: ZERO_RESISTANCE_OHM},
        metadata={"kind": "synthetic current from lab1", "scale_A": sigma},
    )


def _material_dataset(root, config, name, limits, index):
    document, provenance = load_material(name, root / "data/raw")
    provenance["cache_mtime_utc"] = provenance.pop("retrieved_at_utc")
    manifest = json.loads((root / "data/provenance.json").read_text())
    provenance["upstream_record"] = next(
        record for record in manifest if record["name"] == name
    )
    formula = document["DATA"][0]
    coefficients = np.array(
        [float(v) for v in formula["coefficients"].split()]
    )
    power = 2 if formula["type"] == "formula 1" else 1
    predictor = SellmeierPredictor(coefficients[2::2] ** power)
    wavelengths = np.linspace(*limits, config.material_train_points)
    truth = refractive_index(document, wavelengths) ** 2 - 1
    scale = MATERIAL_NOISE_SCALE * (1 + wavelengths / wavelengths.max())
    generator = np.random.default_rng(
        config.seed + MATERIAL_SEED_OFFSET + index
    )
    observations = truth + generator.normal(size=len(wavelengths)) * scale
    amplitudes = coefficients[1::2]
    count = len(amplitudes)
    model = Regression(
        x=wavelengths,
        y=observations,
        scale=scale,
        names=tuple(f"a{j + 1}" for j in range(count)),
        predict=predictor,
        initial=amplitudes.copy(),
        bounds=(
            np.full(count, MATERIAL_AMPLITUDE_BOUNDS[0]),
            np.full(count, MATERIAL_AMPLITUDE_BOUNDS[1]),
        ),
    )
    validation_x = np.linspace(*limits, config.material_validation_points)
    metadata = dict(
        provenance,
        imposed_error="2e-5*(1+lambda/lambda_max) in n^2-1",
        reference_coefficients=amplitudes.tolist(),
        design_condition_number=float(
            np.linalg.cond(predictor.design(wavelengths) / scale[:, None])
        ),
    )
    return Dataset(
        name=name,
        model=model,
        validation_x=validation_x,
        validation_y=refractive_index(document, validation_x) ** 2 - 1,
        tolerance=config.material_tolerance,
        ablations=dict.fromkeys(range(count), 0.0),
        metadata=metadata,
    )


def _offset_dataset(silica):
    """Use the same silica observations with an extra fitted constant term."""
    original = silica.model
    predictor = replace(original.predict, include_offset=True)
    model = replace(
        original,
        predict=predictor,
        names=(*original.names, OFFSET_PARAMETER_NAME),
        initial=np.r_[original.initial, OFFSET_INITIAL],
        bounds=(
            np.r_[original.bounds[0], OFFSET_BOUNDS[0]],
            np.r_[original.bounds[1], OFFSET_BOUNDS[1]],
        ),
    )
    return replace(
        silica,
        name="silica_offset",
        model=model,
        ablations={**silica.ablations, len(original.names): 0.0},
        metadata=dict(
            silica.metadata,
            kind="synthetic redundant constant added to silica model",
            offset_true=OFFSET_TRUE,
            design_condition_number=float(
                np.linalg.cond(
                    predictor.design(model.x) / model.scale[:, None]
                )
            ),
        ),
    )


def prepare_datasets(root, config):
    """Return experiments, including a redundant-parameter example."""
    datasets = {"rlc": _rlc_dataset(root, config)}
    for index, (name, limits) in enumerate(MATERIAL_SPECS):
        datasets[name] = _material_dataset(root, config, name, limits, index)
    datasets["silica_offset"] = _offset_dataset(datasets["silica"])
    return datasets
