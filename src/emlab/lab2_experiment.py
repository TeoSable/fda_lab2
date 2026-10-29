"""Run and persist bootstrap experiments and structural ablation decisions."""

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from .intervention import conditional_noise, marginal_w1, residual_bootstrap
from .lab2_config import (
    BOOTSTRAP_SEED_OFFSET,
    CACHE_VERSION,
    DECISION_LOSS,
    INTERVENTION_SEED_OFFSET,
    OUTPUT_PATH,
)


@dataclass
class ExperimentResults:
    """Data consumed by reporting and plotting functions."""

    datasets: dict
    config: object
    output: Path
    arrays: dict
    records: list
    diagnostics: dict
    baselines: dict
    decisions: list

    def baseline_key(self, name, loss):
        return f"{name}_{loss}_baseline"

    def sample_key(self, name, loss, alpha, repeat):
        return f"{name}_{loss}_a{float(alpha)}_r{repeat}"

    def example_draws(self, name, loss):
        return self.arrays[
            self.sample_key(
                name,
                loss,
                self.config.example_alpha,
                self.config.example_repeat,
            )
        ]


def _write_json(path, content):
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fingerprint(datasets, config):
    digest = hashlib.sha256()
    settings = {
        "version": CACHE_VERSION,
        "config": asdict(config),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for filename in ("intervention.py", "lab2_data.py", "lab2_config.py"):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    for name, dataset in datasets.items():
        digest.update(name.encode())
        digest.update(json.dumps(dataset.model.names).encode())
        for values in (
            dataset.model.x,
            dataset.model.y,
            dataset.model.scale,
            dataset.model.initial,
            dataset.model.bounds,
            dataset.validation_x,
            dataset.validation_y,
        ):
            digest.update(np.asarray(values, dtype=float).tobytes())
    return digest.hexdigest()


def _record_condition(results, name, loss, alpha, repeat, draws, point, diag):
    model = results.datasets[name].model
    baseline = results.arrays[results.baseline_key(name, loss)]
    raw, scaled = marginal_w1(baseline, draws)
    intervals = np.quantile(draws, results.config.confidence_quantiles, axis=0)
    for index, parameter in enumerate(model.names):
        results.records.append(
            {
                "dataset": name,
                "loss": loss,
                "alpha": float(alpha),
                "repeat": repeat,
                "parameter": parameter,
                "estimate": float(point[index]),
                "ci_low": float(intervals[0, index]),
                "ci_high": float(intervals[1, index]),
                "w1": float(raw[index]),
                "w1_scaled": float(scaled[index]),
                "boundary_fraction": diag["boundary_fraction"],
            }
        )


def _bootstrap_dataset(results, name):
    model = results.datasets[name].model
    config = results.config
    generator = np.random.default_rng(config.seed + BOOTSTRAP_SEED_OFFSET)
    indices = generator.integers(
        0, len(model.y), size=(config.bootstrap_replicates, len(model.y))
    )
    noise = np.random.default_rng(
        config.seed + INTERVENTION_SEED_OFFSET
    ).normal(size=(config.intervention_repeats, len(model.y)))
    for loss in config.losses:
        baseline, point, diag = residual_bootstrap(
            model, model.y, indices, loss
        )
        key = results.baseline_key(name, loss)
        results.arrays[key] = baseline
        results.diagnostics[key] = diag
        results.baselines[name, loss] = point
        for alpha in config.levels:
            for repeat, standard_noise in enumerate(noise):
                if alpha == 0:
                    draws, estimate, diagnostics = baseline.copy(), point, diag
                else:
                    observations = conditional_noise(
                        model.y, model.scale, alpha, standard_noise
                    )
                    draws, estimate, diagnostics = residual_bootstrap(
                        model, observations, indices, loss
                    )
                key = results.sample_key(name, loss, alpha, repeat)
                results.arrays[key] = draws
                results.diagnostics[key] = diagnostics
                _record_condition(
                    results,
                    name,
                    loss,
                    alpha,
                    repeat,
                    draws,
                    estimate,
                    diagnostics,
                )
        print(f"  {name}: {loss} complete", flush=True)


def rmse(prediction, target):
    """Root mean squared prediction error."""
    return float(np.sqrt(np.mean((np.asarray(prediction) - target) ** 2)))


def decide_elements(dataset, point, boundary_fraction, config):
    """Refit after each ablation and apply the common prediction rule."""
    model = dataset.model
    full_error = rmse(
        model.predict(dataset.validation_x, point), dataset.validation_y
    )
    decisions = []
    for index, value in dataset.ablations.items():
        if dataset.name == "rlc" and index == 0:
            reduced = point.copy()
            reduced[index] = value
        else:
            reduced, _ = model.fit(loss=DECISION_LOSS, fixed={index: value})
        reduced_error = rmse(
            model.predict(dataset.validation_x, reduced), dataset.validation_y
        )
        delta = reduced_error - full_error
        remove = (
            delta <= dataset.tolerance
            and boundary_fraction <= config.max_boundary_fraction
        )
        decisions.append(
            {
                "dataset": dataset.name,
                "parameter": model.names[index],
                "estimate": float(point[index]),
                "ablated_value": float(value),
                "full_rmse": full_error,
                "reduced_rmse": reduced_error,
                "delta_rmse": delta,
                "tolerance": float(dataset.tolerance),
                "boundary_fraction": boundary_fraction,
                "decision": "УДАЛИТЬ" if remove else "СОХРАНИТЬ",
            }
        )
    return decisions


def _calculate_decisions(results):
    for name, dataset in results.datasets.items():
        boundary_fraction = max(
            diag["boundary_fraction"]
            for key, diag in results.diagnostics.items()
            if key.startswith(f"{name}_{DECISION_LOSS}_")
        )
        results.decisions.extend(
            decide_elements(
                dataset,
                results.baselines[name, DECISION_LOSS],
                boundary_fraction,
                results.config,
            )
        )


def _load_cache(results, fingerprint):
    output = results.output
    manifest = json.loads((output / "cache_manifest.json").read_text())
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(
            "Cache differs from inputs/configuration; rerun without cache"
        )
    with np.load(output / "bootstrap_draws.npz") as archive:
        results.arrays = dict(archive)
    results.records = json.loads(
        (output / "distances.json").read_text(encoding="utf-8")
    )
    results.diagnostics = json.loads(
        (output / "diagnostics.json").read_text(encoding="utf-8")
    )
    for name, dataset in results.datasets.items():
        for loss in results.config.losses:
            results.baselines[name, loss] = dataset.model.fit(loss=loss)[0]
    print("Loaded saved bootstrap draws (inputs and configuration verified).")


def run_experiment(root, datasets, config, reuse_cache=None):
    """Run all fits, or explicitly reuse fingerprint-validated saved draws."""
    output = root / OUTPUT_PATH
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(datasets, config)
    results = ExperimentResults(datasets, config, output, {}, [], {}, {}, [])
    if reuse_cache is None:
        reuse_cache = os.environ.get("LAB2_REUSE_BOOTSTRAP") == "1"
    if reuse_cache:
        _load_cache(results, fingerprint)
    else:
        for name in datasets:
            print(f"Computing {name}", flush=True)
            _bootstrap_dataset(results, name)
        np.savez_compressed(output / "bootstrap_draws.npz", **results.arrays)
        _write_json(output / "distances.json", results.records)
        _write_json(output / "diagnostics.json", results.diagnostics)
        _write_json(
            output / "cache_manifest.json", {"fingerprint": fingerprint}
        )
    _calculate_decisions(results)
    _write_json(output / "decisions.json", results.decisions)
    _write_json(
        output / "summary.json",
        {
            "config": asdict(config),
            "metadata": {
                name: data.metadata for name, data in datasets.items()
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "max_boundary_fraction": max(
                diag["boundary_fraction"]
                for diag in results.diagnostics.values()
            ),
        },
    )
    return results
