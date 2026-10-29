"""Compact notebook tables and environment checks."""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from IPython.display import Markdown, display

from .lab2_config import DECISION_LOSS, OFFSET_PARAMETER_NAME


def _table(headers, rows):
    lines = ["|" + "|".join(headers) + "|"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("|" + "|".join(map(str, row)) + "|" for row in rows)
    display(Markdown("\n".join(lines)))


def show_environment(root, config):
    """Verify that the notebook uses the project virtual environment."""
    expected = root / ".venv/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(
            f"Select Python (EMLab — lab2): expected {expected}; "
            f"got {sys.executable}"
        )
    print(f"Python: {sys.executable}")
    display(asdict(config))


def show_datasets(datasets):
    """Display the sizes and fitted coefficients of each experiment."""
    _table(
        ["Набор", "Обучение", "Проверка", "Коэффициенты"],
        [
            (
                name,
                len(data.model.x),
                len(data.validation_x),
                ", ".join(data.model.names),
            )
            for name, data in datasets.items()
        ],
    )


def show_parameter_summary(results):
    """Display point estimates, baseline intervals and mean example W1."""
    rows = []
    for name, data in results.datasets.items():
        for loss in results.config.losses:
            baseline = results.arrays[results.baseline_key(name, loss)]
            intervals = np.quantile(
                baseline, results.config.confidence_quantiles, axis=0
            )
            for index, parameter in enumerate(data.model.names):
                distances = [
                    record["w1"]
                    for record in results.records
                    if record["dataset"] == name
                    and record["loss"] == loss
                    and record["parameter"] == parameter
                    and record["alpha"] == results.config.example_alpha
                ]
                point = results.baselines[name, loss][index]
                interval = (
                    f"[{intervals[0, index]:.6g}; {intervals[1, index]:.6g}]"
                )
                rows.append(
                    (
                        name,
                        loss,
                        parameter,
                        f"{point:.6g}",
                        interval,
                        f"{np.mean(distances):.6g}",
                    )
                )
    _table(
        ["Набор", "Потери", "Параметр", "Оценка", "95% интервал", "W1"],
        rows,
    )


def show_decisions(results):
    """Display the actual prediction-based deletion decisions."""
    _table(
        [
            "Набор",
            "Параметр",
            "RMSE full",
            "RMSE reduced",
            "ΔRMSE",
            "Допуск",
            "Решение",
        ],
        [
            (
                record["dataset"],
                record["parameter"],
                f"{record['full_rmse']:.4g}",
                f"{record['reduced_rmse']:.4g}",
                f"{record['delta_rmse']:.4g}",
                f"{record['tolerance']:.4g}",
                record["decision"],
            )
            for record in results.decisions
        ],
    )


def show_offset_example(results):
    """Report the deletion rule for the redundant constant."""
    name = "silica_offset"
    index = results.datasets[name].model.names.index(OFFSET_PARAMETER_NAME)
    baseline = results.arrays[results.baseline_key(name, DECISION_LOSS)]
    lower, upper = np.quantile(
        baseline[:, index], results.config.confidence_quantiles
    )
    record = next(
        item
        for item in results.decisions
        if item["dataset"] == name
        and item["parameter"] == OFFSET_PARAMETER_NAME
    )
    _table(
        [
            "Параметр",
            "Истинное значение",
            "Оценка",
            "95% интервал",
            "ΔRMSE",
            "Допуск",
            "Решение",
        ],
        [
            (
                OFFSET_PARAMETER_NAME,
                0,
                f"{record['estimate']:.6g}",
                f"[{lower:.6g}; {upper:.6g}]",
                f"{record['delta_rmse']:.4g}",
                f"{record['tolerance']:.4g}",
                record["decision"],
            )
        ],
    )
