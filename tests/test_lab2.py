"""Checks for modular experiments, caching and parameter deletion."""

import ast
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pytest

from emlab.lab2_config import DEFAULT_CONFIG, NOTEBOOK_PATH
from emlab.lab2_data import prepare_datasets
from emlab.lab2_experiment import decide_elements, run_experiment

ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG = replace(
    DEFAULT_CONFIG,
    bootstrap_replicates=30,
    intervention_repeats=2,
    levels=(0.0, 2.0),
)


@pytest.fixture(scope="module")
def datasets():
    return prepare_datasets(ROOT, TEST_CONFIG)


def test_extra_offset_uses_the_same_data_and_reduces_to_original(datasets):
    silica = datasets["silica"].model
    extended = datasets["silica_offset"].model
    np.testing.assert_array_equal(extended.y, silica.y)
    np.testing.assert_array_equal(extended.scale, silica.scale)
    np.testing.assert_allclose(
        extended.predict(extended.x, np.r_[silica.initial, 0]),
        silica.predict(silica.x, silica.initial),
    )
    assert extended.bounds[0][-1] < 0 < extended.bounds[1][-1]


def test_prediction_rule_removes_redundant_offset(datasets):
    dataset = datasets["silica_offset"]
    point, boundary = dataset.model.fit()
    assert not boundary
    decisions = decide_elements(dataset, point, 0.0, TEST_CONFIG)
    offset = next(item for item in decisions if item["parameter"] == "c0")
    assert offset["decision"] == "УДАЛИТЬ"
    assert offset["delta_rmse"] <= offset["tolerance"]
    for item in decisions:
        if item["parameter"].startswith("a"):
            assert item["decision"] == "СОХРАНИТЬ"


def test_boundary_diagnostic_prevents_deletion(datasets):
    dataset = datasets["silica_offset"]
    point, _ = dataset.model.fit()
    decisions = decide_elements(dataset, point, 1.0, TEST_CONFIG)
    assert all(item["decision"] == "СОХРАНИТЬ" for item in decisions)


def test_cache_round_trip_and_configuration_validation(tmp_path, datasets):
    selected = {"silica_offset": datasets["silica_offset"]}
    calculated = run_experiment(
        tmp_path, selected, TEST_CONFIG, reuse_cache=False
    )
    cached = run_experiment(tmp_path, selected, TEST_CONFIG, reuse_cache=True)
    for key in calculated.arrays:
        np.testing.assert_array_equal(
            calculated.arrays[key], cached.arrays[key]
        )
    assert all(
        item["w1"] == 0 for item in calculated.records if item["alpha"] == 0
    )
    altered = replace(TEST_CONFIG, seed=TEST_CONFIG.seed + 1)
    with pytest.raises(ValueError, match="Cache differs"):
        run_experiment(tmp_path, selected, altered, reuse_cache=True)
    original = selected["silica_offset"]
    modified = replace(
        original, model=replace(original.model, y=original.model.y + 0.01)
    )
    with pytest.raises(ValueError, match="Cache differs"):
        run_experiment(
            tmp_path,
            {"silica_offset": modified},
            TEST_CONFIG,
            reuse_cache=True,
        )


def test_notebook_contains_only_short_orchestration_cells():
    notebook = nbformat.read(ROOT / NOTEBOOK_PATH, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert len(cell.source.splitlines()) <= 12
        assert not any(
            isinstance(
                node, (ast.For, ast.While, ast.FunctionDef, ast.ClassDef)
            )
            for node in ast.walk(ast.parse(cell.source))
        )
    text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "Выводы" not in text
    assert "неопределённость опубликованных" not in text


def test_matplotlib_cache_is_configured_before_font_manager(tmp_path):
    blocked = tmp_path / "unusable-cache"
    blocked.write_text("This path is a file, not a cache directory.")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(blocked)
    environment["PYTHONIOENCODING"] = "utf-8"
    script = """
import json
from pathlib import Path
from emlab.lab2_runtime import configure_matplotlib_cache

directory = configure_matplotlib_cache(Path.cwd())
import matplotlib
from matplotlib import font_manager

assert Path(matplotlib.get_cachedir()) == directory
assert list(directory.glob('fontlist-*.json'))
print(json.dumps(str(directory)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    assert Path(json.loads(completed.stdout)) == tmp_path / "tmp/matplotlib"
    assert "Permission denied" not in completed.stderr
