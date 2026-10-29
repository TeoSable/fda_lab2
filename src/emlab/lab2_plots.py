"""Notebook visualizations for laboratory 2."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.ticker import MaxNLocator

from .lab2_config import DECISION_LOSS, OFFSET_PARAMETER_NAME


@dataclass(frozen=True)
class PlotConfig:
    """Display and export settings shared by every figure."""

    dpi: int = 120
    bins: int = 25
    max_ticks: int = 4
    panel_width: float = 4.0
    distribution_height: float = 6.0
    distance_height: float = 3.5
    histogram_alpha: float = 0.5
    band_alpha: float = 0.15
    scatter_alpha: float = 0.4
    scatter_size: int = 7
    legend_font_size: int = 8
    node_size: int = 2400
    arrow_size: int = 22
    node_color: str = "#cce5f3"


PLOT_CONFIG = PlotConfig()
JOINT_FIGURE_SIZE = (10, 4)
STRUCTURE_FIGURE_SIZE = (11, 4)
OFFSET_FIGURE_SIZE = (10, 4)
RLC_POSITIONS = {"reader": (0, 0), "tag": (1, 0)}
RLC_VIEW_X = (-0.25, 1.4)
RLC_VIEW_Y = (-0.18, 0.5)
RLC_CONNECTION_STYLE = "arc3,rad=0.18"
RLC_SELF_LOOP_START = (0.92, 0.04)
RLC_SELF_LOOP_END = (1.08, 0.04)
RLC_SELF_LOOP_STYLE = "arc3,rad=-2.8"
RLC_EDGE_LABELS = ((0.5, -0.055, "M"), (0.5, 0.055, "M"), (1, 0.31, "R2"))
OPTICAL_GRAPH_MARGIN = 0.25


def configure_plots():
    """Apply the common notebook style."""
    plt.rcParams.update(
        {
            "figure.dpi": PLOT_CONFIG.dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _finish(figure, results, filename):
    figure.tight_layout()
    figure.savefig(results.output / f"{filename}.png", bbox_inches="tight")
    plt.show()
    plt.close(figure)


def plot_distributions(results):
    """Compare baseline and intervened marginal bootstrap distributions."""
    configure_plots()
    style = PLOT_CONFIG
    for name, dataset in results.datasets.items():
        count = len(dataset.model.names)
        figure, axes = plt.subplots(
            len(results.config.losses),
            count,
            figsize=(style.panel_width * count, style.distribution_height),
            squeeze=False,
        )
        for row, loss in enumerate(results.config.losses):
            baseline = results.arrays[results.baseline_key(name, loss)]
            changed = results.example_draws(name, loss)
            for index, parameter in enumerate(dataset.model.names):
                axis = axes[row, index]
                bins = np.histogram_bin_edges(
                    np.r_[baseline[:, index], changed[:, index]],
                    bins=style.bins,
                )
                for draws, label in (
                    (baseline, "baseline"),
                    (changed, f"alpha={results.config.example_alpha:g}"),
                ):
                    axis.hist(
                        draws[:, index],
                        bins=bins,
                        density=True,
                        alpha=style.histogram_alpha,
                        label=label,
                    )
                axis.xaxis.set_major_locator(MaxNLocator(style.max_ticks))
                axis.set(
                    title=f"{loss}: {parameter}",
                    xlabel=parameter,
                    ylabel="density",
                )
                axis.legend(fontsize=style.legend_font_size)
        figure.suptitle(f"{name}: bootstrap distributions")
        _finish(figure, results, f"{name}_distributions")


def plot_distances(results):
    """Plot mean W1 with min/max bands across intervention repeats."""
    configure_plots()
    style = PLOT_CONFIG
    for name, dataset in results.datasets.items():
        count = len(dataset.model.names)
        figure, axes = plt.subplots(
            1,
            count,
            figsize=(style.panel_width * count, style.distance_height),
            squeeze=False,
        )
        for index, parameter in enumerate(dataset.model.names):
            axis = axes[0, index]
            for loss in results.config.losses:
                values = np.array(
                    [
                        [
                            record["w1_scaled"]
                            for record in results.records
                            if record["dataset"] == name
                            and record["loss"] == loss
                            and record["parameter"] == parameter
                            and record["alpha"] == alpha
                        ]
                        for alpha in results.config.levels
                    ]
                )
                axis.plot(
                    results.config.levels,
                    values.mean(axis=1),
                    "o-",
                    label=loss,
                )
                axis.fill_between(
                    results.config.levels,
                    values.min(axis=1),
                    values.max(axis=1),
                    alpha=style.band_alpha,
                )
            axis.set(
                title=parameter, xlabel="alpha", ylabel="W1 / baseline SD"
            )
            axis.legend(fontsize=style.legend_font_size)
        figure.suptitle(f"{name}: intervention sensitivity")
        _finish(figure, results, f"{name}_distance")


def plot_joint_rlc(results):
    """Show joint M/R2 bootstrap clouds for each loss."""
    figure, axes = plt.subplots(
        1, len(results.config.losses), figsize=JOINT_FIGURE_SIZE, squeeze=False
    )
    for axis, loss in zip(axes[0], results.config.losses):
        baseline = results.arrays[results.baseline_key("rlc", loss)]
        changed = results.example_draws("rlc", loss)
        for draws, label in (
            (baseline, "baseline"),
            (changed, "intervention"),
        ):
            axis.scatter(
                *draws.T,
                s=PLOT_CONFIG.scatter_size,
                alpha=PLOT_CONFIG.scatter_alpha,
                label=label,
            )
        axis.set(title=loss, xlabel="M [nH]", ylabel="R2 [ohm]")
        axis.legend()
    _finish(figure, results, "rlc_joint")


def plot_structure(results):
    """Draw RLC couplings and the extra optical-model branch."""
    figure, axes = plt.subplots(1, 2, figsize=STRUCTURE_FIGURE_SIZE)
    graph = nx.DiGraph()
    graph.add_edge("reader", "tag", parameter="M")
    graph.add_edge("tag", "reader", parameter="M")
    graph.add_edge("tag", "tag", parameter="R2")
    nx.write_graphml(graph, results.output / "rlc_structure.graphml")
    nx.draw_networkx(
        graph,
        RLC_POSITIONS,
        ax=axes[0],
        edgelist=[("reader", "tag"), ("tag", "reader")],
        node_color=PLOT_CONFIG.node_color,
        node_size=PLOT_CONFIG.node_size,
        arrowsize=PLOT_CONFIG.arrow_size,
        connectionstyle=RLC_CONNECTION_STYLE,
    )
    axes[0].annotate(
        "",
        xy=RLC_SELF_LOOP_END,
        xytext=RLC_SELF_LOOP_START,
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": RLC_SELF_LOOP_STYLE,
        },
    )
    for x_position, y_position, label in RLC_EDGE_LABELS:
        axes[0].text(x_position, y_position, label, ha="center")
    axes[0].set(xlim=RLC_VIEW_X, ylim=RLC_VIEW_Y, title="RLC structure")

    optical = nx.DiGraph()
    branches = ["a1 term", "a2 term", "a3 term", OFFSET_PARAMETER_NAME]
    optical.add_edges_from((branch, "n² - 1") for branch in branches)
    positions = {branch: (0, index) for index, branch in enumerate(branches)}
    positions["n² - 1"] = (1, (len(branches) - 1) / 2)
    nx.draw_networkx(
        optical,
        positions,
        ax=axes[1],
        node_color=PLOT_CONFIG.node_color,
        node_size=PLOT_CONFIG.node_size,
        arrowsize=PLOT_CONFIG.arrow_size,
    )
    axes[1].set_title("Dispersion with an extra constant")
    axes[1].margins(OPTICAL_GRAPH_MARGIN)
    nx.write_graphml(optical, results.output / "offset_structure.graphml")
    for axis in axes:
        axis.axis("off")
    _finish(figure, results, "rlc_structure")


def plot_offset_example(results):
    """Show uncertainty of the redundant constant and validation errors."""
    name = "silica_offset"
    parameter_index = results.datasets[name].model.names.index(
        OFFSET_PARAMETER_NAME
    )
    figure, axes = plt.subplots(1, 2, figsize=OFFSET_FIGURE_SIZE)
    baseline = results.arrays[results.baseline_key(name, DECISION_LOSS)]
    axes[0].hist(
        baseline[:, parameter_index], bins=PLOT_CONFIG.bins, density=True
    )
    axes[0].axvline(0, color="black", linestyle="--", label="true c0 = 0")
    axes[0].set(xlabel="c0", ylabel="density", title="Extra coefficient")
    axes[0].xaxis.set_major_locator(MaxNLocator(PLOT_CONFIG.max_ticks))
    axes[0].legend()
    decision = next(
        record
        for record in results.decisions
        if record["dataset"] == name
        and record["parameter"] == OFFSET_PARAMETER_NAME
    )
    axes[1].bar(
        ["Full model", "c0 removed"],
        [decision["full_rmse"], decision["reduced_rmse"]],
    )
    axes[1].set(ylabel="Validation RMSE in n² - 1", title="Ablation check")
    _finish(figure, results, "offset_example")
