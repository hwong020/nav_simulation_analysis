"""Core plotting utilities for minimum-distance (mindist) analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 12.5
TICK_LABEL_FONT_SIZE = 11.5
LEGEND_FONT_SIZE = 12.5

FIGURE_SIZE = (5, 3.9)
TRACE_LINE_WIDTH = 0.8
TRACE_ALPHA = 0.4
RUNNING_AVG_LINE_WIDTH = 1.05
RUNNING_AVG_ALPHA = 0.95
RUNNING_WINDOW = 101
CUTOFF_LINE_WIDTH = 1.2
CUTOFF_LINE_ALPHA = 0.95
Y_PADDING_FRACTION = 0.08
MIN_Y_PADDING = 0.5
GRID_LINE_WIDTH = 0.45
GRID_ALPHA = 0.4


@dataclass(frozen=True)
class MindistScenario:
    """Configuration for a single channel/scenario mindist analysis."""
    name: str
    input_dir: Path
    output_dir: Path
    residues: list[str]
    trials: list[int]
    hbond_cutoff_angstrom: float = 3.5
    hydrophobic_cutoff_angstrom: float = 4.0


# Mapping of channel → DEKA residue labels for plot annotations.
DEKA_LABELS = {
    "nav1-1": ["D382", "E951", "K1432", "A1724"],
    "nav1-2": ["D384", "E942", "K1422", "A1714"],
    "nav1-3": ["D383", "E943", "K1417", "A1709"],
    "nav1-4": ["D406", "E761", "K1244", "A1536"],
    "nav1-5": ["D372", "E898", "K1419", "A1711"],
    "nav1-6": ["D370", "E936", "K1413", "A1705"],
    "nav1-7": ["D361", "E927", "K1406", "A1698"],
    "nav1-8": ["D356", "E849", "K1367", "A1661"],
    "nav1-9": ["D359", "E768", "K1257", "A1551"],
}


def _format_channel_name(scenario_name: str) -> str:
    """Convert scenario name into a human-friendly channel label."""
    base = scenario_name.split("-")[0:2]
    if len(base) == 2 and base[0].lower() == "nav1":
        return f"Nav1.{base[1]}"
    return scenario_name.replace("nav1-", "Nav1.")


def _format_time_ns(time_ps: np.ndarray) -> np.ndarray:
    """Convert GROMACS time values from ps to ns for plotting."""
    return time_ps / 1000.0


def _get_residue_labels(scenario_name: str, residues: list[str]) -> list[str]:
    """Return DEKA labels for known channels or default to residue codes."""
    key = "-".join(scenario_name.split("-")[0:2]).lower()
    return DEKA_LABELS.get(key, [res.upper() for res in residues])


def _load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an XVG file, returning time and distance arrays.

    Some XVG exports contain malformed rows with missing columns. We tolerate
    these rows by dropping any line that does not have exactly two numeric
    values. We also strip GROMACS-style metadata lines that start with @ or #.
    """
    with path.open("r", encoding="utf-8") as handle:
        cleaned_lines = [
            line
            for line in handle
            if line.strip() and not line.startswith("@") and not line.startswith("#")
        ]
    if not cleaned_lines:
        raise ValueError(f"No data found in {path}")
    data = np.genfromtxt(cleaned_lines, invalid_raise=False)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.size == 0:
        raise ValueError(f"No data found in {path}")
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    data = data[:, :2]
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    if data.size == 0:
        raise ValueError(f"No valid rows found in {path}")
    return data[:, 0], data[:, 1]


def _collect_trial_series(
    scenario: MindistScenario, residue: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load all trial distance series for a residue and stack them."""
    times: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for trial in scenario.trials:
        # Each trial uses a filename like mindist_{residue}_{trial}.xvg
        filename = scenario.input_dir / f"mindist_{residue}_{trial}.xvg"
        if not filename.exists():
            raise FileNotFoundError(f"Missing {filename}")
        time, dist = _load_xvg(filename)
        times.append(time)
        dists.append(dist)
    time0 = times[0]
    for t in times[1:]:
        # Ensure the time axis is consistent across trials.
        if not np.array_equal(t, time0):
            raise ValueError(
                f"Time axis mismatch for residue {residue} in {scenario.input_dir}"
            )
    return time0, np.vstack(dists)


def _running_average(values: np.ndarray, window: int = RUNNING_WINDOW) -> np.ndarray:
    """Return a centered running average with an odd window size."""
    if values.size < 3:
        return values

    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values

    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_trial_time_series_overlay(scenario: MindistScenario) -> None:
    """Plot per-trial time series overlays for all residues in a scenario."""
    scenario.output_dir.mkdir(parents=True, exist_ok=True)
    channel_label = _format_channel_name(scenario.name)
    residue_labels = _get_residue_labels(scenario.name, scenario.residues)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

    for trial in scenario.trials:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.set_box_aspect(1)
        time_ref: np.ndarray | None = None
        min_len: int | None = None
        residue_series: list[tuple[int, np.ndarray, np.ndarray]] = []
        plotted_distances: list[np.ndarray] = []

        for idx, residue in enumerate(scenario.residues):
            filename = scenario.input_dir / f"mindist_{residue}_{trial}.xvg"
            if not filename.exists():
                raise FileNotFoundError(f"Missing {filename}")
            time, dist = _load_xvg(filename)
            if time_ref is None:
                time_ref = time
                min_len = time.shape[0]
            else:
                min_len = min(min_len or time.shape[0], time.shape[0])
            residue_series.append((idx, time, dist))

        if time_ref is None or min_len is None:
            raise ValueError(f"No data loaded for {scenario.name} trial {trial}")

        for idx, time, dist in residue_series:
            time = _format_time_ns(time[:min_len])
            dist = dist[:min_len]
            dist_angstrom = dist * 10.0
            plotted_distances.append(dist_angstrom)

            try:
                residue_label = residue_labels[idx]
            except IndexError:
                residue_label = residue.upper()
            color = colors[idx % len(colors)]
            ax.plot(
                time,
                dist_angstrom,
                color=color,
                linewidth=TRACE_LINE_WIDTH,
                alpha=TRACE_ALPHA,
                label=residue_label,
            )
            ax.plot(
                time,
                _running_average(dist_angstrom),
                color=color,
                linewidth=RUNNING_AVG_LINE_WIDTH,
                alpha=RUNNING_AVG_ALPHA,
            )

        ax.axhline(
            scenario.hydrophobic_cutoff_angstrom,
            color="#000000",
            linestyle=":",
            linewidth=CUTOFF_LINE_WIDTH,
            alpha=CUTOFF_LINE_ALPHA,
            zorder=4,
            label=f"Contact ({scenario.hydrophobic_cutoff_angstrom:.1f} Å)",
        )
        ax.set_xlabel("Time (ns)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=4)
        ax.set_ylabel(
            "Distance between Specified \nResidue and TTX (Å)",
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=4,
        )
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 250, 500, 750, 1000])
        if plotted_distances:
            all_distances = np.concatenate(plotted_distances)
            y_min = float(np.min(all_distances))
            y_max = float(np.max(all_distances))
            y_span = max(y_max - y_min, 1e-6)
            y_padding = max(y_span * Y_PADDING_FRACTION, MIN_Y_PADDING)
            ax.set_ylim(max(0.0, y_min - y_padding), y_max + y_padding)
        ax.grid(True, linestyle="-", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA, color="#b3b3b3")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, pad=2, width=0.8, length=3)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        legend = ax.legend(
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderpad=0.3,
            borderaxespad=0.0,
            handlelength=1.4,
            handletextpad=0.35,
            columnspacing=0.7,
            fontsize=LEGEND_FONT_SIZE,
        )
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
            handle.set_linewidth(max(2.0, TRACE_LINE_WIDTH))
        ax.set_title(channel_label, fontsize=TITLE_FONT_SIZE, pad=8)
        fig.tight_layout(rect=(0.08, 0.06, 0.84, 0.96), pad=0.04)

        output_path = (
            scenario.output_dir
            / f"mindist_{scenario.name}_trial{trial}_DEKA.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def run_scenario(scenario: MindistScenario) -> None:
    """Run all plots for a scenario and write outputs to disk."""
    plot_trial_time_series_overlay(scenario)