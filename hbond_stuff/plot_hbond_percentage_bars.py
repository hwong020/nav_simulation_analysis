"""Plot H-bond frame-percentage averages with standard deviation error bars.

This script parses the sectioned ``hbond_percentages_stats_*.csv`` files in this
folder and creates individual bar charts for non-zero H-bond counts 1, 2, 3,
and 4. The 0 H-bond section is intentionally excluded, and any selected H-bond
count with a zero/missing average is omitted from that dataset's plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATTERN = "hbond_percentages_stats_*.csv"
HBOND_COUNTS = [1, 2, 3, 4]

FIGURE_SIZE = (9.0, 6.8)
DPI = 300

TITLE_FONT_SIZE = 30
AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 24
BAR_WIDTH = 0.65
BAR_SPACING = 1.35

BAR_COLOR = "#ff69b4"
ERROR_BAR_COLOR = "#222222"
GRID_ALPHA = 0.22


@dataclass(frozen=True)
class HbondStats:
    """Average and standard deviation frame percentages for one data file."""

    label: str
    source_path: Path
    averages: dict[int, float]
    std_devs: dict[int, float]


def _label_from_path(path: Path) -> str:
    """Create a concise plot label from a stats filename."""

    match = re.search(r"hbond_percentages_stats_(.+)$", path.stem)
    if match:
        return match.group(1)
    return path.stem


def _display_label(label: str) -> str:
    """Convert labels like '1-1' to display labels like 'Nav1.1'."""

    match = re.fullmatch(r"(\d+)-(\d+)", label)
    if match:
        return f"Nav{match.group(1)}.{match.group(2)}"
    return label


def _x_axis_label(label: str) -> str:
    """Return the H-bond count axis label for one Nav channel."""

    channel_label = _display_label(label)
    if channel_label in {"Nav1.1", "Nav1.2", "Nav1.3"}:
        return "Number of H-bonds between N and TTX"
    if channel_label == "Nav1.5":
        return "Number of H-bonds between R and TTX"
    return "Number of H-bonds"


def parse_hbond_stats(path: Path) -> HbondStats:
    """Parse average and standard deviation values by H-bond count."""

    averages: dict[int, float] = {}
    std_devs: dict[int, float] = {}
    current_count: int | None = None

    section_pattern = re.compile(r"---\s+(\d+)\s+HYDROGEN BOND\(S\) SECTION\s+---")

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            section_match = section_pattern.fullmatch(line)
            if section_match:
                current_count = int(section_match.group(1))
                continue

            if current_count not in HBOND_COUNTS:
                continue

            if line.startswith("AVERAGE,"):
                averages[current_count] = float(line.split(",", maxsplit=1)[1])
            elif line.startswith("STANDARD DEVIATION,"):
                std_devs[current_count] = float(line.split(",", maxsplit=1)[1])

    return HbondStats(
        label=_label_from_path(path),
        source_path=path,
        averages=averages,
        std_devs=std_devs,
    )


def _nonzero_values_for_counts(stats: HbondStats) -> tuple[list[int], list[float], list[float]]:
    """Return counts and average/std values, omitting zero or missing averages."""

    counts: list[int] = []
    means: list[float] = []
    errors: list[float] = []
    for count in HBOND_COUNTS:
        mean = stats.averages.get(count, 0.0)
        if mean <= 0:
            continue
        counts.append(count)
        means.append(mean)
        errors.append(stats.std_devs.get(count, 0.0))
    return counts, means, errors


def plot_individual(stats: HbondStats) -> Path:
    """Create one average ± standard deviation bar chart for one stats file."""

    counts, means, errors = _nonzero_values_for_counts(stats)
    if not counts:
        raise ValueError(f"No non-zero H-bond averages found for {stats.source_path}")

    # Center the visible, non-zero bars as a group while preserving a fixed
    # axis span and fixed bar width across all generated plots.
    x = np.arange(len(counts), dtype=float)
    x -= x.mean()
    x *= BAR_SPACING

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        yerr=errors,
        capsize=6,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": ERROR_BAR_COLOR},
    )

    y_max = max((mean + std for mean, std in zip(means, errors, strict=False)), default=1.0)
    ax.set_ylim(0, max(1.0, y_max * 1.25))
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in counts], fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.set_xlabel(_x_axis_label(stats.label), fontsize=AXIS_LABEL_FONT_SIZE, labelpad=14)
    ax.set_ylabel("Frames percentage (%)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=14)
    ax.set_title(
        f"H-bond frame percentages ({_display_label(stats.label)})",
        fontsize=TITLE_FONT_SIZE,
        pad=20,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = SCRIPT_DIR / f"hbond_percentage_avg_std_{stats.label}.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Parse all stats files and generate individual plots."""

    input_paths = sorted(SCRIPT_DIR.glob(INPUT_PATTERN))
    if not input_paths:
        raise FileNotFoundError(f"No files matching {INPUT_PATTERN!r} found in {SCRIPT_DIR}")

    all_stats = [parse_hbond_stats(path) for path in input_paths]
    output_paths = [plot_individual(stats) for stats in all_stats]

    print("Generated H-bond percentage plots:")
    for output_path in output_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()