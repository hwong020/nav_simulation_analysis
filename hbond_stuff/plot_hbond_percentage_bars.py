"""Plot H-bond frame percentages calculated directly from raw XVG files.

This script reads ``src/<channel>/hbonds/hb_num_ps_*.xvg`` files, calculates the
percentage of frames with each H-bond count from 1 through 5 for each trial,
and plots the mean percentage with standard deviation error bars per channel.
The 0 H-bond count is intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
TARGET_CHANNELS = ["nav1-1", "nav1-2", "nav1-3", "nav1-5"]
HBOND_COUNTS = list(range(1, 6))

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
Y_AXIS_MAX = 50.0
Y_AXIS_TICKS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


@dataclass(frozen=True)
class HbondStats:
    """Average and standard deviation frame percentages for one channel."""

    label: str
    source_paths: tuple[Path, ...]
    averages: dict[int, float]
    std_devs: dict[int, float]


def _display_label(label: str) -> str:
    """Convert labels like '1-1' to display labels like 'Nav1.1'."""

    match = re.fullmatch(r"(\d+)-(\d+)", label)
    if match:
        return f"Nav{match.group(1)}.{match.group(2)}"
    return label


def _x_axis_label(label: str) -> str:
    """Return the H-bond count axis label for one Nav channel."""

    return "Number of H bonds"


def _label_from_channel(channel: str) -> str:
    """Create the output label used by the existing plot filenames."""

    return channel.removeprefix("nav")


def load_hbond_counts(path: Path) -> np.ndarray:
    """Load the H-bond count column from a GROMACS XVG file."""

    with path.open("r", encoding="utf-8") as handle:
        lines = [
            line
            for line in handle
            if line.strip() and not line.startswith("#") and not line.startswith("@")
        ]

    if not lines:
        raise ValueError(f"No data found in {path}")

    data = np.genfromtxt(lines, invalid_raise=False)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] < 2:
        raise ValueError(f"No valid H-bond data found in {path}")

    hbond_counts = data[:, 1]
    hbond_counts = hbond_counts[~np.isnan(hbond_counts)]
    if hbond_counts.size == 0:
        raise ValueError(f"No valid H-bond count rows found in {path}")

    return np.rint(hbond_counts).astype(int)


def calculate_trial_percentages(hbond_counts: np.ndarray) -> dict[int, float]:
    """Calculate frame percentages by H-bond count for one trial."""

    total_frames = hbond_counts.size
    if total_frames == 0:
        raise ValueError("Cannot calculate H-bond percentages with zero frames")

    return {
        count: float(np.count_nonzero(hbond_counts == count) / total_frames * 100.0)
        for count in HBOND_COUNTS
    }


def calculate_hbond_stats(channel_dir: Path) -> HbondStats:
    """Calculate mean and standard deviation frame percentages for one channel."""

    hbond_dir = channel_dir / "hbonds"
    trial_paths = tuple(sorted(hbond_dir.glob("hb_num_ps_*.xvg")))
    if not trial_paths:
        raise FileNotFoundError(f"No H-bond XVG files found in {hbond_dir}")

    trial_counts = [load_hbond_counts(path) for path in trial_paths]
    trial_percentages = [calculate_trial_percentages(hbond_counts) for hbond_counts in trial_counts]

    averages = {
        count: float(np.mean([trial[count] for trial in trial_percentages]))
        for count in HBOND_COUNTS
    }
    std_devs = {
        count: float(np.std([trial[count] for trial in trial_percentages], ddof=0))
        for count in HBOND_COUNTS
    }

    return HbondStats(
        label=_label_from_channel(channel_dir.name),
        source_paths=trial_paths,
        averages=averages,
        std_devs=std_devs,
    )


def _values_for_counts(stats: HbondStats) -> tuple[list[int], list[float], list[float]]:
    """Return average/std values for all displayed H-bond counts."""

    counts = HBOND_COUNTS.copy()
    means = [stats.averages.get(count, 0.0) for count in counts]
    errors = [stats.std_devs.get(count, 0.0) for count in counts]
    return counts, means, errors


def plot_individual(stats: HbondStats) -> Path:
    """Create one average ± standard deviation bar chart for one stats file."""

    counts, means, errors = _values_for_counts(stats)
    if not any(mean > 0 for mean in means):
        raise ValueError(f"No H-bond percentages found for counts 1-5 in {stats.source_paths}")

    x = np.asarray(counts, dtype=float) * BAR_SPACING

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

    ax.set_ylim(0, Y_AXIS_MAX)
    ax.set_yticks(Y_AXIS_TICKS)
    ax.set_xlim(0.5 * BAR_SPACING, 5.5 * BAR_SPACING)
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
    """Calculate frame percentages from XVG files and generate plots."""

    all_stats: list[HbondStats] = []
    for channel in TARGET_CHANNELS:
        channel_dir = SRC_ROOT / channel
        try:
            all_stats.append(calculate_hbond_stats(channel_dir))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel}: {exc}")

    if not all_stats:
        raise FileNotFoundError(f"No H-bond XVG files found under {SRC_ROOT}")

    output_paths = [plot_individual(stats) for stats in all_stats]

    print("Generated H-bond percentage plots:")
    for output_path in output_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()