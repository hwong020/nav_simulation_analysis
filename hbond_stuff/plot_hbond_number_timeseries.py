"""Generate H-bond number time-series plots.

This follows the compact visual format used by ``rms_scripts/run_rmsd_channels.py``:
one raw trace for each trial, time in ns, and one output PNG per Nav channel
under ``results/<channel>/graphs``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (3.8, 3.0)
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 14

RAW_ALPHA = 0.8
RAW_LINE_WIDTH = 1.2
HBOND_Y_MAX = 6.0
HBOND_Y_TICKS = list(np.arange(0.0, HBOND_Y_MAX + 1.0, 2.0))

TARGET_CHANNELS = ["nav1-1", "nav1-2", "nav1-3", "nav1-5"]

TRIAL_COLORS = [
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#dc2626",
    "#6b7280",
]

def load_hbond_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time and H-bond number columns from a GROMACS XVG file."""
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

    data = data[:, :2]
    data = data[~np.isnan(data).any(axis=1)]
    if data.size == 0:
        raise ValueError(f"No valid H-bond rows found in {path}")

    time_ns = data[:, 0] / 1000.0
    hbond_number = data[:, 1]
    return time_ns, hbond_number


def _y_axis_label(channel: str) -> str:
    """Return the H-bond number axis label for one Nav channel."""

    return "Number of H-bonds\nbetween DEKA and TTX"


def style_axes(ax: plt.Axes, y_axis_label: str) -> None:
    """Apply the same compact template-like styling as the RMSD graphs."""
    ax.set_xlabel("Time (ns)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=4)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=1.0, length=3)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.minorticks_on()


def plot_channel(channel_dir: Path, output_dir: Path) -> Path:
    """Plot raw H-bond traces for one channel."""
    hbond_dir = channel_dir / "hbonds"
    trial_paths = sorted(hbond_dir.glob("hb_num_ps_*.xvg"))
    if not trial_paths:
        raise FileNotFoundError(f"No H-bond files found in {hbond_dir}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for idx, path in enumerate(trial_paths):
        time_ns, hbond_number = load_hbond_xvg(path)
        color = TRIAL_COLORS[idx % len(TRIAL_COLORS)]

        ax.plot(time_ns, hbond_number, color=color, alpha=RAW_ALPHA, linewidth=RAW_LINE_WIDTH)

    channel_name = channel_dir.name.replace("nav1-", "Nav1.")

    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 250, 500, 750, 1000])

    ax.set_ylim(0, HBOND_Y_MAX)
    ax.set_yticks(HBOND_Y_TICKS)

    style_axes(ax, _y_axis_label(channel_dir.name))
    fig.suptitle(channel_name, fontsize=TITLE_FONT_SIZE, x=0.6, y=0.96)
    fig.tight_layout(rect=(0.055, 0.025, 1, 0.985), pad=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hbond_number_{channel_dir.name}_timeseries.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    """Run H-bond number plotting for selected Nav channels."""
    src_root = Path("src")
    results_root = Path("results")

    for channel in TARGET_CHANNELS:
        channel_dir = src_root / channel
        hbond_dir = channel_dir / "hbonds"
        if not hbond_dir.exists():
            print(f"Skipping {channel}: no hbonds directory")
            continue

        try:
            output_path = plot_channel(channel_dir, results_root / channel / "graphs")
            print(f"Generated H-bond number plot: {output_path}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel}: {exc}")


if __name__ == "__main__":
    main()