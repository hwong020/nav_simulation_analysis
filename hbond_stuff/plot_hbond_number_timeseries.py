"""Generate H-bond number plots with per-trial running averages.

This follows the compact visual format used by ``rms_scripts/run_rmsd_channels.py``:
faint raw traces for each trial, stronger centered running averages, time in ns,
and one output PNG per Nav channel under ``results/<channel>/graphs``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (3.8, 3.0)
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 14

RAW_ALPHA = 0.14
RAW_LINE_WIDTH = 0.7
AVG_ALPHA = 0.8
AVG_LINE_WIDTH = 1.2
RUNNING_WINDOW = 101

TARGET_CHANNELS = ["nav1-1", "nav1-2", "nav1-3", "nav1-5"]

TRIAL_COLORS = [
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#dc2626",
    "#6b7280",
]

CHANNEL_Y_SETTINGS: dict[str, tuple[float, list[float]]] = {
    "nav1-1": (4.0, [0.0, 1.0, 2.0, 3.0, 4.0]),
    "nav1-2": (4.0, [0.0, 1.0, 2.0, 3.0, 4.0]),
    "nav1-3": (1.0, [0.0, 0.25, 0.5, 0.75, 1.0]),
    "nav1-5": (6.0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
}


def nice_step(value: float) -> float:
    """Round a positive value up to a clean 1/2/5×10^n step."""
    if value <= 0:
        return 1.0

    exponent = np.floor(np.log10(value))
    fraction = value / (10**exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return float(nice_fraction * (10**exponent))


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


def running_average(values: np.ndarray, window: int = RUNNING_WINDOW) -> np.ndarray:
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


def _y_axis_label(channel: str) -> str:
    """Return the H-bond number axis label for one Nav channel."""

    if channel in {"nav1-1", "nav1-2", "nav1-3"}:
        return "Number of H-bonds \nbetween N and TTX"
    if channel == "nav1-5":
        return "Number of H-bonds \nbetween R and TTX"
    return "Number of H-bonds"


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
    """Plot raw H-bond traces and running averages for one channel."""
    hbond_dir = channel_dir / "hbonds"
    trial_paths = sorted(hbond_dir.glob("hb_num_ps_*.xvg"))
    if not trial_paths:
        raise FileNotFoundError(f"No H-bond files found in {hbond_dir}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    all_series: list[np.ndarray] = []

    for idx, path in enumerate(trial_paths):
        time_ns, hbond_number = load_hbond_xvg(path)
        smoothed = running_average(hbond_number)
        color = TRIAL_COLORS[idx % len(TRIAL_COLORS)]

        ax.plot(time_ns, hbond_number, color=color, alpha=RAW_ALPHA, linewidth=RAW_LINE_WIDTH)
        ax.plot(time_ns, smoothed, color=color, alpha=AVG_ALPHA, linewidth=AVG_LINE_WIDTH)

        all_series.append(hbond_number)

    channel_name = channel_dir.name.replace("nav1-", "Nav1.")

    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 250, 500, 750, 1000])

    if channel_dir.name in CHANNEL_Y_SETTINGS:
        y_max, y_ticks = CHANNEL_Y_SETTINGS[channel_dir.name]
        ax.set_ylim(0, y_max)
        ax.set_yticks(y_ticks)
    else:
        max_hbond = max(float(np.max(series)) for series in all_series)
        target_y_max = max(1.0, max_hbond * 1.15)
        y_step = nice_step(target_y_max / 5.0)
        y_max = y_step * 5.0
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0.0, y_max + (y_step * 0.5), y_step))

    style_axes(ax, _y_axis_label(channel_dir.name))
    fig.suptitle(channel_name, fontsize=TITLE_FONT_SIZE, x=0.6, y=0.96)
    fig.tight_layout(rect=(0.055, 0.025, 1, 0.985), pad=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hbond_number_{channel_dir.name}_running_avg.png"
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