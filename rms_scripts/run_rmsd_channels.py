"""Generate RMSD plots with per-trial running averages for available Nav channels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (5.75, 4.45)
PLOT_LEFT_INCHES = 1.20
PLOT_BOTTOM_INCHES = 0.75
PLOT_SIZE_INCHES = 3.15
TITLE_GAP_INCHES = 0.08
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 18
RMSD_Y_MAX = 60.0
RMSD_Y_TICKS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

RAW_ALPHA = 0.14
RAW_LINE_WIDTH = 0.7
AVG_ALPHA = 0.8
AVG_LINE_WIDTH = 1.2
RUNNING_WINDOW = 101

TRIAL_COLORS = [
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#dc2626",
    "#6b7280",
]


def add_centered_plot_title(fig: plt.Figure, title: str) -> None:
    """Place title at a fixed physical offset above the square plotting box."""
    title_x = (PLOT_LEFT_INCHES + (PLOT_SIZE_INCHES / 2.0)) / FIGURE_SIZE[0]
    title_y = (PLOT_BOTTOM_INCHES + PLOT_SIZE_INCHES + TITLE_GAP_INCHES) / FIGURE_SIZE[1]
    fig.text(
        title_x,
        title_y,
        title,
        ha="center",
        va="bottom",
        fontsize=TITLE_FONT_SIZE,
    )

def nice_step(value: float) -> float:
    """Round a positive value up to a clean 1/2/5×10^n step."""
    if value <= 0:
        return 1.0

    exponent = np.floor(np.log10(value))
    fraction = value / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return float(nice_fraction * (10 ** exponent))


def load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time/RMSD columns from a GROMACS XVG file."""
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
        raise ValueError(f"No valid RMSD data found in {path}")

    data = data[:, :2]
    data = data[~np.isnan(data).any(axis=1)]
    if data.size == 0:
        raise ValueError(f"No valid RMSD rows found in {path}")

    time_ns = data[:, 0] / 1000.0
    rmsd_angstrom = data[:, 1] * 10.0
    return time_ns, rmsd_angstrom


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


def style_axes(ax: plt.Axes) -> None:
    """Apply compact template-like styling."""
    ax.set_xlabel("Time (ns)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(
        "RMSD of TTX Inside \nBinding Position (Å)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=4,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=1.0, length=3)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.minorticks_on()


def plot_channel(channel_dir: Path, output_dir: Path) -> None:
    """Plot raw RMSD traces and running averages for one channel."""
    rmsd_dir = channel_dir / "rmsd"
    trial_paths = sorted(rmsd_dir.glob("rmsd_lig_*.xvg"))
    if not trial_paths:
        raise FileNotFoundError(f"No RMSD files found in {rmsd_dir}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_box_aspect(1)
    all_series: list[np.ndarray] = []
    all_times: list[np.ndarray] = []

    for idx, path in enumerate(trial_paths):
        time_ns, rmsd_angstrom = load_xvg(path)
        smoothed = running_average(rmsd_angstrom)
        color = TRIAL_COLORS[idx % len(TRIAL_COLORS)]

        ax.plot(time_ns, rmsd_angstrom, color=color, alpha=RAW_ALPHA, linewidth=RAW_LINE_WIDTH)
        ax.plot(time_ns, smoothed, color=color, alpha=AVG_ALPHA, linewidth=AVG_LINE_WIDTH)

        all_series.append(rmsd_angstrom)
        all_times.append(time_ns)

    channel_name = channel_dir.name.replace("nav1-", "Nav1.")

    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 250, 500, 750, 1000])

    ax.set_ylim(0, RMSD_Y_MAX)
    ax.set_yticks(RMSD_Y_TICKS)
    style_axes(ax)
    add_centered_plot_title(fig, channel_name)
    fig.subplots_adjust(
        left=PLOT_LEFT_INCHES / FIGURE_SIZE[0],
        bottom=PLOT_BOTTOM_INCHES / FIGURE_SIZE[1],
        right=(PLOT_LEFT_INCHES + PLOT_SIZE_INCHES) / FIGURE_SIZE[0],
        top=(PLOT_BOTTOM_INCHES + PLOT_SIZE_INCHES) / FIGURE_SIZE[1],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"rmsd_{channel_dir.name}_running_avg.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run RMSD plotting for channels that contain RMSD data."""
    src_root = Path("src")
    results_root = Path("results")

    for channel_dir in sorted(src_root.glob("nav1-*")):
        rmsd_dir = channel_dir / "rmsd"
        if not rmsd_dir.exists():
            print(f"Skipping {channel_dir.name}: no rmsd directory")
            continue

        try:
            plot_channel(channel_dir, results_root / channel_dir.name / "graphs")
            print(f"Generated RMSD plot for {channel_dir.name}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel_dir.name}: {exc}")


if __name__ == "__main__":
    main()