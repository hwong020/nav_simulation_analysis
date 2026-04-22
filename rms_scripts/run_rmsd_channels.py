"""Generate RMSD plots with per-trial running averages for available Nav channels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (3.8, 3.0)
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 10

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
    ax.set_ylabel("RMSD (Å)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=1.0, length=3)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.minorticks_on()


def plot_channel(channel_dir: Path, output_dir: Path) -> None:
    """Plot raw RMSD traces and running averages for one channel."""
    rmsd_dir = channel_dir / "rmsd"
    trial_paths = [rmsd_dir / f"rmsd_lig_{trial}.xvg" for trial in range(1, 6)]
    missing = [path.name for path in trial_paths if not path.exists()]
    if missing:
        missing_names = ", ".join(missing)
        raise FileNotFoundError(f"Missing RMSD files in {rmsd_dir}: {missing_names}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
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

    max_time = max(float(np.max(times)) for times in all_times)
    max_rmsd = max(float(np.max(series)) for series in all_series)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max(1.5, max_rmsd * 1.05))
    style_axes(ax)
    fig.suptitle(channel_name, fontsize=TITLE_FONT_SIZE, fontweight="bold", x=0.55, y=0.96)
    fig.tight_layout(rect=(0.02, 0.01, 1, 0.985), pad=0.2)

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