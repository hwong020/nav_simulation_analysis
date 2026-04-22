"""Generate mean ± SD RMSF plots from five XVG trials per channel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (3.8, 2.35)
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 7

MEAN_COLOR = "#0f766e"
SHADE_COLOR = "#5eead4"

DOMAIN_STYLES = {
    "DI": {"start": 1, "end": 423, "color": "#fde68a"},
    "DII": {"start": 423, "end": 992, "color": "#bfdbfe"},
    "DIII": {"start": 993, "end": 1483, "color": "#c7d2fe"},
    "DIV": {"start": 1484, "end": 2009, "color": "#fecdd3"},
}


def load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load residue/RMSF columns from a GROMACS XVG file."""
    with path.open("r", encoding="utf-8") as handle:
        lines = [
            line for line in handle if line.strip() and not line.startswith("#") and not line.startswith("@")
        ]

    if not lines:
        raise ValueError(f"No data found in {path}")

    data = np.genfromtxt(lines, invalid_raise=False)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] < 2:
        raise ValueError(f"No valid RMSF data found in {path}")

    data = data[:, :2]
    data = data[~np.isnan(data).any(axis=1)]
    if data.size == 0:
        raise ValueError(f"No valid RMSF rows found in {path}")

    residues = data[:, 0]
    rmsf_angstrom = data[:, 1] * 10.0
    return residues, rmsf_angstrom


def collect_trials(rmsf_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load five RMSF trials and return residue, mean, and standard deviation."""
    trial_paths = [rmsf_dir / f"rmsf_ca_{trial}.xvg" for trial in range(1, 6)]
    missing = [path.name for path in trial_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RMSF files in {rmsf_dir}: {', '.join(missing)}")

    residue_sets: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    for path in trial_paths:
        residues, values = load_xvg(path)
        residue_sets.append(residues)
        traces.append(values)

    reference = residue_sets[0]
    for residues in residue_sets[1:]:
        if not np.array_equal(reference, residues):
            raise ValueError(f"Residue axis mismatch among RMSF trials in {rmsf_dir}")

    stacked = np.vstack(traces)
    return reference, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def style_axes(ax: plt.Axes) -> None:
    """Apply compact template-like styling."""
    ax.set_xlabel("Residues", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("RMSF (Å)", fontsize=AXIS_LABEL_FONT_SIZE)
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


def add_nav11_domain_highlights(ax: plt.Axes) -> None:
    """Add Nav1.1 domain background spans and centered domain labels."""
    x_min, x_max = ax.get_xlim()
    y_top = ax.get_ylim()[1]
    for label, spec in DOMAIN_STYLES.items():
        ax.axvspan(spec["start"], spec["end"], color=spec["color"], alpha=0.28, zorder=0)
        visible_start = max(spec["start"], x_min)
        visible_end = min(spec["end"], x_max)
        if visible_end <= visible_start:
            continue
        x_center = (visible_start + visible_end) / 2
        ax.text(
            x_center,
            y_top * 0.965,
            label,
            ha="center",
            va="top",
            fontsize=LEGEND_FONT_SIZE + 1,
            fontweight="bold",
            color="#334155",
            zorder=3,
        )


def plot_channel(channel_dir: Path, output_dir: Path) -> None:
    """Create the RMSF mean ± SD plot for one channel."""
    residues, mean_rmsf, std_rmsf = collect_trials(channel_dir / "rmsf")
    channel_name = channel_dir.name.replace("nav1-", "Nav1.")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.fill_between(
        residues,
        mean_rmsf - std_rmsf,
        mean_rmsf + std_rmsf,
        color=SHADE_COLOR,
        alpha=0.45,
        linewidth=0,
        zorder=1,
    )
    ax.plot(residues, mean_rmsf, color=MEAN_COLOR, linewidth=1.8, zorder=2)

    ax.set_xlim(float(residues.min()), float(residues.max()))
    ymax = max(float(np.max(mean_rmsf + std_rmsf)) * 1.08, 1.0)
    ax.set_ylim(0, ymax)
    if channel_dir.name == "nav1-1":
        add_nav11_domain_highlights(ax)
    style_axes(ax)
    fig.suptitle(channel_name, fontsize=TITLE_FONT_SIZE, fontweight="bold", x=0.5, y=0.965)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.985), pad=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"rmsf_{channel_dir.name}_mean_sd.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run RMSF plotting for all channels that contain RMSF trial data."""
    src_root = Path("src")
    results_root = Path("results")

    for channel_dir in sorted(src_root.glob("nav1-*")):
        rmsf_dir = channel_dir / "rmsf"
        if not rmsf_dir.exists():
            print(f"Skipping {channel_dir.name}: no rmsf directory")
            continue

        try:
            plot_channel(channel_dir, results_root / channel_dir.name / "graphs")
            print(f"Generated RMSF plot for {channel_dir.name}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel_dir.name}: {exc}")


if __name__ == "__main__":
    main()