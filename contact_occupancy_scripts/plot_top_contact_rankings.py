"""Generate per-channel top-contact ranking plots with mean ± standard deviation."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    from .combined_contact_heatmap import RING_RESIDUES, _format_channel_name
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from contact_occupancy_scripts.combined_contact_heatmap import (  # type: ignore
        RING_RESIDUES,
        _format_channel_name,
    )


RESULTS_ROOT = Path("results")
DATA_ROOT = Path("src")
CHANNELS = sorted(
    path.name for path in RESULTS_ROOT.glob("nav1-*") if path.is_dir()
)
TOP_N = 8
TITLE_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 11
X_TICK_FONT_SIZE = 6
Y_TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 6.5

FIGURE_SIZE = (4.2, 2.4)
GRID_LINE_WIDTH = 0.45
GRID_ALPHA = 0.18

OUTER_COLOR = "#f28e2b"
INNER_COLOR = "#4e79a7"
OTHER_COLOR = "#9d9da1"


def _load_contact_probability_tables(channel: str) -> list[pd.DataFrame]:
    """Load all per-trial contact probability tables for a channel."""
    folder = DATA_ROOT / channel / "contact_occupancies"
    csv_files = sorted(folder.glob("contact_probability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No contact probability CSVs found in {folder}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        required_cols = {"ResID", "Resname", "Probability"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}")
        frames.append(df.loc[:, ["ResID", "Resname", "Probability"]].copy())
    return frames


def _build_summary(channel: str) -> pd.DataFrame:
    """Return mean and standard deviation of contact probabilities per residue."""
    trial_tables = _load_contact_probability_tables(channel)
    merged: pd.DataFrame | None = None

    for trial_idx, df in enumerate(trial_tables, start=1):
        trial_df = df.rename(columns={"Probability": f"trial_{trial_idx}"})
        if merged is None:
            merged = trial_df
        else:
            merged = merged.merge(trial_df, on=["ResID", "Resname"], how="outer")

    if merged is None:
        raise ValueError(f"No trial tables loaded for {channel}")

    trial_cols = [col for col in merged.columns if col.startswith("trial_")]
    merged[trial_cols] = merged[trial_cols].fillna(0.0)
    merged["mean_probability"] = merged[trial_cols].mean(axis=1)
    merged["std_probability"] = merged[trial_cols].std(axis=1, ddof=1).fillna(0.0)

    return merged.sort_values("mean_probability", ascending=False).head(TOP_N).copy()


def _classify_residue(channel: str, resid: int) -> str:
    """Return whether a residue belongs to the outer ring, inner ring, or neither."""
    ring_map = RING_RESIDUES.get(channel, {})
    if resid in ring_map.get("outer", set()):
        return "Outer ring"
    if resid in ring_map.get("inner", set()):
        return "Inner ring"
    return "Other"


def _bar_color(category: str) -> str:
    """Map residue category to bar color."""
    if category == "Outer ring":
        return OUTER_COLOR
    if category == "Inner ring":
        return INNER_COLOR
    return OTHER_COLOR


def plot_top_contacts(channel: str) -> Path:
    """Generate the top-8 contact probability plot for a channel."""
    summary = _build_summary(channel)
    summary["category"] = [
        _classify_residue(channel, int(resid)) for resid in summary["ResID"]
    ]
    summary["label"] = [
        f"{resname}{int(resid)}" for resname, resid in zip(summary["Resname"], summary["ResID"], strict=False)
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(summary))
    colors = [_bar_color(category) for category in summary["category"]]

    ax.bar(
        x,
        summary["mean_probability"],
        yerr=summary["std_probability"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
    )

    ax.set_title(
        f"{_format_channel_name(channel)}",
        fontsize=TITLE_FONT_SIZE,
        pad=6,
        fontweight='bold',
        x = 0.45,
    )
    ax.set_xlabel("Residue", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=4, fontweight="bold", x = 0.45)
    ax.set_ylabel(
        "Contact Probability",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=4,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE, pad=2, width=0.8, length=3)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, pad=2, width=0.8, length=3)
    ax.set_ylim(bottom=0)
    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
        color="#b3b3b3",
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#666666")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    legend_handles = [
        Patch(facecolor=OUTER_COLOR, edgecolor="black", label="Outer Ring (EEDD)"),
        Patch(facecolor=INNER_COLOR, edgecolor="black", label="Inner Ring (DEKA)"),
    ]
    if (summary["category"] == "Other").any():
        legend_handles.append(
            Patch(facecolor=OTHER_COLOR, edgecolor="black", label="Other Residues")
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor="#666666",
        framealpha=0.95,
        borderpad=0.3,
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.35,
        fancybox=False,
        columnspacing=0.7,
    )

    output_dir = RESULTS_ROOT / channel / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "top8_contact_probability_mean_std.png"
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate ranking plots for all available channels."""
    generated: list[Path] = []
    skipped: list[str] = []
    for channel in CHANNELS:
        try:
            output_path = plot_top_contacts(channel)
        except FileNotFoundError:
            skipped.append(channel)
            continue
        generated.append(output_path)
        print(f"Saved {output_path}")

    if skipped:
        print("Skipped channels with no contact occupancy tables:", ", ".join(skipped))


if __name__ == "__main__":
    main()