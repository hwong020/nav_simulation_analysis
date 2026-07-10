"""Generate per-channel top-contact ranking plots with mean ± standard deviation."""

from __future__ import annotations

from pathlib import Path
import re
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
TOP_CSV_N = 30
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 16
X_TICK_FONT_SIZE = 14
Y_TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14

FIGURE_SIZE = (8.0, 4.8)
GRID_LINE_WIDTH = 0.5
GRID_ALPHA = 0.18

OUTER_COLOR = "#f28e2b"
INNER_COLOR = "#4e79a7"
OTHER_COLOR = "#FFFF00"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

CUSTOM_TOP8_RESIDUES = {
    "nav1-1": ["F383", "D1436", "E385", "N386", "E954", "K1412", "E328", "G329"],
    "nav1-2": ["E387", "F385", "E945", "G1715", "D1426", "D1717", "E942", "N388"],
    "nav1-3": ["E386", "K1397", "D1421", "Y384", "G1418", "V1398", "D918", "S1708"],
    "nav1-5": ["R376", "E375", "E901", "C373", "D322", "D1714", "K317", "D310"],
}


def _parse_residue_number(label: str) -> int:
    """Extract numeric residue id from a label like 'F383'."""
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits:
        raise ValueError(f"No residue number in label: {label}")
    return int(digits)


def _one_letter_label(resname: str, resid: int) -> str:
    """Convert 3-letter residue name + id to 1-letter label (e.g., PHE,383 -> F383)."""
    one = THREE_TO_ONE.get(str(resname).upper(), str(resname).upper()[:1])
    return f"{one}{int(resid)}"


def _load_contact_probability_tables(channel: str) -> list[tuple[str, pd.DataFrame]]:
    """Load all per-trial contact probability tables for a channel."""
    folder = DATA_ROOT / channel / "contact_occupancies"
    csv_files = sorted(folder.glob("contact_probability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No contact probability CSVs found in {folder}")

    frames: list[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        required_cols = {"ResID", "Resname", "Probability"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}")
        trial_label = csv_path.stem.replace("contact_probability_", "trial")
        frames.append((trial_label, df.loc[:, ["ResID", "Resname", "Probability"]].copy()))
    return frames


def _build_summary(channel: str) -> pd.DataFrame:
    """Return mean and standard deviation of contact probabilities per residue."""
    trial_tables = _load_contact_probability_tables(channel)
    merged: pd.DataFrame | None = None

    for trial_idx, (_, df) in enumerate(trial_tables, start=1):
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

    return merged.sort_values("mean_probability", ascending=False).copy()


def export_top_contacts_per_trial(channel: str) -> list[Path]:
    """Export top contact occupancy residues (top-30) for each trial as CSV."""
    trial_tables = _load_contact_probability_tables(channel)
    output_dir = RESULTS_ROOT / channel / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    for trial_label, df in trial_tables:
        ranked = df.sort_values("Probability", ascending=False).head(TOP_CSV_N).copy()
        ranked["Rank"] = np.arange(1, len(ranked) + 1)
        ranked["Residue"] = [
            f"{resname}{int(resid)}"
            for resname, resid in zip(ranked["Resname"], ranked["ResID"], strict=False)
        ]
        ranked["category"] = [
            _classify_residue(channel, int(resid)) for resid in ranked["ResID"]
        ]

        trial_suffix = re.sub(r"[^0-9a-zA-Z]+", "", trial_label.lower())
        output_path = output_dir / f"top30_contact_occupancy_{trial_suffix}.csv"
        ranked.rename(columns={"Probability": "probability"}, inplace=True)
        ranked.loc[
            :,
            [
                "Rank",
                "ResID",
                "Resname",
                "Residue",
                "category",
                "probability",
            ],
        ].to_csv(output_path, index=False)
        exported_paths.append(output_path)

    return exported_paths


def export_average_contact_ranking(channel: str) -> Path:
    """Export full residue ranking by mean contact probability across trials."""
    summary = _build_summary(channel).copy()
    summary["Rank"] = np.arange(1, len(summary) + 1)
    summary["Residue"] = [
        _one_letter_label(resname, int(resid))
        for resname, resid in zip(summary["Resname"], summary["ResID"], strict=False)
    ]
    summary["category"] = [
        _classify_residue(channel, int(resid)) for resid in summary["ResID"]
    ]

    output_dir = RESULTS_ROOT / channel / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "average_contact_occupancy_ranking.csv"
    summary.loc[
        :,
        [
            "Rank",
            "ResID",
            "Resname",
            "Residue",
            "category",
            "mean_probability",
            "std_probability",
        ],
    ].to_csv(output_path, index=False)
    return output_path


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
    summary_all = _build_summary(channel).copy()
    summary_all["label"] = [
        _one_letter_label(resname, int(resid))
        for resname, resid in zip(summary_all["Resname"], summary_all["ResID"], strict=False)
    ]

    custom_labels = CUSTOM_TOP8_RESIDUES.get(channel)
    if not custom_labels:
        summary = summary_all.head(TOP_N).copy()
    else:
        by_resid = summary_all.set_index("ResID", drop=False)
        rows: list[pd.Series] = []
        missing: list[str] = []
        for label in custom_labels:
            resid = _parse_residue_number(label)
            if resid in by_resid.index:
                rows.append(by_resid.loc[resid])
            else:
                missing.append(label)
        if missing:
            print(f"Warning: {channel} missing residues from custom list: {', '.join(missing)}")
        if not rows:
            raise ValueError(f"No custom residues found for {channel}")
        summary = pd.DataFrame(rows).reset_index(drop=True)

    summary["category"] = [
        _classify_residue(channel, int(resid)) for resid in summary["ResID"]
    ]
    if custom_labels:
        summary["label"] = custom_labels[: len(summary)]
    else:
        summary["label"] = [
            _one_letter_label(resname, int(resid))
            for resname, resid in zip(summary["Resname"], summary["ResID"], strict=False)
        ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(summary)) * 0.86
    colors = [_bar_color(category) for category in summary["category"]]
    mean_percent = summary["mean_probability"] * 100.0
    std_percent = summary["std_probability"] * 100.0
    yerr = np.vstack([np.minimum(std_percent, mean_percent), std_percent])

    bar_container = ax.bar(
        x,
        mean_percent,
        yerr=yerr,
        color=colors,
        edgecolor="#000000",
        linewidth=0.9,
        width=0.52,
        capsize=6,
        error_kw={"ecolor": "#000000", "elinewidth": 1.0, "capthick": 1.0},
    )
    if bar_container.errorbar is not None:
        errorbar_lines = bar_container.errorbar.lines
        for artist in errorbar_lines:
            if artist is None:
                continue
            if isinstance(artist, (list, tuple)):
                for sub_artist in artist:
                    sub_artist.set_clip_on(False)
            else:
                artist.set_clip_on(False)

    ax.set_title(
        f"{_format_channel_name(channel)}",
        fontsize=TITLE_FONT_SIZE,
        pad=6,
    )
    ax.set_xlabel("Residue", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=4)
    ax.set_ylabel(
        "Contact Occupancy (%)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=90, ha="center", va="top")
    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE, pad=1.0, width=0.8, length=3)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, pad=2, width=0.8, length=3)
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    legend_handles = [
        Patch(facecolor=OUTER_COLOR, edgecolor="none", label="Outer Ring (EEDD)"),
        Patch(facecolor=INNER_COLOR, edgecolor="none", label="Inner Ring (DEKA)"),
    ]
    if (summary["category"] == "Other").any():
        legend_handles.append(
            Patch(facecolor=OTHER_COLOR, edgecolor="none", label="Other Residues")
        )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        borderpad=0.3,
        borderaxespad=0.0,
        handlelength=1.35,
        handletextpad=0.45,
        fancybox=False,
        columnspacing=0.8,
    )

    output_dir = RESULTS_ROOT / channel / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "top8_contact_probability_mean_std.png"
    fig.tight_layout(rect=(0.01, 0.08, 0.83, 0.98), pad=0.02)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate ranking plots for all available channels."""
    generated: list[Path] = []
    exported_csvs: list[Path] = []
    exported_avg_rankings: list[Path] = []
    skipped: list[str] = []
    for channel in CHANNELS:
        try:
            csv_paths = export_top_contacts_per_trial(channel)
            avg_ranking_path = export_average_contact_ranking(channel)
            output_path = plot_top_contacts(channel)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel}: {exc}")
            skipped.append(channel)
            continue
        exported_csvs.extend(csv_paths)
        exported_avg_rankings.append(avg_ranking_path)
        generated.append(output_path)
        print(f"Saved {output_path}")
        print(f"Saved {avg_ranking_path}")
        for csv_path in csv_paths:
            print(f"Saved {csv_path}")

    if skipped:
        print("Skipped channels with no contact occupancy tables:", ", ".join(skipped))


if __name__ == "__main__":
    main()