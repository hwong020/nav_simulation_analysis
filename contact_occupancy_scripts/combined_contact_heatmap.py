"""Create a combined contact occupancy heatmap across all Nav channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHANNELS = [f"nav1-{idx}" for idx in range(1, 10)]
DATA_ROOT = Path("src")
OUTPUT_PATH = Path("results/contact_occupancy_all_channels.png")

RING_RESIDUES = {
    "nav1-1": {
        "outer": {385, 954, 1436, 1727},
        "inner": {382, 951, 1432, 1724},
    },
    "nav1-2": {
        "outer": {387, 945, 1426, 1717},
        "inner": {384, 942, 1422, 1714},
    },
    "nav1-3": {
        "outer": {386, 946, 1421, 1712},
        "inner": {383, 943, 1417, 1709},
    },
    "nav1-4": {
        "outer": {409, 764, 1248, 1539},
        "inner": {406, 761, 1244, 1536},
    },
    "nav1-5": {
        "outer": {375, 901, 1423, 1714},
        "inner": {372, 898, 1419, 1711},
    },
    "nav1-6": {
        "outer": {373, 939, 1417, 1708},
        "inner": {370, 936, 1413, 1705},
    },
    "nav1-7": {
        "outer": {364, 930, 1410, 1701},
        "inner": {361, 927, 1406, 1698},
    },
    "nav1-8": {
        "outer": {359, 852, 1371, 1664},
        "inner": {356, 849, 1367, 1661},
    },
    "nav1-9": {
        "outer": {362, 771, 1261, 1554},
        "inner": {359, 768, 1257, 1551},
    },
}


def _format_channel_name(channel: str) -> str:
    """Normalize channel naming for plot labels."""
    channel = channel.lower()
    if channel.startswith("nav1-"):
        return channel.replace("nav1-", "Nav1.")
    return channel


def load_contact_probabilities(folder: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Load all contact probability CSVs in a folder.

    Returns a tuple of (labels, matrix, residue_ids) where:
    - labels: list like ["trial 1", "trial 2", ...]
    - matrix: shape (n_trials, n_residues) probabilities
    - residue_ids: residue IDs from the CSV (used for x-axis scale)
    """
    csv_files = sorted(folder.glob("contact_probability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No contact_probability_*.csv found in {folder}")

    trial_labels: list[str] = []
    trial_data: list[np.ndarray] = []
    residue_ids: np.ndarray | None = None

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "Probability" not in df.columns:
            raise ValueError(f"Missing 'Probability' column in {csv_path}")
        trial_label = csv_path.stem.replace("contact_probability_", "trial ")
        trial_labels.append(trial_label.capitalize())
        trial_data.append(df["Probability"].to_numpy())
        if residue_ids is None:
            if "ResID" not in df.columns:
                raise ValueError(f"Missing 'ResID' column in {csv_path}")
            residue_ids = df["ResID"].to_numpy()
        elif "ResID" in df.columns and not np.array_equal(residue_ids, df["ResID"].to_numpy()):
            raise ValueError(f"Residue IDs differ across files; check {csv_path}")

    matrix = np.vstack(trial_data)
    if residue_ids is None:
        raise ValueError("Residue IDs could not be determined from CSV files")

    min_resid = int(residue_ids.min())
    max_resid = int(residue_ids.max())
    full_residues = np.arange(min_resid, max_resid + 1)
    resid_to_index = {resid: idx for idx, resid in enumerate(full_residues)}
    expanded_matrix = np.zeros((matrix.shape[0], full_residues.size), dtype=float)
    for row_idx, row in enumerate(matrix):
        for resid, prob in zip(residue_ids, row, strict=False):
            expanded_matrix[row_idx, resid_to_index[int(resid)]] = prob

    return trial_labels, expanded_matrix, full_residues


def load_channel_average(channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and average contact probabilities for a channel."""
    data_folder = DATA_ROOT / channel / "contact_occupancies"
    _, matrix, residues = load_contact_probabilities(data_folder)
    averaged = matrix.mean(axis=0) if matrix.size else np.array([])
    return averaged, residues


def align_to_residues(
    averaged: np.ndarray, residues: np.ndarray, global_residues: np.ndarray
) -> np.ndarray:
    """Align a channel's averaged data to the global residue index."""
    aligned = np.zeros(global_residues.size, dtype=float)
    resid_to_index = {resid: idx for idx, resid in enumerate(global_residues)}
    for resid, value in zip(residues, averaged, strict=False):
        aligned[resid_to_index[int(resid)]] = value
    return aligned


def build_combined_matrix(channels: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load all channels and return a stacked matrix + global residue axis."""
    channel_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    min_resid = None
    max_resid = None

    for channel in channels:
        try:
            averaged, residues = load_channel_average(channel)
        except FileNotFoundError:
            skipped.append(channel)
            continue
        channel_data.append((channel, averaged, residues))
        min_resid = int(residues.min()) if min_resid is None else min(min_resid, int(residues.min()))
        max_resid = int(residues.max()) if max_resid is None else max(max_resid, int(residues.max()))

    if min_resid is None or max_resid is None or not channel_data:
        raise ValueError("No channels contained contact occupancy data; nothing to plot.")

    global_residues = np.arange(min_resid, max_resid + 1)
    stacked = np.vstack(
        [align_to_residues(averaged, residues, global_residues) for _, averaged, residues in channel_data]
    )
    kept_channels = [channel for channel, _, _ in channel_data]
    return stacked, global_residues, kept_channels, skipped


def plot_combined_heatmap(
    matrix: np.ndarray, residues: np.ndarray, channels: list[str], output_path: Path
) -> None:
    """Plot one heatmap bar per channel with individual x-axes."""
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.titlesize": 20,
        }
    )
    fig_height = max(10.0, 2.3 * len(channels))
    fig, axes = plt.subplots(
        nrows=len(channels),
        ncols=1,
        figsize=(20, fig_height),
        sharex=False,
    )

    if len(channels) == 1:
        axes = [axes]

    vmin = 0.0
    vmax = float(np.nanmax(matrix)) if matrix.size else 1.0

    for idx, (ax, channel) in enumerate(zip(axes, channels, strict=False)):
        row = matrix[idx][None, :]
        ax.imshow(
            row,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=(residues[0], residues[-1], 0, 1),
        )
        ax.set_yticks([])
        ax.set_ylabel(_format_channel_name(channel), rotation=0, labelpad=35, va="center")
        ax.set_xlabel("Residue ID")

    fig.suptitle("Trial-Averaged Contact Occupancy Across Nav Channels", y=0.99)
    fig.subplots_adjust(right=0.9, hspace=1.4, top=0.9, bottom=0.12)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(axes[0].images[0], cax=cbar_ax, label="Contact Occupancy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Entry point for generating the combined contact occupancy heatmap."""
    matrix, residues, kept_channels, skipped = build_combined_matrix(CHANNELS)
    plot_combined_heatmap(matrix, residues, kept_channels, OUTPUT_PATH)
    if skipped:
        skipped_label = ", ".join(_format_channel_name(ch) for ch in skipped)
        print(f"Skipped channels with no contact data: {skipped_label}")
    print(f"Saved combined heatmap to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()