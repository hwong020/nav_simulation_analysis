"""Plot contact occupancy heatmaps from per-trial CSV probability files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _format_channel_name(channel: str) -> str:
    """Normalize channel naming for plot titles."""
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
    # Collect per-trial contact probability CSVs in a stable order.
    csv_files = sorted(folder.glob("contact_probability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No contact_probability_*.csv found in {folder}")

    trial_labels: list[str] = []
    trial_data: list[np.ndarray] = []
    residue_ids: np.ndarray | None = None

    for csv_path in csv_files:
        # Load each trial and validate required columns.
        df = pd.read_csv(csv_path)
        if "Probability" not in df.columns:
            raise ValueError(f"Missing 'Probability' column in {csv_path}")
        trial_label = csv_path.stem.replace("contact_probability_", "trial ")
        trial_labels.append(trial_label.capitalize())
        trial_data.append(df["Probability"].to_numpy())
        if residue_ids is None:
            # The first CSV defines the residue ID axis; later files must match.
            if "ResID" not in df.columns:
                raise ValueError(f"Missing 'ResID' column in {csv_path}")
            residue_ids = df["ResID"].to_numpy()
        elif "ResID" in df.columns and not np.array_equal(residue_ids, df["ResID"].to_numpy()):
            raise ValueError(f"Residue IDs differ across files; check {csv_path}")

    # Stack trial vectors into a matrix with shape (n_trials, n_residues).
    matrix = np.vstack(trial_data)
    if residue_ids is None:
        raise ValueError("Residue IDs could not be determined from CSV files")

    # Expand to a full residue index (fill missing residues with zeros).
    min_resid = int(residue_ids.min())
    max_resid = int(residue_ids.max())
    full_residues = np.arange(min_resid, max_resid + 1)
    resid_to_index = {resid: idx for idx, resid in enumerate(full_residues)}
    expanded_matrix = np.zeros((matrix.shape[0], full_residues.size), dtype=float)
    for row_idx, row in enumerate(matrix):
        for resid, prob in zip(residue_ids, row, strict=False):
            expanded_matrix[row_idx, resid_to_index[int(resid)]] = prob

    return trial_labels, expanded_matrix, full_residues


def plot_average_heatmap(
    averaged: np.ndarray,
    residue_ids: np.ndarray,
    output_path: Path,
    channel: str,
) -> None:
    """Plot a single-row heatmap averaged across trials."""
    # Standardize plot styling so figures are consistent across runs.
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 3.2))

    vmin = 0.0
    vmax = float(np.nanmax(averaged)) if averaged.size else 1.0

    channel_label = _format_channel_name(channel)
    data = averaged[None, :]
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=(residue_ids[0], residue_ids[-1], 0, 1),
    )
    ax.set_title(f"{channel_label} Trial-Averaged Contact Occupancy")
    ax.set_yticks([])
    ax.set_xlabel("Residue ID", labelpad=10)

    fig.subplots_adjust(right=0.88, top=0.84)
    cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label="Contact Occupancy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Entry point for generating contact occupancy heatmaps."""
    data_folder = Path("src/nav1-2/contact_occupancies")
    output_path = Path("results/nav1-2/graphs/contact_occupancy_heatmaps.png")
    channel = data_folder.parent.name

    _, matrix, residue_ids = load_contact_probabilities(data_folder)
    averaged = matrix.mean(axis=0) if matrix.size else np.array([])
    plot_average_heatmap(averaged, residue_ids, output_path, channel)
    print(f"Saved averaged heatmap to {output_path}")


if __name__ == "__main__":
    main()