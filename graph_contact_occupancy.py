from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    return trial_labels, matrix, residue_ids


def plot_trial_heatmaps(
    trial_labels: list[str],
    matrix: np.ndarray,
    residue_ids: np.ndarray,
    output_path: Path,
) -> None:
    """Plot per-trial heatmaps on a single figure with a shared colorbar."""
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 16,
        }
    )
    n_trials, n_residues = matrix.shape
    n_cols = 1
    n_rows = n_trials

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, max(2.4 * n_rows, 4)),
        squeeze=False,
    )

    vmin = 0.0
    vmax = float(np.nanmax(matrix)) if matrix.size else 1.0

    im = None
    for idx, label in enumerate(trial_labels):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        data = matrix[idx][None, :]
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=(residue_ids[0], residue_ids[-1], 0, 1),
        )
        ax.set_title(label)
        ax.set_yticks([])
        ax.set_xlabel("Residue ID", labelpad=10)

    # Hide unused axes
    for idx in range(n_trials, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis("off")

    if im is not None:
        fig.subplots_adjust(right=0.84, hspace=0.7, top=0.92)
        cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
        fig.colorbar(im, cax=cbar_ax, label="Contact Occupancy")

    fig.suptitle("Contact Occupancy Heatmaps per Trial", fontsize=16, y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    data_folder = Path("src/nav1-3/contact_occupancies")
    output_path = Path("results/nav1-3/graphs/contact_occupancy_heatmaps.png")

    trial_labels, matrix, residue_ids = load_contact_probabilities(data_folder)
    plot_trial_heatmaps(trial_labels, matrix, residue_ids, output_path)
    print(f"Saved heatmap grid to {output_path}")


if __name__ == "__main__":
    main()