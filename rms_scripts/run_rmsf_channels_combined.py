"""Generate grouped Flooding vs Single Ligand RMSF bar charts per Nav channel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (7.2, 5.0)
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 16
X_TICK_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16

Y_AXIS_MAX = 12.0
RESIDUE_GROUP_SPACING = 1.15
BAR_WIDTH = 0.32

# Same shades used in the previous RMSF scripts.
FLOODING_COLOR = "#fde68a"       # yellow
SINGLE_LIGAND_COLOR = "#59a14f"  # green
CHANNELS_WITHOUT_FLOODING = {"nav1-5"}

TOP8_CONTACT_RESIDUES = {
    "nav1-1": [328, 329, 383, 385, 386, 954, 1412, 1436],
    "nav1-2": [385, 387, 388, 942, 945, 1426, 1715, 1717],
    "nav1-3": [384, 386, 918, 1397, 1398, 1418, 1421, 1708],
    "nav1-5": [310, 317, 322, 373, 375, 376, 901, 1714],
}

TOP8_CONTACT_LABELS = {
    "nav1-1": ["F383", "D1436", "E385", "N386", "E954", "K1412", "E328", "G329"],
    "nav1-2": ["E387", "F385", "E945", "G1715", "D1426", "D1717", "E942", "N388"],
    "nav1-3": ["E386", "K1397", "D1421", "Y384", "G1418", "V1398", "D918", "S1708"],
    "nav1-5": ["R376", "E375", "E901", "C373", "D322", "D1714", "K317", "D310"],
}


def _parse_residue_number(label: str) -> int:
    """Extract residue number from a label like 'F383'."""
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits:
        raise ValueError(f"No residue number in label: {label}")
    return int(digits)


def _label_map_for_channel(channel_key: str) -> dict[int, str]:
    """Build residue-id -> label mapping independent of input order."""
    labels = TOP8_CONTACT_LABELS.get(channel_key, [])
    mapping: dict[int, str] = {}
    for label in labels:
        mapping[_parse_residue_number(label)] = label
    return mapping


def load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load residue/RMSF columns from a GROMACS XVG file."""
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
        raise ValueError(f"No valid RMSF data found in {path}")

    data = data[:, :2]
    data = data[~np.isnan(data).any(axis=1)]
    if data.size == 0:
        raise ValueError(f"No valid RMSF rows found in {path}")

    residues = data[:, 0]
    rmsf_angstrom = data[:, 1] * 10.0
    return residues, rmsf_angstrom


def collect_trials(rmsf_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load available RMSF trials and return residue axis + stacked trial values."""
    trial_paths = sorted(rmsf_dir.glob("rmsf_ca_*.xvg"))
    if not trial_paths:
        raise FileNotFoundError(f"No RMSF files found in {rmsf_dir}")

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
    return reference, stacked


def style_axes(ax: plt.Axes) -> None:
    """Apply compact RMSF plot styling."""
    ax.set_xlabel("Residues", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Average RMSF (Å)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=1.0, length=3)
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def mean_and_std_for_residues(
    residues: np.ndarray,
    trial_values: np.ndarray,
    chosen: list[int],
) -> tuple[list[float], list[float]]:
    """Return per-residue trial mean and sample standard deviation."""
    residue_to_idx = {int(res): i for i, res in enumerate(residues)}
    means: list[float] = []
    stds: list[float] = []

    for resid in chosen:
        values = trial_values[:, residue_to_idx[resid]]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    return means, stds


def plot_channel(channel_dir: Path, output_dir: Path) -> None:
    """Create one grouped Flooding vs Single Ligand RMSF bar chart for a channel."""
    include_flooding = channel_dir.name not in CHANNELS_WITHOUT_FLOODING
    flooding_residues = None
    flooding_trial_values = None
    if include_flooding:
        flooding_residues, flooding_trial_values = collect_trials(channel_dir / "rmsf_multiple_lig")
    single_ligand_residues, single_ligand_trial_values = collect_trials(channel_dir / "rmsf_single_lig")

    if include_flooding and not np.array_equal(flooding_residues, single_ligand_residues):
        raise ValueError(f"Residue axis mismatch between flooding/single ligand RMSF data for {channel_dir.name}")

    top_res = TOP8_CONTACT_RESIDUES.get(channel_dir.name)
    if not top_res:
        raise ValueError(f"No top-8 residue mapping provided for {channel_dir.name}")

    residue_to_idx = {int(res): i for i, res in enumerate(single_ligand_residues)}
    chosen = sorted(res for res in top_res if res in residue_to_idx)
    if not chosen:
        raise ValueError(f"None of top-8 residues found in RMSF axis for {channel_dir.name}")

    single_ligand_means, single_ligand_stds = mean_and_std_for_residues(
        single_ligand_residues,
        single_ligand_trial_values,
        chosen,
    )
    if include_flooding:
        flooding_means, flooding_stds = mean_and_std_for_residues(
            flooding_residues,
            flooding_trial_values,
            chosen,
        )

    x = np.arange(len(chosen), dtype=float) * RESIDUE_GROUP_SPACING
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    error_kw = {"elinewidth": 1.4, "capsize": 3, "capthick": 1.4, "ecolor": "#111827"}

    single_ligand_x = x
    if include_flooding:
        ax.bar(
            x - BAR_WIDTH / 2,
            flooding_means,
            width=BAR_WIDTH,
            yerr=flooding_stds,
            color=FLOODING_COLOR,
            edgecolor="#111827",
            linewidth=0.8,
            label="Flooding (First 500 ns)",
            error_kw=error_kw,
            zorder=3,
        )
        single_ligand_x = x + BAR_WIDTH / 2

    ax.bar(
        single_ligand_x,
        single_ligand_means,
        width=BAR_WIDTH,
        yerr=single_ligand_stds,
        color=SINGLE_LIGAND_COLOR,
        edgecolor="#111827",
        linewidth=0.8,
        label="Single Ligand",
        error_kw=error_kw,
        zorder=3,
    )

    label_map = _label_map_for_channel(channel_dir.name)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_map.get(r, str(r)) for r in chosen],
        fontsize=X_TICK_LABEL_FONT_SIZE,
        rotation=0,
        ha="center",
        va="top",
    )
    ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
    ax.set_ylim(0, Y_AXIS_MAX)

    channel_name = channel_dir.name.replace("nav1-", "Nav1.")
    title_suffix = "Flooding (First 500 ns) vs Single Ligand" if include_flooding else "Single Ligand"
    ax.set_title(f"{channel_name}: {title_suffix}", fontsize=TITLE_FONT_SIZE, pad=10)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.92),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        borderaxespad=0.0,
    )
    style_axes(ax)
    fig.tight_layout(rect=(0.005, 0.08, 0.995, 0.965), pad=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"rmsf_{channel_dir.name}_flooding_vs_single_ligand_bar_mean_sd.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def main() -> None:
    """Run combined Flooding vs Single Ligand RMSF bar plotting for mapped Nav channels."""
    src_root = Path("src")
    results_root = Path("results")

    for channel_key in sorted(TOP8_CONTACT_RESIDUES):
        channel_dir = src_root / channel_key
        if not channel_dir.exists():
            print(f"Skipping {channel_key}: no channel directory")
            continue
        if channel_key not in CHANNELS_WITHOUT_FLOODING and not (channel_dir / "rmsf_multiple_lig").exists():
            print(f"Skipping {channel_key}: no rmsf_multiple_lig directory")
            continue
        if not (channel_dir / "rmsf_single_lig").exists():
            print(f"Skipping {channel_key}: no rmsf_single_lig directory")
            continue

        try:
            plot_channel(channel_dir, results_root / channel_key / "graphs")
            print(f"Generated combined RMSF bar plot for {channel_key}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel_key}: {exc}")


if __name__ == "__main__":
    main()