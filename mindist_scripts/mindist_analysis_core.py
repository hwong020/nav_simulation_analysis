"""Core plotting utilities for minimum-distance (mindist) analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MindistScenario:
    """Configuration for a single channel/scenario mindist analysis."""
    name: str
    input_dir: Path
    output_dir: Path
    residues: list[str]
    trials: list[int]
    hbond_cutoff_angstrom: float = 3.5
    hydrophobic_cutoff_angstrom: float = 4.0


# Mapping of channel → DEKA residue labels for plot annotations.
DEKA_LABELS = {
    "nav1-1": ["D382", "E951", "K1432", "A1724"],
    "nav1-2": ["D384", "E942", "K1422", "A1714"],
    "nav1-3": ["D383", "E943", "K1417", "A1709"],
    "nav1-4": ["D406", "E761", "K1244", "A1536"],
    "nav1-5": ["D372", "E898", "K1419", "A1711"],
    "nav1-6": ["D370", "E936", "K1413", "A1705"],
    "nav1-7": ["D361", "E927", "K1406", "A1698"],
    "nav1-8": ["D356", "E849", "K1367", "A1661"],
    "nav1-9": ["D359", "E768", "K1257", "A1551"],
}


def _format_channel_name(scenario_name: str) -> str:
    """Convert scenario name into a human-friendly channel label."""
    base = scenario_name.split("-")[0:2]
    if len(base) == 2 and base[0].lower() == "nav1":
        return f"Nav1.{base[1]}"
    return scenario_name


def _format_ligand_label(scenario_name: str) -> str:
    """Return a short ligand-count label based on the scenario name."""
    scenario_key = scenario_name.split("-", 2)[-1].lower()
    if "multiple" in scenario_key:
        return "10 TTX"
    if "single" in scenario_key:
        return "1 TTX"
    return scenario_key


def _get_residue_labels(scenario_name: str, residues: list[str]) -> list[str]:
    """Return DEKA labels for known channels or default to residue codes."""
    key = "-".join(scenario_name.split("-")[0:2]).lower()
    return DEKA_LABELS.get(key, [res.upper() for res in residues])


def _load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an XVG file, returning time and distance arrays."""
    data = np.loadtxt(path, comments=["@", "#"])
    return data[:, 0], data[:, 1]


def _collect_trial_series(
    scenario: MindistScenario, residue: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load all trial distance series for a residue and stack them."""
    times: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for trial in scenario.trials:
        # Each trial uses a filename like mindist_{residue}_{trial}.xvg
        filename = scenario.input_dir / f"mindist_{residue}_{trial}.xvg"
        if not filename.exists():
            raise FileNotFoundError(f"Missing {filename}")
        time, dist = _load_xvg(filename)
        times.append(time)
        dists.append(dist)
    time0 = times[0]
    for t in times[1:]:
        # Ensure the time axis is consistent across trials.
        if not np.array_equal(t, time0):
            raise ValueError(
                f"Time axis mismatch for residue {residue} in {scenario.input_dir}"
            )
    return time0, np.vstack(dists)


def plot_trial_time_series_overlay(scenario: MindistScenario) -> None:
    """Plot per-trial time series overlays for all residues in a scenario."""
    scenario.output_dir.mkdir(parents=True, exist_ok=True)
    channel_label = _format_channel_name(scenario.name)
    ligand_label = _format_ligand_label(scenario.name)
    residue_labels = _get_residue_labels(scenario.name, scenario.residues)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    scenario_suffix = scenario.name.split("-", 2)[-1]

    for trial in scenario.trials:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        time_ref: np.ndarray | None = None
        for idx, residue in enumerate(scenario.residues):
            filename = scenario.input_dir / f"mindist_{residue}_{trial}.xvg"
            if not filename.exists():
                raise FileNotFoundError(f"Missing {filename}")
            time, dist = _load_xvg(filename)
            dist_angstrom = dist * 10.0
            if time_ref is None:
                time_ref = time
            elif not np.array_equal(time_ref, time):
                raise ValueError(
                    f"Time axis mismatch for trial {trial} in {scenario.input_dir}"
                )

            try:
                residue_label = residue_labels[idx]
            except IndexError:
                residue_label = residue.upper()
            color = colors[idx % len(colors)]
            ax.plot(
                time,
                dist_angstrom,
                color=color,
                linewidth=2.0,
                label=residue_label,
            )

        ax.axhline(
            scenario.hbond_cutoff_angstrom,
            color="#d62728",
            linestyle="--",
            linewidth=2.2,
            alpha=0.9,
            zorder=5,
            label=f"H-bond cutoff ({scenario.hbond_cutoff_angstrom:.1f} Å)",
        )
        ax.axhline(
            scenario.hydrophobic_cutoff_angstrom,
            color="#000000",
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            zorder=4,
            label=f"Hydrophobic cutoff ({scenario.hydrophobic_cutoff_angstrom:.1f} Å)",
        )
        ax.set_title(
            f"{channel_label} ({ligand_label}): Trial {trial} Minimum Distance of DEKA residues"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance (Å)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 0.82, 1))

        output_path = (
            scenario.output_dir
            / f"mindist_{scenario.name}_trial{trial}_DEKA.png"
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def _fraction_in_contact(dist_angstrom: np.ndarray, cutoff: float) -> float:
    """Return fraction of distances within cutoff, guarding empty arrays."""
    if dist_angstrom.size == 0:
        return float("nan")
    return float(np.sum(dist_angstrom <= cutoff) / dist_angstrom.size)


def plot_occupancy_bars(scenario: MindistScenario) -> None:
    """Plot H-bond occupancy fractions for all residues in a scenario."""
    scenario.output_dir.mkdir(parents=True, exist_ok=True)

    residues = scenario.residues
    channel_label = _format_channel_name(scenario.name)
    ligand_label = _format_ligand_label(scenario.name)
    residue_labels = _get_residue_labels(scenario.name, residues)
    mean_fractions: list[float] = []
    std_fractions: list[float] = []

    for residue in residues:
        # Compute per-trial fractions of frames under the H-bond cutoff.
        time, dist_matrix = _collect_trial_series(scenario, residue)
        dist_angstrom = dist_matrix * 10.0
        fractions = [
            _fraction_in_contact(dist_angstrom[trial_idx], scenario.hbond_cutoff_angstrom)
            for trial_idx in range(dist_angstrom.shape[0])
        ]
        mean_fractions.append(float(np.mean(fractions)))
        std_fractions.append(float(np.std(fractions)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(residues))
    ax.bar(x, mean_fractions, yerr=std_fractions, capsize=4, color="#4c78a8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(residue_labels)
    ax.set_ylabel("Fraction of time in H-bond state")
    ax.set_xlabel("Residue")
    ax.set_title(f"{channel_label} ({ligand_label}) - H-bond occupancy fraction")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = scenario.output_dir / f"hydrogen_bond_{scenario.name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hydrophobic_bars(scenario: MindistScenario) -> None:
    """Plot hydrophobic contact fractions for all residues in a scenario."""
    scenario.output_dir.mkdir(parents=True, exist_ok=True)

    residues = scenario.residues
    channel_label = _format_channel_name(scenario.name)
    ligand_label = _format_ligand_label(scenario.name)
    residue_labels = _get_residue_labels(scenario.name, residues)
    mean_fractions: list[float] = []
    std_fractions: list[float] = []

    for residue in residues:
        # Compute per-trial fractions of frames under the hydrophobic cutoff.
        _, dist_matrix = _collect_trial_series(scenario, residue)
        dist_angstrom = dist_matrix * 10.0
        fractions = [
            _fraction_in_contact(dist_angstrom[trial_idx], scenario.hydrophobic_cutoff_angstrom)
            for trial_idx in range(dist_angstrom.shape[0])
        ]
        mean_fractions.append(float(np.mean(fractions)))
        std_fractions.append(float(np.std(fractions)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(residues))
    ax.bar(x, mean_fractions, yerr=std_fractions, capsize=4, color="#f28e2b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(residue_labels)
    ax.set_ylabel("Fraction of time in hydrophobic contact")
    ax.set_xlabel("Residue")
    ax.set_title(
        f"Hydrophobic occupancy fraction for DEKA filter (<{scenario.hydrophobic_cutoff_angstrom:.1f}A) for {channel_label} ({ligand_label})"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = scenario.output_dir / f"hydrophobic_{scenario.name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_scenario(scenario: MindistScenario) -> None:
    """Run all plots for a scenario and write outputs to disk."""
    plot_trial_time_series_overlay(scenario)
    plot_occupancy_bars(scenario)
    plot_hydrophobic_bars(scenario)