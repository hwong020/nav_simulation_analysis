from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MindistScenario:
    name: str
    input_dir: Path
    output_dir: Path
    residues: list[str]
    trials: list[int]
    hbond_cutoff_angstrom: float = 3.5
    hydrophobic_cutoff_angstrom: float = 4.0


def _load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments=["@", "#"])
    return data[:, 0], data[:, 1]


def _collect_trial_series(
    scenario: MindistScenario, residue: str
) -> tuple[np.ndarray, np.ndarray]:
    times: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for trial in scenario.trials:
        filename = scenario.input_dir / f"mindist_{residue}_{trial}.xvg"
        if not filename.exists():
            raise FileNotFoundError(f"Missing {filename}")
        time, dist = _load_xvg(filename)
        times.append(time)
        dists.append(dist)
    time0 = times[0]
    for t in times[1:]:
        if not np.array_equal(t, time0):
            raise ValueError(
                f"Time axis mismatch for residue {residue} in {scenario.input_dir}"
            )
    return time0, np.vstack(dists)


def plot_residue_time_series(scenario: MindistScenario) -> None:
    scenario.output_dir.mkdir(parents=True, exist_ok=True)
    for residue in scenario.residues:
        time, dist_matrix = _collect_trial_series(scenario, residue)
        dist_angstrom = dist_matrix * 10.0
        mean_dist = np.mean(dist_angstrom, axis=0)
        std_dist = np.std(dist_angstrom, axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, mean_dist, color="#1f77b4", linewidth=2.2, label="Mean")
        ax.fill_between(
            time,
            mean_dist - std_dist,
            mean_dist + std_dist,
            color="#9ecae1",
            alpha=0.5,
            label="Std. dev",
        )
        ax.axhline(
            scenario.hbond_cutoff_angstrom,
            color="#d62728",
            linestyle="--",
            linewidth=2.4,
            alpha=0.9,
            zorder=5,
            label=f"H-bond cutoff ({scenario.hbond_cutoff_angstrom:.1f} Å)",
        )
        ax.axhline(
            scenario.hydrophobic_cutoff_angstrom,
            color="#ff7f0e",
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            zorder=4,
            label=f"Hydrophobic cutoff ({scenario.hydrophobic_cutoff_angstrom:.1f} Å)",
        )
        ax.set_title(f"{scenario.name} - Residue {residue.upper()} Mean Min Dist")
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance (Å)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        output_path = scenario.output_dir / f"mindist_{scenario.name}_{residue}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def _fraction_in_contact(dist_angstrom: np.ndarray, cutoff: float) -> float:
    if dist_angstrom.size == 0:
        return float("nan")
    return float(np.sum(dist_angstrom <= cutoff) / dist_angstrom.size)


def plot_occupancy_bars(scenario: MindistScenario) -> None:
    scenario.output_dir.mkdir(parents=True, exist_ok=True)

    residues = scenario.residues
    mean_fractions: list[float] = []
    std_fractions: list[float] = []

    for residue in residues:
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
    ax.set_xticklabels([res.upper() for res in residues])
    ax.set_ylabel("Fraction of time in H-bond state")
    ax.set_xlabel("Residue")
    ax.set_title(f"{scenario.name} - H-bond occupancy fraction")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = scenario.output_dir / f"hydrogen_bond_{scenario.name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hydrophobic_bars(scenario: MindistScenario) -> None:
    scenario.output_dir.mkdir(parents=True, exist_ok=True)

    residues = scenario.residues
    mean_fractions: list[float] = []
    std_fractions: list[float] = []

    for residue in residues:
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
    ax.set_xticklabels([res.upper() for res in residues])
    ax.set_ylabel("Fraction of time in hydrophobic contact")
    ax.set_xlabel("Residue")
    ax.set_title(
        f"{scenario.name} - Hydrophobic occupancy fraction (≤ {scenario.hydrophobic_cutoff_angstrom:.1f} Å)"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = scenario.output_dir / f"hydrophobic_{scenario.name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_scenario(scenario: MindistScenario) -> None:
    plot_residue_time_series(scenario)
    plot_occupancy_bars(scenario)
    plot_hydrophobic_bars(scenario)