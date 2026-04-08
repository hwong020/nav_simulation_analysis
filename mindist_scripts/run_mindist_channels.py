"""Run mindist plotting for a list of channels and scenarios."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from .mindist_analysis_core import MindistScenario, run_scenario
except ImportError:  # Allows running via "Run" in VS Code (direct script execution)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mindist_scripts.mindist_analysis_core import MindistScenario, run_scenario


def main() -> None:
    """Run mindist analysis for selected channels and scenarios.

    Edit the channels list to reuse this runner for other Nav channels.
    """
    # Update this list when you want to analyze additional channels.
    channels = [
        "nav1-1",
        "nav1-2",
        "nav1-3",
    ]

    # Residues are ordered to match DEKA labels in the analysis core.
    residues = ["d", "e", "k", "a"]
    trials = [1, 2, 3, 4, 5]

    for channel in channels:
        base_path = Path("src") / channel
        output_dir = Path("results") / channel / "graphs"

        # Scenario name -> input directory mapping.
        scenarios = {
            "single": base_path / "single_lig_mindist",
            "multiple": base_path / "multiple_lig_mindist",
        }

        for scenario_name, scenario_path in scenarios.items():
            if not scenario_path.exists():
                print(f"Skipping {channel} {scenario_name}: {scenario_path} not found")
                continue

            # Bundle scenario configuration for plotting.
            scenario = MindistScenario(
                name=f"{channel}-{scenario_name}",
                input_dir=scenario_path,
                output_dir=output_dir,
                residues=residues,
                trials=trials,
                hbond_cutoff_angstrom=3.5,
                hydrophobic_cutoff_angstrom=4.0,
            )
            print(f"Processing {scenario.name}...")
            run_scenario(scenario)


if __name__ == "__main__":
    main()