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


def _expected_mindist_files(residues: list[str], trials: list[int]) -> list[str]:
    """Return the expected XVG filenames for a scenario."""
    return [f"mindist_{residue}_{trial}.xvg" for residue in residues for trial in trials]


def _find_missing_files(
    scenario_path: Path, residues: list[str], trials: list[int]
) -> list[str]:
    """Return any expected XVG files that are missing from a scenario directory."""
    return [
        filename
        for filename in _expected_mindist_files(residues, trials)
        if not (scenario_path / filename).exists()
    ]


def main() -> None:
    """Run mindist analysis for selected channels and scenarios.

    Edit the channels list to reuse this runner for other Nav channels.
    """
    # Update this list when you want to analyze additional channels.
    channels = [
        "nav1-1",
        "nav1-2",  
        "nav1-3", 
        "nav1-4",
        "nav1-5",
        "nav1-6",
        "nav1-7",
        "nav1-8",
        "nav1-9",
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

            missing_files = _find_missing_files(scenario_path, residues, trials)
            if missing_files:
                preview = ", ".join(missing_files[:4])
                if len(missing_files) > 4:
                    preview += f", ... (+{len(missing_files) - 4} more)"
                print(
                    f"Skipping {channel} {scenario_name}: incomplete input set in "
                    f"{scenario_path} (missing {len(missing_files)} file(s): {preview})"
                )
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
            try:
                run_scenario(scenario)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skipping {scenario.name}: {exc}")
                continue


if __name__ == "__main__":
    main()