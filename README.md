# Nav Simulation Analysis

This repository contains Python scripts and analysis outputs used to generate figures for a paper based on Nav channel molecular simulation data. The project focuses on comparing ligand/channel behavior across Nav1.x systems (Nav1.1, Nav1.2, Nav1.3, and Nav1.5) using contact occupancy, hydrogen bonding, minimum-distance, RMSD, RMSF, and PyLipID-style ligand interaction analyses.

The repository is organized around input data in `src/`, analysis/plotting scripts in topic-specific folders, and generated paper-ready graphs and tables in `results/`.

## Project purpose

The main goal of this repository is to turn processed molecular dynamics simulation data into clear, publication-quality graphs for comparing Nav channel simulations. The scripts generate figures such as:

- Contact occupancy heatmaps across Nav channels
- Top-contact residue ranking bar plots with mean and standard deviation
- Hydrogen-bond number time-series plots
- Hydrogen-bond frame percentage bar charts
- Minimum-distance time-series plots for DEKA/filter residues
- RMSD plots for TTX inside the binding position
- RMSF bar charts comparing flooding and single-ligand simulations
- PyLipID ligand interaction plots and binding-site summaries

## Repository structure

```text
nav_simulation_analysis/
├── contact_occupancy_scripts/
│   ├── combined_contact_heatmap.py
│   └── plot_top_contact_rankings.py
├── hbond_stuff/
│   ├── plot_hbond_number_timeseries.py
│   └── plot_hbond_percentage_bars.py
├── mindist_scripts/
│   ├── mindist_analysis_core.py
│   └── run_mindist_channels.py
├── pylipid_scripts/
│   └── analyse_lig.py
├── rms_scripts/
│   ├── run_rmsd_channels.py
│   ├── run_rmsf_channels_combined.py
│   ├── run_rmsf_channels_multiple.py
│   └── run_rmsf_channels_single.py
├── src/
│   ├── nav1-1/
│   ├── nav1-2/
│   ├── nav1-3/
│   ├── nav1-4/
│   ├── nav1-5/
│   ├── nav1-6/
│   ├── nav1-7/
│   ├── nav1-8/
│   └── nav1-9/
└── results/
    ├── nav1-1/
    ├── nav1-2/
    ├── nav1-3/
    ├── nav1-4/
    ├── nav1-5/
    ├── nav1-6/
    ├── nav1-7/
    ├── nav1-8/
    └── nav1-9/
```

## Data layout

Each Nav channel has its own folder under `src/`, for example:

```text
src/nav1-1/
├── contact_occupancies/
├── hbonds/
├── rmsd/
├── rmsf_multiple_lig/
├── rmsf_single_lig/
└── ...
```

Common input file types include:

- `contact_probability_*.csv` — per-trial contact occupancy/probability tables
- `hb_num_ps_*.xvg` — GROMACS hydrogen-bond number time-series files
- `rmsd_lig_*.xvg` — GROMACS RMSD time-series files
- `rmsf_ca_*.xvg` — GROMACS RMSF files
- `mindist_<residue>_<trial>.xvg` — minimum-distance time-series files
- `pose_info.txt` and PyLipID output files — ligand/binding-site information

Generated outputs are written mainly under `results/<channel>/graphs/`, `results/<channel>/rankings/`, or directly into the relevant analysis script folder depending on the script.

## Requirements

The plotting scripts are written in Python and use scientific Python libraries. The core scripts use:

- Python 3.10+
- `numpy`
- `pandas`
- `matplotlib`

The PyLipID workflow additionally uses:

- `pylipid`
- `mdtraj` and related PyLipID dependencies

A typical setup is:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas matplotlib pylipid
```

If you only need to regenerate the standard plotting figures and are not running the PyLipID workflow, `numpy`, `pandas`, and `matplotlib` should be sufficient for most scripts.

## Usage

Run scripts from the repository root so that relative paths such as `src/` and `results/` resolve correctly.

```bash
cd c:\Users\hogan\nav_simulation_analysis
```

### Contact occupancy heatmap

Creates a combined trial-averaged contact occupancy heatmap across available Nav channels.

```bash
python contact_occupancy_scripts/combined_contact_heatmap.py
```

Output:

```text
results/contact_occupancy_all_channels.png
```

### Top contact rankings

Creates per-channel top-contact residue plots and exports ranking CSV files.

```bash
python contact_occupancy_scripts/plot_top_contact_rankings.py
```

Typical outputs:

```text
results/<channel>/rankings/average_contact_occupancy_ranking.csv
results/<channel>/rankings/top30_contact_occupancy_trial*.csv
results/<channel>/graphs/top_contact_occupancy_<channel>.png
```

### Hydrogen-bond number time series

Creates raw H-bond number time-series plots for selected channels.

```bash
python hbond_stuff/plot_hbond_number_timeseries.py
```

Typical output:

```text
results/<channel>/graphs/hbond_number_<channel>_timeseries.png
```

### Hydrogen-bond percentage plots

Calculates the percentage of frames with 1 through 5 H-bonds from raw XVG files and plots mean ± standard deviation.

```bash
python hbond_stuff/plot_hbond_percentage_bars.py
```

Typical output:

```text
hbond_stuff/hbond_percentage_avg_std_<channel>.png
```

### RMSD plots

Creates RMSD plots with raw traces and running averages for channels with RMSD data.

```bash
python rms_scripts/run_rmsd_channels.py
```

Typical output:

```text
results/<channel>/graphs/rmsd_<channel>_running_avg.png
```

### RMSF plots

Creates RMSF plots for selected top-contact residues. The combined script compares flooding and single-ligand conditions where both are available.

```bash
python rms_scripts/run_rmsf_channels_combined.py
```

Additional RMSF scripts are available for multiple-ligand and single-ligand workflows:

```bash
python rms_scripts/run_rmsf_channels_multiple.py
python rms_scripts/run_rmsf_channels_single.py
```

### Minimum-distance analysis

Minimum-distance plotting utilities are located in `mindist_scripts/`. The core plotting functions are defined in:

```text
mindist_scripts/mindist_analysis_core.py
```

Run the channel wrapper script from the repository root:

```bash
python mindist_scripts/run_mindist_channels.py
```

### PyLipID ligand interaction analysis

The PyLipID configuration script is:

```bash
python pylipid_scripts/analyse_lig.py
```

Before running it, check the script settings for the trajectory files, topology files, ligand residue name, cutoffs, stride, and output options. The current script is configured around ligand residue name `LIG` and trajectory folders such as `MD_01/`, `MD_02/`, etc.

## Notes for reproducibility

- Run scripts from the repository root unless otherwise noted.
- Many scripts automatically skip channels that do not contain the expected input folder or files.
- GROMACS `.xvg` time values are generally converted from ps to ns in the plotting scripts.
- RMSD and RMSF values are converted from nm to Å where appropriate.
- Several plots use fixed axis limits, colors, and residue mappings so figures remain consistent across Nav channel comparisons.
- Contact residue classifications include outer-ring, inner-ring, and other residues based on hard-coded Nav channel residue maps in the plotting scripts.

## Outputs used for the paper

The most paper-relevant generated files are stored under `results/`, including per-channel graph folders and ranking tables. These outputs can be regenerated from the scripts when the corresponding input data exists in `src/`.

## Citation / attribution

To be added (Paper is still undergoing peer review)