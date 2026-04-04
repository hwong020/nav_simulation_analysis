from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_top_contacts(
    input_folder: Path,
    output_folder: Path,
    top_n: int = 30,
) -> None:
    """Export top-N contact occupancy residues for each trial CSV in input_folder."""
    csv_files = sorted(input_folder.glob("contact_probability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No contact_probability_*.csv found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        required_cols = {"ResID", "Resname", "Probability"}
        if not required_cols.issubset(df.columns):
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        top_df = df.sort_values("Probability", ascending=False).head(top_n)
        output_name = csv_path.stem.replace("contact_probability_", "top30_contact_probability_")
        output_path = output_folder / f"{output_name}.csv"
        top_df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def main() -> None:
    input_folder = Path("src/nav1-3/contact_occupancies")
    output_folder = Path("results/nav1-3/rankings")
    export_top_contacts(input_folder, output_folder, top_n=30)


if __name__ == "__main__":
    main()