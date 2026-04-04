from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Update these values directly when you want a different channel or folder.
CHANNEL = "nav1-1"
TOP_N = 15
INPUT_FOLDER = Path("results") / CHANNEL / "rankings"
OUTPUT_IMAGE = (
    Path("results") / CHANNEL / "rankings" / f"top{TOP_N}_contact_probability_mean_std.png"
)

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


def annotate_ring(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    channel_key = channel.lower()
    if channel_key not in RING_RESIDUES:
        raise ValueError(f"Unknown channel '{channel}'. Expected one of: {', '.join(RING_RESIDUES)}")

    outer = RING_RESIDUES[channel_key]["outer"]
    inner = RING_RESIDUES[channel_key]["inner"]

    def ring_label(resid: int) -> str:
        if resid in outer:
            return "Outer"
        if resid in inner:
            return "Inner"
        return "None"

    df = df.copy()
    df["Ring"] = df["ResID"].astype(int).map(ring_label)
    return df


def load_top_contacts(input_folder: Path) -> pd.DataFrame:
    csv_files = sorted(input_folder.glob("top30_contact_probability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No top30_contact_probability_*.csv found in {input_folder}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        required_cols = {"ResID", "Resname", "Probability"}
        if not required_cols.issubset(df.columns):
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        df = df.copy()
        df["Trial"] = csv_path.stem.replace("top30_contact_probability_", "")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize_top_contacts(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    grouped = (
        df.groupby(["ResID", "Resname"], as_index=False)["Probability"]
        .agg(MeanProbability="mean", StdProbability="std")
        .sort_values("MeanProbability", ascending=False)
    )
    return grouped.head(top_n).reset_index(drop=True)


def export_top_stats(
    channel: str,
    input_folder: Path,
    top_n: int,
) -> pd.DataFrame:
    df = load_top_contacts(input_folder)
    summary = summarize_top_contacts(df, top_n)
    summary = annotate_ring(summary, channel)
    return summary


def plot_top_contacts(summary: pd.DataFrame, output_path: Path, channel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_series = summary["Resname"].astype(str) + summary["ResID"].astype(str)
    colors = summary["Ring"].map({"Outer": "#d95f02", "Inner": "#1b9e77", "None": "#757575"})

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        label_series,
        summary["MeanProbability"],
        yerr=summary["StdProbability"].fillna(0),
        color=colors,
        capsize=4,
    )
    ax.set_xlabel("Residue")
    ax.set_ylabel("Contact probability (mean ± std)")
    ax.set_title(f"Top {len(summary)} contact probabilities ({channel})")
    ax.tick_params(axis="x", rotation=45)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d95f02", label="Outer ring"),
        plt.Rectangle((0, 0), 1, 1, color="#1b9e77", label="Inner ring"),
        plt.Rectangle((0, 0), 1, 1, color="#757575", label="Other"),
    ]
    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    channel = CHANNEL.lower()
    summary = export_top_stats(channel, INPUT_FOLDER, TOP_N)
    plot_top_contacts(summary, OUTPUT_IMAGE, channel)
    print(f"Saved {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()