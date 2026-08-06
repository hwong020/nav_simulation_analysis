"""Generate mean ± SD RMSF plots from five XVG trials per channel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE = (5, 4.5)
TITLE_FONT_SIZE = 16
AUTO_FIG_WIDTH = False
AUTO_FIG_WIDTH_PER_RESIDUE = 0.7
AUTO_FIG_WIDTH_PADDING = 1.5
RESIDUE_X_SPACING = 0.01
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
BREAK_GAP = 6.0
BREAK_MARK_HEIGHT = 0.028
X_TICK_LABEL_FONT_SIZE = 12

MEAN_COLOR = "#0f766e"
SHADE_COLOR = "#5eead4"

DOMAIN_POINT_COLORS = {
    "DI": "#4e79a7",
    "DII": "#f28e2b",
    "DIII": "#59a14f",
    "DIV": "#e15759",
    "Other": "#9d9da1",
}

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

CHANNEL_YMAX = {
    "nav1-1": 8.0,
    "nav1-2": 2.5,
    "nav1-3": 8.0,
    "nav1-5": 12.5,
}

DOMAIN_STYLES = {
    "DI": {"start": 1, "end": 423, "color": "#fde68a"},
    "DII": {"start": 423, "end": 992, "color": "#bfdbfe"},
    "DIII": {"start": 993, "end": 1483, "color": "#c7d2fe"},
    "DIV": {"start": 1484, "end": 2009, "color": "#fecdd3"},
}

CHANNEL_DOMAIN_STYLES = {
    "nav1-1": {
        "DI": {"start": 250, "end": 420, "color": "#fde68a"},
        "DII": {"start": 897, "end": 992, "color": "#bfdbfe"},
        "DIII": {"start": 1347, "end": 1478, "color": "#c7d2fe"},
        "DIV": {"start": 1670, "end": 1783, "color": "#fecdd3"},
    },
    "nav1-2": {
        "DI": {"start": 251, "end": 422, "color": "#fde68a"},
        "DII": {"start": 888, "end": 983, "color": "#bfdbfe"},
        "DIII": {"start": 1337, "end": 1468, "color": "#c7d2fe"},
        "DIV": {"start": 1660, "end": 1773, "color": "#fecdd3"},
    },
    "nav1-3": {
        "DI": {"start": 250, "end": 421, "color": "#fde68a"},
        "DII": {"start": 889, "end": 984, "color": "#bfdbfe"},
        "DIII": {"start": 1335, "end": 1463, "color": "#c7d2fe"},
        "DIV": {"start": 1655, "end": 1768, "color": "#fecdd3"},
    },
    "nav1-5": {
        "DI": {"start": 253, "end": 410, "color": "#fde68a"},
        "DII": {"start": 846, "end": 939, "color": "#bfdbfe"},
        "DIII": {"start": 1334, "end": 1465, "color": "#c7d2fe"},
        "DIV": {"start": 1657, "end": 1769, "color": "#fecdd3"},
    },
}


def load_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load residue/RMSF columns from a GROMACS XVG file."""
    with path.open("r", encoding="utf-8") as handle:
        lines = [
            line for line in handle if line.strip() and not line.startswith("#") and not line.startswith("@")
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
    """Apply compact template-like styling."""
    ax.set_xlabel("Residue Number", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("RMSF of Residue (Å)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=1.0, length=3)
    ax.xaxis.set_label_coords(0.5, -0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.minorticks_on()


def build_compressed_axis(
    residues: np.ndarray,
    break_gap: float = BREAK_GAP,
) -> tuple[np.ndarray, list[dict[str, float]], list[float]]:
    """Use true residue numbers directly on x-axis (no compression)."""
    residues = residues.astype(float)
    return residues.copy(), [], []


def split_contiguous_segments(residues: np.ndarray) -> list[np.ndarray]:
    """Return index segments for contiguous residue blocks (step of 1)."""
    split_indices = np.where(np.diff(residues) > 1)[0] + 1
    return np.split(np.arange(len(residues)), split_indices)


def map_residue_to_compressed_x(residue: float, segments: list[dict[str, float]]) -> float | None:
    """Identity map: residue IDs are plotted directly on x-axis."""
    return float(residue)


def domain_for_residue(channel_key: str, residue: int) -> str:
    """Return DI/DII/DIII/DIV category for a residue, else Other."""
    domain_styles = CHANNEL_DOMAIN_STYLES.get(channel_key, DOMAIN_STYLES)
    for label, spec in domain_styles.items():
        if int(spec["start"]) <= residue <= int(spec["end"]):
            return label
    return "Other"


def set_domain_boundary_ticks(ax: plt.Axes, channel_key: str, segments: list[dict[str, float]]) -> None:
    """Set domain-boundary ticks with deterministic two-row, pair-split labels."""
    domain_styles = CHANNEL_DOMAIN_STYLES.get(channel_key, DOMAIN_STYLES)
    residues: list[int] = []
    for spec in domain_styles.values():
        residues.extend([int(spec["start"]), int(spec["end"])])

    unique_residues: list[int] = []
    for res in residues:
        if res not in unique_residues:
            unique_residues.append(res)

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for res in unique_residues:
        x = map_residue_to_compressed_x(float(res), segments)
        if x is None:
            continue
        tick_positions.append(x)
        tick_labels.append(str(res))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([""] * len(tick_positions))

    # Deterministic single-row layout with explicit horizontal splitting.
    placed = []
    renderer = ax.figure.canvas.get_renderer()
    min_pad_px = 12.0

    for idx, (x, label) in enumerate(zip(tick_positions, tick_labels)):
        is_start = idx % 2 == 0
        y_offset = -9

        # Keep boundary neighbors apart: ... end(idx odd) | start(idx+1 even) ...
        if is_start and idx > 0:
            x_offset = 0.5
            ha = "left"
        elif not is_start:
            x_offset = -0.5
            ha = "right"
        else:
            x_offset = 0.0
            ha = "center"

        ann = ax.annotate(
            label,
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=X_TICK_LABEL_FONT_SIZE,
            color="#111827",
            clip_on=False,
            zorder=6,
        )

        # Single-row overlap resolver in display space (pixels).
        ax.figure.canvas.draw()
        bbox = ann.get_window_extent(renderer=renderer)
        if placed:
            prev = placed[-1]
            overlap_px = (prev.x1 + min_pad_px) - bbox.x0
            if overlap_px > 0:
                # convert pixels -> points for annotate offset
                delta_pts = overlap_px * 72.0 / ax.figure.dpi
                x_offset += delta_pts
                ann.set_position((x_offset, y_offset))
                ax.figure.canvas.draw()
                bbox = ann.get_window_extent(renderer=renderer)

        placed.append(bbox)


def plot_channel(channel_dir: Path, output_dir: Path) -> None:
    """Create discrete top-8 RMSF plot (trial points + mean±SD) for one channel."""
    residues, trial_values = collect_trials(channel_dir / "rmsf_multiple_lig")
    channel_name = channel_dir.name.replace("nav1-", "Nav1.")
    top_res = TOP8_CONTACT_RESIDUES.get(channel_dir.name)
    if not top_res:
        raise ValueError(f"No top-8 residue mapping provided for {channel_dir.name}")

    residue_to_idx = {int(res): i for i, res in enumerate(residues)}
    chosen = [res for res in top_res if res in residue_to_idx]
    chosen = sorted(chosen)
    if not chosen:
        raise ValueError(f"None of top-8 residues found in RMSF axis for {channel_dir.name}")

    if AUTO_FIG_WIDTH:
        fig_width = max(FIGURE_SIZE[0], len(chosen) * AUTO_FIG_WIDTH_PER_RESIDUE + AUTO_FIG_WIDTH_PADDING)
    else:
        fig_width = FIGURE_SIZE[0]
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_SIZE[1]))
    x = np.arange(len(chosen), dtype=float) * RESIDUE_X_SPACING

    for i, resid in enumerate(chosen):
        idx = residue_to_idx[resid]
        values = trial_values[:, idx]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        domain = domain_for_residue(channel_dir.name, resid)
        color = DOMAIN_POINT_COLORS.get(domain, DOMAIN_POINT_COLORS["Other"])

        ax.scatter(np.full_like(values, x[i], dtype=float), values, s=30, color=color, alpha=0.45, edgecolors="none", zorder=2)
        ax.errorbar(x[i], mean_val, yerr=std_val, fmt="o", color=color, ecolor=color, elinewidth=1.8, capsize=4, markersize=5, zorder=3)

    ax.set_xlim(-0.6 * RESIDUE_X_SPACING, (len(chosen) - 0.4) * RESIDUE_X_SPACING)
    ax.set_ylim(0, CHANNEL_YMAX.get(channel_dir.name, 15.0))
    ax.set_xticks(x)
    label_map = _label_map_for_channel(channel_dir.name)
    ax.set_xticklabels(
        [label_map.get(r, str(r)) for r in chosen],
        fontsize=X_TICK_LABEL_FONT_SIZE,
        rotation=90,
        ha="center",
        va="top",
    )

    from matplotlib.patches import Patch
    present_domains = []
    for resid in chosen:
        d = domain_for_residue(channel_dir.name, resid)
        if d not in present_domains:
            present_domains.append(d)
    handles = [Patch(facecolor=DOMAIN_POINT_COLORS[d], edgecolor="none", label=d) for d in present_domains]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=12,
        frameon=False,
        borderaxespad=0.0,
    )

    style_axes(ax)
    ax.set_title(f"{channel_name} (Pre-bound)", fontsize=TITLE_FONT_SIZE, pad=10)
    fig.tight_layout(rect=(0.005, 0.1, 0.84, 0.965), pad=0.05)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"rmsf_{channel_dir.name}_multiple_lig_mean_sd.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def main() -> None:
    """Run RMSF plotting for all channels that contain RMSF trial data."""
    src_root = Path("src")
    results_root = Path("results")

    for channel_dir in sorted(src_root.glob("nav1-*")):
        rmsf_dir = channel_dir / "rmsf_multiple_lig"
        if not rmsf_dir.exists():
            print(f"Skipping {channel_dir.name}: no rmsf_multiple_lig directory")
            continue

        try:
            plot_channel(channel_dir, results_root / channel_dir.name / "graphs")
            print(f"Generated RMSF plot for {channel_dir.name}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {channel_dir.name}: {exc}")


if __name__ == "__main__":
    main()