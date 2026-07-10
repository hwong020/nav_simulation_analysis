#!/usr/bin/env python3
"""
Generate 4 compact domain PNGs from CLUSTAL aln-clustal_num alignment.

Design:
- strict ClustalX-like residue palette
- motif-focused windows (motif ± flank)
- centered domain title
- no consensus row
- horizontal cylinder at top over conserved core motif zone
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


# Canonical ClustalX base colors (simplified non-context version for fixed mapping)
CLUSTALX_COLORS: Dict[str, str] = {
    "A": "#80A0F0",
    "I": "#80A0F0",
    "L": "#80A0F0",
    "M": "#80A0F0",
    "F": "#80A0F0",
    "W": "#80A0F0",
    "V": "#80A0F0",
    "K": "#F01505",
    "R": "#F01505",
    "E": "#C048C0",
    "D": "#C048C0",
    "N": "#15C015",
    "Q": "#15C015",
    "S": "#15C015",
    "T": "#15C015",
    "C": "#F08080",
    "G": "#F09048",
    "P": "#C0C000",
    "H": "#15A4A4",
    "Y": "#15A4A4",
    "-": "#FFFFFF",
    "X": "#EFEFEF",
}

DISPLAY_NAME_MAP = {
    "Nav1.4Fugu": "Nav1.4 (Takifugu rubripes)",
    "Nav1.1Snake": "Nav1.1 (Thamnophis sirtalis)",
}


def parse_clustal_num(path: Path) -> Tuple[List[str], List[str]]:
    seq_chunks: "OrderedDict[str, List[str]]" = OrderedDict()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("CLUSTAL"):
                continue
            if line[0].isspace():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq_chunks.setdefault(parts[0], []).append(parts[1])
    if not seq_chunks:
        raise ValueError("No sequence data parsed from file")
    names = list(seq_chunks.keys())
    seqs = ["".join(seq_chunks[n]) for n in names]
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent alignment lengths: {sorted(lengths)}")
    return names, seqs


def display_name(raw: str) -> str:
    if raw in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[raw]
    if raw.startswith("Nav"):
        return f"{raw} (Homo sapiens)"
    return raw


def ungapped_and_map(aligned: str) -> Tuple[str, List[int]]:
    ungapped, idx_map = [], []
    for i, c in enumerate(aligned):
        if c != "-":
            ungapped.append(c)
            idx_map.append(i)
    return "".join(ungapped), idx_map


def first_match_from(s: str, motifs: Sequence[str], start: int = 0) -> Tuple[int, int]:
    best_i, best_len = -1, 0
    for m in motifs:
        i = s.find(m, start)
        if i != -1 and (best_i == -1 or i < best_i):
            best_i, best_len = i, len(m)
    return best_i, best_len


def motif_list_for_domain(name: str, domain_key: str) -> List[str]:
    if domain_key == "I":
        if name in {"Nav1.4", "Nav1.4Fugu", "Nav1.6", "Nav1.7", "Nav1.3"}:
            # Nav1.4Fugu can appear as DNWE in this alignment.
            return ["DYWE", "DNWE"]
        if name in {"Nav1.1", "Nav1.1Snake", "Nav1.2"}:
            return ["DFWE"]
        if name == "Nav1.5":
            return ["DCWE"]
        if name in {"Nav1.8", "Nav1.9"}:
            return ["DSWE"]
        return ["DYWE", "DFWE", "DCWE", "DSWE"]
    if domain_key == "II":
        return ["EWIE"]
    if domain_key == "III":
        if name == "Nav1.7":
            return ["KGWTI"]
        return ["KGWMD"]
    if domain_key == "IV":
        return ["AGWD"]
    raise ValueError(f"Unknown domain key: {domain_key}")


def compact_window_from_motifs(
    names: List[str],
    seqs: List[str],
    domain_key: str,
    flank: int,
) -> Tuple[int, int, int, int]:
    """
    Returns:
    dom_start, dom_end, core_start, core_end in alignment coordinates.
    """
    core_starts, core_ends = [], []
    for name, aligned in zip(names, seqs):
        ung, map_u2a = ungapped_and_map(aligned)
        motifs = motif_list_for_domain(name, domain_key)
        mi, mlen = first_match_from(ung, motifs, 0)
        if mi == -1:
            raise ValueError(f"Cannot find motif for domain {domain_key} in {name}; expected one of {motifs}")
        a0 = map_u2a[mi]
        a1 = map_u2a[mi + mlen - 1]
        core_starts.append(a0)
        core_ends.append(a1)

    core_start = min(core_starts)
    core_end = max(core_ends)
    aln_len = len(seqs[0])
    dom_start = max(0, core_start - flank)
    dom_end = min(aln_len - 1, core_end + flank)
    return dom_start, dom_end, core_start, core_end


def draw_cylinder(ax, x0: float, x1: float, y: float, h: float = 0.58) -> None:
    w = max(1.0, x1 - x0)
    body = Rectangle((x0, y - h / 2), w, h, facecolor="#BFC5CF", edgecolor="#667085", linewidth=0.8)
    left = Ellipse((x0, y), width=h, height=h, facecolor="#D9DEE6", edgecolor="#667085", linewidth=0.8)
    right = Ellipse((x0 + w, y), width=h, height=h, facecolor="#AAB4C3", edgecolor="#667085", linewidth=0.8)
    ax.add_patch(body)
    ax.add_patch(left)
    ax.add_patch(right)


def draw_domain(
    names: List[str],
    seqs: List[str],
    domain_key: str,
    dom_start: int,
    dom_end: int,
    core_start: int,
    core_end: int,
    output: Path,
    dpi: int,
    show_letters: bool,
) -> None:
    disp_names = [display_name(n) for n in names]
    nseq = len(names)
    ncol = dom_end - dom_start + 1
    # Tight left margin to push Nav labels further left and remove blank area.
    left_margin = max(len(n) + 0.8 for n in disp_names)

    total_w = left_margin + ncol + 2
    total_h = nseq + 1.65
    fig, ax = plt.subplots(figsize=(total_w * 0.112, total_h * 0.198), dpi=dpi)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    mono = dict(fontfamily="DejaVu Sans Mono", fontsize=7)

    # Keep title in figure coordinates (outside axes) and center it over content.
    title_obj = fig.suptitle(f"Domain {domain_key}", fontsize=8.8, fontweight="bold", y=0.998, x=0.5)

    # Cylinder removed per user request.
    row0 = 0.78
    for i, (name, seq) in enumerate(zip(disp_names, seqs)):
        y = row0 + i
        ax.text(left_margin - 0.4, y + 0.55, name, ha="right", va="center", **mono)
        for j, aa in enumerate(seq[dom_start : dom_end + 1]):
            x = left_margin + j
            col = CLUSTALX_COLORS.get(aa.upper(), CLUSTALX_COLORS["X"])
            ax.add_patch(Rectangle((x, y), 1.0, 1.0, facecolor=col, edgecolor="none"))
            if show_letters:
                ax.text(x + 0.5, y + 0.61, aa, ha="center", va="center", **mono)

    # Pixel-accurate crop to remove all unused whitespace.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bboxes = []
    for artist in list(ax.patches) + list(ax.texts):
        try:
            bb = artist.get_window_extent(renderer=renderer)
            if bb is not None and np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all():
                bboxes.append(bb)
        except Exception:
            pass

    if bboxes:
        from matplotlib.transforms import Bbox

        full_bb = Bbox.union(bboxes)

        # Center title over rendered graphic content (not raw figure canvas center).
        content_center_px = 0.5 * (full_bb.x0 + full_bb.x1)
        fig_w_px = fig.bbox.width
        if fig_w_px > 0:
            title_obj.set_x(float(content_center_px / fig_w_px))

        # Re-draw so title position update is reflected before cropping/export.
        fig.canvas.draw()

        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([[full_bb.x0, full_bb.y0], [full_bb.x1, full_bb.y1]])

        # Asymmetric micro-margins: trim left edge aggressively.
        left_pad = 0.01
        right_pad = 0.06
        top_pad = 0.06
        bottom_pad = 0.06
        ax.set_xlim(x0 - left_pad, x1 + right_pad)
        ax.set_ylim(y1 + bottom_pad, y0 - top_pad)

    # Reserve a tiny top band for the figure-level title.
    fig.tight_layout(pad=0.0, rect=(0.0, 0.0, 1.0, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    # Final hard trim: remove any residual uniform border from raster export.
    img = plt.imread(output)
    if img.ndim == 3:
        bg = img[0, 0, :]
        if img.shape[2] == 4:
            diff = np.max(np.abs(img[..., :4] - bg[:4]), axis=2)
        else:
            diff = np.max(np.abs(img[..., :3] - bg[:3]), axis=2)
        mask = diff > 1e-5
    else:
        bg = img[0, 0]
        mask = np.abs(img - bg) > 1e-5

    ys, xs = np.where(mask)
    if ys.size and xs.size:
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cropped = img[y0:y1, x0:x1]
        plt.imsave(output, cropped)


def main() -> None:
    p = argparse.ArgumentParser(description="Render compact motif-focused Nav domain images")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--flank", type=int, default=15, help="Residue context on each side of motif")
    p.add_argument("--hide-letters", action="store_true")
    args = p.parse_args()

    names, seqs = parse_clustal_num(args.input)

    outputs = []
    for d in ["I", "II", "III", "IV"]:
        d0, d1, c0, c1 = compact_window_from_motifs(names, seqs, d, flank=args.flank)
        out = args.outdir / f"domain_{d}.png"
        draw_domain(
            names=names,
            seqs=seqs,
            domain_key=d,
            dom_start=d0,
            dom_end=d1,
            core_start=c0,
            core_end=c1,
            output=out,
            dpi=args.dpi,
            show_letters=not args.hide_letters,
        )
        outputs.append((d, out, d0, d1, c0, c1))

    for d, out, d0, d1, c0, c1 in outputs:
        print(f"Domain {d}: {out} | window={d0+1}-{d1+1}, core={c0+1}-{c1+1}")


if __name__ == "__main__":
    main()
