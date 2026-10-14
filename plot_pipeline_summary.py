#!/usr/bin/python
"""
plot_pipeline_summary.py
------------------------
Standalone script: combines per-locus TSV summaries produced by either
the Python amplicon_pipeline.py or the PBS amplicon_pipeline.pbs, then
regenerates all summary plots and insert-size histograms.

Run this on the HPC after all pipeline jobs have finished.

Usage:
    python plot_pipeline_summary.py -b <base_path> [-p primers.txt]

    -b / --base-path   Root output directory used for the pipeline run.
    -p / --primers     primers.txt path  [default: <base_path>/primers.txt
                       or ./primers.txt]

TSV discovery (tries in order for each stage):
  1. Combined file written by the Python pipeline:
         <base_path>/cutadapt_summary.tsv
         <base_path>/bbduk_summary.tsv
         <base_path>/bbmerge_summary.tsv
  2. Per-locus files written by the PBS pipeline:
         <base_path>/cutadapt_<LOCUS>_summary.tsv
         <base_path>/bbduk_<LOCUS>_summary.tsv
         <base_path>/bbmerge_<LOCUS>_summary.tsv
     These are merged on the Sample column and written to combined TSVs
     so subsequent --plots-only runs can use option 1.

Outputs (all written to <base_path>):
    cutadapt_summary.tsv          (combined, if built from per-locus files)
    bbduk_summary.tsv
    bbmerge_summary.tsv
    pipeline_summary.png
    cutadapt_summary.png
    bbduk_bbmerge_summary.png
    <locus>_merge_hist/combined_histogram.tsv
    <locus>_merge_hist/combined_histogram.png
"""

import argparse
import glob
import json
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------
_CMAP_NAME = "viridis"
_CMAP_LO   = 0.05
_CMAP_HI   = 0.90


def _locus_colours(n: int) -> list[tuple]:
    cmap = plt.get_cmap(_CMAP_NAME)
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(p) for p in np.linspace(_CMAP_LO, _CMAP_HI, n)]


# ---------------------------------------------------------------------------
# Primer loading
# ---------------------------------------------------------------------------

def load_primers(path: str) -> list[dict]:
    primers = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                sys.exit(
                    f"ERROR: {path} line {lineno} has {len(parts)} fields "
                    f"(expected 5: locus fwd_name fwd_seq rev_name rev_seq):\n  {raw}"
                )
            locus, fwd_name, fwd_seq, rev_name, rev_seq = parts
            primers.append(dict(locus=locus, fwd_name=fwd_name, fwd_seq=fwd_seq,
                                rev_name=rev_name, rev_seq=rev_seq))
    if not primers:
        sys.exit(f"ERROR: No primers found in {path}")
    return primers


# ---------------------------------------------------------------------------
# TSV loading + merging
# ---------------------------------------------------------------------------

def _read_combined_tsv(path: str, col_suffixes: list[str] | None = None) -> tuple[list[str], dict]:
    """Read a combined TSV (Sample + triplet-of-cols-per-locus).

    Returns (samples_ordered, stats) where
    stats[sample][locus] = (int, int, str) or None.

    If `col_suffixes` is given, the locus name is recovered by stripping the
    known first suffix (e.g. "COI_Pairs" → "COI"). This is required for stages
    whose suffixes are single tokens (bbmerge: Pairs/Joined/Pct_Joined);
    the previous "drop last two _-tokens" heuristic only worked for two-token
    suffixes (cutadapt/bbduk) and yielded an empty locus name otherwise.
    """
    samples = []
    stats: dict[str, dict] = {}
    if not os.path.exists(path):
        return samples, stats

    suf0 = col_suffixes[0] if col_suffixes else None

    def _locus_from_header(h: str) -> str:
        if suf0 and h.endswith("_" + suf0):
            return h[: -(len(suf0) + 1)]
        # Fallback: original heuristic (drop last two _-tokens)
        return "_".join(h.split("_")[:-2])

    with open(path) as fh:
        headers = fh.readline().rstrip("\n").split("\t")
        # Build locus → column-index mapping from header triplets
        locus_cols: list[tuple[str, int]] = []
        for i in range(1, len(headers), 3):
            locus = _locus_from_header(headers[i])
            locus_cols.append((locus, i))

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            sample = parts[0]
            samples.append(sample)
            stats[sample] = {}
            for locus, col in locus_cols:
                try:
                    v0 = int(parts[col])
                    v1 = int(parts[col + 1])
                    v2 = parts[col + 2]
                    stats[sample][locus] = (v0, v1, v2)
                except (ValueError, IndexError):
                    stats[sample][locus] = None

    return samples, stats


def _read_per_locus_tsvs(base: str, stage: str, loci: list[str]) -> tuple[list[str], dict]:
    """Read per-locus TSVs written by the PBS pipeline.

    Each file has columns: Sample, Input_X, Result_X, Pct_X
    (column names vary by stage but position is always 0,1,2,3).

    Returns (samples_ordered, stats).
    """
    # Collect all per-locus dataframes
    locus_data: dict[str, pd.DataFrame] = {}
    all_samples_ordered: list[str] = []
    seen: set[str] = set()

    for locus in loci:
        path = os.path.join(base, f"{stage}_{locus}_summary.tsv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — locus {locus} will be missing from {stage} stats")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str)
        locus_data[locus] = df
        for s in df.iloc[:, 0]:
            if s not in seen:
                all_samples_ordered.append(s)
                seen.add(s)

    if not locus_data:
        return [], {}

    # Build stats dict
    stats: dict[str, dict] = {s: {} for s in all_samples_ordered}
    for locus, df in locus_data.items():
        for _, row in df.iterrows():
            sample = row.iloc[0]
            try:
                v0 = int(row.iloc[1])
                v1 = int(row.iloc[2])
                v2 = str(row.iloc[3])
                stats[sample][locus] = (v0, v1, v2)
            except (ValueError, IndexError):
                stats[sample][locus] = None

    return all_samples_ordered, stats


def _write_combined_tsv(samples: list[str], loci: list[str], stats: dict,
                        col_suffixes: list[str], path: str) -> None:
    """Write stats back out as a combined TSV."""
    header = ["Sample"] + [
        f"{locus}_{suf}" for locus in loci for suf in col_suffixes
    ]
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for sample in samples:
            row = [sample]
            for locus in loci:
                entry = stats.get(sample, {}).get(locus)
                if entry is None:
                    row += ["NA"] * 3
                else:
                    row += [str(entry[0]), str(entry[1]), entry[2]]
            fh.write("\t".join(row) + "\n")


def load_stage_stats(base: str, stage: str, loci: list[str],
                     col_suffixes: list[str]) -> tuple[list[str], dict]:
    """Try combined TSV first; fall back to per-locus PBS TSVs.

    If per-locus files are used, writes a combined TSV for future use.
    Returns (samples, stats).
    """
    combined_path = os.path.join(base, f"{stage}_summary.tsv")

    if os.path.exists(combined_path):
        print(f"  Reading combined TSV: {combined_path}")
        samples, stats = _read_combined_tsv(combined_path, col_suffixes)
        if samples:
            return samples, stats
        print(f"  WARNING: Combined TSV was empty, trying per-locus files...")

    print(f"  Combined TSV not found, looking for per-locus files...")
    samples, stats = _read_per_locus_tsvs(base, stage, loci)

    if samples:
        # Write combined for next time
        _write_combined_tsv(samples, loci, stats, col_suffixes, combined_path)
        print(f"  Combined TSV written: {combined_path}")

    return samples, stats


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

LOST_COLOUR = "#dddddd"


def _plot_setup(n: int) -> tuple[float, int, int]:
    """Return (fig_h, out_dpi, lbl_fs)."""
    fig_h   = min(max(8, n * 0.22), 400)
    out_dpi = 150 if fig_h < 200 else 100
    lbl_fs  = max(4, min(7, 600 // n))
    return fig_h, out_dpi, lbl_fs


def _style_hbar_ax(ax: plt.Axes, samples: list[str], lbl_fs: int,
                   show_labels: bool = True) -> None:
    n = len(samples)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(samples if show_labels else [""] * n, fontsize=lbl_fs)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()


def _get_pct(entry) -> float:
    """Safely extract percentage from a stats entry tuple."""
    if entry is None:
        return 0.0
    try:
        return float(entry[2])
    except (TypeError, ValueError, IndexError):
        return 0.0


def _get_count(entry) -> int:
    """Safely extract output count (index 1) from a stats entry tuple."""
    if entry is None or not isinstance(entry[1], int):
        return 0
    return entry[1]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_pipeline_summary(samples, primers, ca_stats, bbd_stats, bbm_stats,
                          out_path):
    """Combined overview: all stages in one figure."""
    n  = len(samples)
    nl = len(primers)
    loci    = [p["locus"] for p in primers]
    colours = _locus_colours(nl)
    y       = np.arange(n)

    fig_h, out_dpi, lbl_fs = _plot_setup(n)
    n_cols       = 1 + nl + nl + nl
    col_w_counts = 4.0
    col_w_pct    = 2.5
    fig_w        = col_w_counts + (n_cols - 1) * col_w_pct
    width_ratios = [col_w_counts] + [col_w_pct] * (n_cols - 1)

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": width_ratios})
    if n_cols == 1:
        axes = [axes]

    col = 0

    # ── cutadapt absolute counts ───────────────────────────────────────────
    ax0 = axes[col]; col += 1
    group_h = 0.75
    bh = group_h / nl
    offsets = np.linspace(-(group_h / 2) + bh / 2, (group_h / 2) - bh / 2, nl)
    for locus, colour, offset in zip(loci, colours, offsets):
        counts = [_get_count(ca_stats.get(s, {}).get(locus)) for s in samples]
        ax0.barh(y + offset, counts, height=bh, color=colour, label=locus, zorder=3)
    ax0.set_title("cutadapt\npairs written", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Pairs written", fontsize=8)
    ax0.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax0.legend(title="Locus", fontsize=6, title_fontsize=7,
               loc="lower right", framealpha=0.85)
    _style_hbar_ax(ax0, samples, lbl_fs, show_labels=True)

    # ── cutadapt % ─────────────────────────────────────────────────────────
    for li, (locus, colour) in enumerate(zip(loci, colours)):
        ax = axes[col]; col += 1
        ret  = [_get_pct(ca_stats.get(s, {}).get(locus)) for s in samples]
        lost = [max(0.0, 100.0 - v) for v in ret]
        ax.barh(y, ret,  height=0.8, color=colour,      label="written",   zorder=3)
        ax.barh(y, lost, height=0.8, color=LOST_COLOUR, left=ret,
                label="discarded", zorder=3)
        ax.set_title(f"cutadapt %\n{locus}", fontsize=9, fontweight="bold")
        ax.set_xlabel("% pairs", fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        _style_hbar_ax(ax, samples, lbl_fs, show_labels=False)

    # ── bbduk % ────────────────────────────────────────────────────────────
    for li, (locus, colour) in enumerate(zip(loci, colours)):
        ax = axes[col]; col += 1
        ret  = [_get_pct(bbd_stats.get(s, {}).get(locus)) for s in samples]
        lost = [max(0.0, 100.0 - v) for v in ret]
        ax.barh(y, ret,  height=0.8, color=colour,      label="retained", zorder=3)
        ax.barh(y, lost, height=0.8, color=LOST_COLOUR, left=ret,
                label="lost", zorder=3)
        ax.set_title(f"bbduk %\n{locus}", fontsize=9, fontweight="bold")
        ax.set_xlabel("% reads", fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        _style_hbar_ax(ax, samples, lbl_fs, show_labels=False)

    # ── bbmerge % ──────────────────────────────────────────────────────────
    for li, (locus, colour) in enumerate(zip(loci, colours)):
        ax = axes[col]; col += 1
        mrg  = [_get_pct(bbm_stats.get(s, {}).get(locus)) for s in samples]
        lost = [max(0.0, 100.0 - v) for v in mrg]
        ax.barh(y, mrg,  height=0.8, color=colour,      label="merged",   zorder=3)
        ax.barh(y, lost, height=0.8, color=LOST_COLOUR, left=mrg,
                label="unmerged", zorder=3)
        ax.set_title(f"bbmerge %\n{locus}", fontsize=9, fontweight="bold")
        ax.set_xlabel("% pairs", fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        _style_hbar_ax(ax, samples, lbl_fs, show_labels=False)

    fig.suptitle(f"Amplicon pipeline summary  ({n} samples)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=out_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({fig_w:.0f}×{fig_h:.0f} in, {out_dpi} dpi)")


def plot_cutadapt_summary(samples, primers, ca_stats, out_path):
    """Cutadapt-only plot: absolute counts + % written/discarded."""
    n  = len(samples)
    nl = len(primers)
    loci    = [p["locus"] for p in primers]
    colours = _locus_colours(nl)
    y       = np.arange(n)

    fig_h, out_dpi, lbl_fs = _plot_setup(n)
    n_cols       = 1 + nl
    col_w_counts = 4.0
    col_w_pct    = 2.5
    fig_w        = col_w_counts + nl * col_w_pct
    width_ratios = [col_w_counts] + [col_w_pct] * nl

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": width_ratios})
    if n_cols == 1:
        axes = [axes]

    # Absolute counts
    ax0 = axes[0]
    group_h = 0.75
    bh = group_h / nl
    offsets = np.linspace(-(group_h / 2) + bh / 2, (group_h / 2) - bh / 2, nl)
    for locus, colour, offset in zip(loci, colours, offsets):
        counts = [_get_count(ca_stats.get(s, {}).get(locus)) for s in samples]
        ax0.barh(y + offset, counts, height=bh, color=colour, label=locus, zorder=3)
    ax0.set_title("cutadapt\npairs written", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Pairs written", fontsize=8)
    ax0.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax0.legend(title="Locus", fontsize=6, title_fontsize=7,
               loc="lower right", framealpha=0.85)
    _style_hbar_ax(ax0, samples, lbl_fs, show_labels=True)

    # % per locus
    for li, (locus, colour) in enumerate(zip(loci, colours)):
        ax = axes[1 + li]
        ret  = [_get_pct(ca_stats.get(s, {}).get(locus)) for s in samples]
        lost = [max(0.0, 100.0 - v) for v in ret]
        ax.barh(y, ret,  height=0.8, color=colour,      label="written",   zorder=3)
        ax.barh(y, lost, height=0.8, color=LOST_COLOUR, left=ret,
                label="discarded", zorder=3)
        ax.set_title(f"cutadapt %\n{locus}", fontsize=9, fontweight="bold")
        ax.set_xlabel("% pairs", fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        _style_hbar_ax(ax, samples, lbl_fs, show_labels=False)

    fig.suptitle(f"cutadapt summary  ({n} samples)", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=out_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({fig_w:.0f}×{fig_h:.0f} in, {out_dpi} dpi)")


def plot_bbduk_bbmerge_summary(samples, primers, bbd_stats, bbm_stats, out_path):
    """bbduk + bbmerge stacked 100% retention plots."""
    n  = len(samples)
    nl = len(primers)
    loci    = [p["locus"] for p in primers]
    colours = _locus_colours(nl)
    y       = np.arange(n)

    fig_h, out_dpi, lbl_fs = _plot_setup(n)
    n_cols = nl + nl
    col_w  = 2.5
    fig_w  = max(8, n_cols * col_w)

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    col = 0

    for li, (locus, colour) in enumerate(zip(loci, colours)):
        ax = axes[col]; col += 1
        ret  = [_get_pct(bbd_stats.get(s, {}).get(locus)) for s in samples]
        lost = [max(0.0, 100.0 - v) for v in ret]
        ax.barh(y, ret,  height=0.8, color=colour,      label="retained", zorder=3)
        ax.barh(y, lost, height=0.8, color=LOST_COLOUR, left=ret,
                label="lost", zorder=3)
        ax.set_title(f"bbduk %\n{locus}", fontsize=9, fontweight="bold")
        ax.set_xlabel("% reads", fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        _style_hbar_ax(ax, samples, lbl_fs, show_labels=(li == 0))

    for li, (locus, colour) in enumerate(zip(loci, colours)):
        ax = axes[col]; col += 1
        mrg  = [_get_pct(bbm_stats.get(s, {}).get(locus)) for s in samples]
        lost = [max(0.0, 100.0 - v) for v in mrg]
        ax.barh(y, mrg,  height=0.8, color=colour,      label="merged",   zorder=3)
        ax.barh(y, lost, height=0.8, color=LOST_COLOUR, left=mrg,
                label="unmerged", zorder=3)
        ax.set_title(f"bbmerge %\n{locus}", fontsize=9, fontweight="bold")
        ax.set_xlabel("% pairs", fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        _style_hbar_ax(ax, samples, lbl_fs, show_labels=False)

    fig.suptitle(f"bbduk + bbmerge summary  ({n} samples)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=out_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({fig_w:.0f}×{fig_h:.0f} in, {out_dpi} dpi)")


def plot_insert_histograms(primers, base):
    """Combined insert-size histogram per locus from *.hist files."""
    for primer in primers:
        locus    = primer["locus"]
        hist_dir = os.path.join(base, f"{locus}_merge_hist")
        hist_files = glob.glob(os.path.join(hist_dir, "*.hist"))

        if not hist_files:
            print(f"  WARNING: No .hist files in {hist_dir} — skipping {locus}")
            continue

        print(f"  {locus}: combining {len(hist_files)} hist file(s)")

        combined: dict[int, int] = {}
        for hf in hist_files:
            with open(hf) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        try:
                            combined[int(parts[0])] = \
                                combined.get(int(parts[0]), 0) + int(parts[1])
                        except ValueError:
                            continue

        if not combined:
            print(f"  WARNING: All hist files for {locus} were empty — skipping")
            continue

        df = pd.DataFrame({
            "InsertSize": sorted(combined.keys()),
            "Count":      [combined[s] for s in sorted(combined.keys())],
        })
        df.to_csv(os.path.join(base, f"{locus}_histogram.tsv"), sep="\t", index=False)

        total          = df["Count"].sum()
        weighted_mean  = (df["InsertSize"] * df["Count"]).sum() / total
        weighted_med   = df.loc[df["Count"].cumsum() >= total / 2, "InsertSize"].iloc[0]
        mode_size      = df.loc[df["Count"].idxmax(), "InsertSize"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df["InsertSize"], df["Count"], width=1,
               color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(weighted_mean, color="red",    linestyle="--", linewidth=1.2,
                   label=f"Mean: {weighted_mean:.1f}")
        ax.axvline(weighted_med,  color="orange", linestyle="--", linewidth=1.2,
                   label=f"Median: {weighted_med}")
        ax.axvline(mode_size,     color="green",  linestyle="--", linewidth=1.2,
                   label=f"Mode: {mode_size}")
        ax.set_xlabel("Insert Size (bp)", fontsize=13)
        ax.set_ylabel("Count (all samples summed)", fontsize=13)
        ax.set_title(
            f"{locus} — Combined Insert Size Histogram\n"
            f"({len(hist_files)} samples, {total:,} total read pairs)",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join(base, f"{locus}_histogram.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plot_path}")


# ---------------------------------------------------------------------------
# Metadata + interactive HTML output
# ---------------------------------------------------------------------------

def load_metadata(path: str, samples: list[str]) -> dict | None:
    """Load a sample metadata CSV/TSV.

    Requires at least a sample-id column (named 'sample_id' if present,
    otherwise the first column is used) plus any number of extra columns.
    Handles a UTF-8 BOM and CRLF line endings. Only rows whose id is in
    `samples` are kept, so the filter options stay meaningful.

    Returns dict(id_col=str, columns=[...], rows={sample: {col: val}}) or
    None if the file can't be used.
    """
    if not path or not os.path.exists(path):
        return None

    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
    except Exception as exc:
        print(f"  WARNING: could not read metadata {path}: {exc}")
        return None

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or len(df.columns) < 1:
        print(f"  WARNING: metadata {path} has no usable columns")
        return None

    # Identify the sample-id column
    id_col = None
    for c in df.columns:
        if c.lower().replace(" ", "_") in ("sample_id", "sampleid", "sample", "id"):
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    other_cols = [c for c in df.columns if c != id_col]

    sample_set = set(samples)
    rows: dict[str, dict] = {}
    matched = 0
    for _, r in df.iterrows():
        sid = r[id_col]
        if not isinstance(sid, str):
            sid = "" if pd.isna(sid) else str(sid)
        sid = sid.strip()
        if sid not in sample_set:
            continue
        matched += 1
        rows[sid] = {
            c: ("" if pd.isna(r[c]) else str(r[c]).strip()) for c in other_cols
        }

    print(f"  Metadata: {path}")
    print(f"    id column : {id_col}")
    print(f"    extra cols: {', '.join(other_cols) if other_cols else '(none)'}")
    print(f"    matched   : {matched}/{len(samples)} samples")

    if matched == 0:
        print("    WARNING: no metadata ids matched the pipeline samples — "
              "check that the id column matches the FASTQ sample names.")

    return {"id_col": id_col, "columns": other_cols, "rows": rows}


def _load_histograms(base: str, loci: list[str]) -> dict:
    """Read per-locus insert-size histogram TSVs (written by
    plot_insert_histograms) so the HTML can draw them. Missing files are
    silently skipped."""
    hist: dict[str, dict] = {}
    for L in loci:
        path = os.path.join(base, f"{L}_histogram.tsv")
        if not os.path.exists(path):
            continue
        sizes, counts = [], []
        try:
            with open(path) as fh:
                fh.readline()  # header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        try:
                            sizes.append(int(float(parts[0])))
                            counts.append(int(float(parts[1])))
                        except ValueError:
                            continue
        except Exception as exc:
            print(f"  WARNING: could not read histogram {path}: {exc}")
            continue
        if sizes:
            hist[L] = {"sizes": sizes, "counts": counts}
    if hist:
        print(f"    histograms : {', '.join(hist.keys())}")
    return hist


def _build_html_data(samples, primers, ca_stats, bbd_stats, bbm_stats,
                     meta, base) -> dict:
    """Assemble the compact JSON payload embedded in the HTML."""
    loci = [p["locus"] for p in primers]

    def stage_block(stats):
        block = {}
        for s in samples:
            row = {}
            for locus in loci:
                e = stats.get(s, {}).get(locus)
                if e is None:
                    row[locus] = None
                else:
                    try:
                        pct = float(e[2])
                    except (TypeError, ValueError, IndexError):
                        pct = None
                    inp = e[0] if isinstance(e[0], int) else None
                    out = e[1] if isinstance(e[1], int) else None
                    row[locus] = [inp, out, pct]
            block[s] = row
        return block

    return {
        "loci": loci,
        "samples": samples,
        "stages": {
            "cutadapt": stage_block(ca_stats),
            "bbduk":    stage_block(bbd_stats),
            "bbmerge":  stage_block(bbm_stats),
        },
        "meta": meta or {"id_col": "sample_id", "columns": [], "rows": {}},
        "hist": _load_histograms(base, loci),
    }


def write_html_summary(samples, primers, ca_stats, bbd_stats, bbm_stats,
                       meta, base, out_path) -> None:
    """Write a self-contained interactive HTML viewer of the pipeline summary.

    The file embeds all data and renders with vanilla SVG/JS — no external
    libraries — so it works offline once downloaded from the HPC.
    """
    data = _build_html_data(samples, primers, ca_stats, bbd_stats, bbm_stats,
                            meta, base)
    payload = json.dumps(data, separators=(",", ":"))
    # Guard against an accidental </script> inside string data
    payload = payload.replace("</", "<\\/")

    n_samples = len(samples)
    title = f"Amplicon pipeline summary — {n_samples} samples"

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", payload).replace(
        "__TITLE__", title)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  Saved: {out_path}  (interactive, {n_samples} samples)")


# The template uses __DATA_JSON__ / __TITLE__ placeholders (not str.format,
# because the JS/CSS contain literal { } braces).
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#f4f6f8; --panel:#ffffff; --head:#eef1f5; --line:#dce1e8;
    --ink:#1f2933; --muted:#6b7280; --accent:#4f46e5; --accent-soft:#eef0fe;
    --lost:#dde2e8; --hl:rgba(79,70,229,.10);
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
    --sans:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);
       font-size:13px;display:flex;flex-direction:column}
  h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.2px}
  .sub{color:var(--muted);font-size:12px;margin-top:2px}

  /* ---- control bar ---- */
  #bar{position:sticky;top:0;z-index:40;background:var(--panel);
       border-bottom:1px solid var(--line);box-shadow:0 1px 4px rgba(0,0,0,.04)}
  .barrow{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-end;
          padding:12px 18px}
  .ctl{display:flex;flex-direction:column;gap:5px}
  .ctl > label{font-size:11px;color:var(--muted);font-weight:600;
               text-transform:uppercase;letter-spacing:.4px}
  select,input[type=text]{font-family:var(--sans);font-size:13px;padding:5px 8px;
       border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink)}
  input[type=range]{accent-color:var(--accent);width:130px}
  .rangeval{font-family:var(--mono);font-size:11px;color:var(--muted);min-width:30px}
  .rangewrap{display:flex;align-items:center;gap:8px}
  button{font-family:var(--sans);font-size:12px;padding:6px 11px;border-radius:7px;
         border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer}
  button:hover{background:var(--head)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
  button.primary:hover{filter:brightness(1.06)}

  .legend{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;
          padding:0 18px 11px}
  .lg{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px}
  .sw{width:13px;height:13px;border-radius:3px;flex:none}
  .count-pill{font-family:var(--mono);font-size:11px;color:var(--muted);
       background:var(--head);border:1px solid var(--line);border-radius:999px;
       padding:3px 10px}

  /* ---- filter popovers ---- */
  details.filter{position:relative}
  details.filter > summary{list-style:none;cursor:pointer;font-size:13px;
       padding:5px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;
       display:flex;align-items:center;gap:7px;min-width:120px;white-space:nowrap}
  details.filter > summary::-webkit-details-marker{display:none}
  details.filter > summary .chev{margin-left:auto;color:var(--muted);font-size:10px}
  details.filter[open] > summary{border-color:var(--accent)}
  .fbadge{font-family:var(--mono);font-size:10px;background:var(--accent-soft);
          color:var(--accent);border-radius:999px;padding:1px 6px}
  .pop{position:absolute;top:calc(100% + 6px);left:0;z-index:50;background:#fff;
       border:1px solid var(--line);border-radius:9px;box-shadow:0 8px 24px rgba(0,0,0,.12);
       width:230px;max-height:300px;display:flex;flex-direction:column;overflow:hidden}
  .pop .pophead{display:flex;gap:8px;padding:8px;border-bottom:1px solid var(--line)}
  .pop .pophead button{flex:1;padding:4px;font-size:11px}
  .pop .opts{overflow:auto;padding:4px 0}
  .opt{display:flex;align-items:center;gap:8px;padding:5px 11px;cursor:pointer;
       font-size:12px}
  .opt:hover{background:var(--head)}
  .opt input{accent-color:var(--accent)}
  .opt .val{font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .opt .n{margin-left:auto;color:var(--muted);font-size:11px;font-family:var(--mono)}

  /* ---- plot area ---- */
  #plot{flex:1;overflow:auto;position:relative}
  .grid{display:flex;align-items:flex-start;width:max-content;padding:0 14px 40px}
  .col{display:flex;flex-direction:column;flex:none}
  .col.labels{position:sticky;left:0;z-index:20;background:var(--bg)}
  .colhead{position:sticky;top:0;z-index:10;background:var(--head);
       border-bottom:1px solid var(--line);padding:7px 9px;height:46px;
       display:flex;flex-direction:column;justify-content:center}
  .col.labels .colhead{z-index:25;border-right:1px solid var(--line)}
  .colhead .t{font-weight:650;font-size:11px;line-height:1.2}
  .colhead .s{color:var(--muted);font-size:10px;font-family:var(--mono);
       overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .colbody{position:relative}
  .col.labels .colbody{border-right:1px solid var(--line);background:var(--bg)}
  .col.panel{border-right:1px solid var(--line)}
  svg{display:block}
  .empty{padding:40px;color:var(--muted);font-family:var(--mono)}

  /* ---- single-sample modal ---- */
  .col.labels .colbody{cursor:pointer}
  .modal-back{position:fixed;inset:0;z-index:300;background:rgba(17,24,39,.45);
       display:flex;align-items:center;justify-content:center;padding:24px}
  .modal-card{background:#fff;border-radius:14px;max-width:700px;width:100%;
       max-height:90vh;display:flex;flex-direction:column;overflow:hidden;
       box-shadow:0 20px 60px rgba(0,0,0,.35)}
  .modal-head{display:flex;align-items:flex-start;gap:12px;padding:18px 20px 13px;
       border-bottom:1px solid var(--line);flex:none;background:#fff}
  .modal-id{font-family:var(--mono);font-size:18px;font-weight:650}
  .modal-meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .mstats{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:5px}
  .mstats b{color:var(--ink);font-weight:650}
  .chsec{margin-bottom:20px}
  .chsec:last-child{margin-bottom:0}
  .chh{font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);
       font-weight:700;margin-bottom:9px;padding-bottom:5px;
       border-bottom:1px solid var(--line)}
  .hstack{display:flex;height:28px;border-radius:7px;overflow:hidden;
       border:1px solid var(--line)}
  .hseg{display:flex;align-items:center;justify-content:center;font-family:var(--mono);
       font-size:10px;font-weight:600;min-width:0;overflow:hidden;white-space:nowrap}
  .hleg{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:9px}
  .hl2{display:flex;align-items:center;gap:6px;font-family:var(--mono);
       font-size:11px;color:var(--muted)}
  .hl2 b{color:var(--ink);font-weight:650}
  .hl2 .sw{width:11px;height:11px;border-radius:3px;flex:none}
  .chip{font-size:11px;background:var(--head);border:1px solid var(--line);
       border-radius:999px;padding:2px 9px;font-family:var(--mono)}
  .chip b{color:var(--muted);font-weight:600;margin-right:3px}
  #m-close{margin-left:auto;border:none;background:var(--head);width:30px;height:30px;
       border-radius:8px;font-size:14px;cursor:pointer;color:var(--muted);flex:none}
  #m-close:hover{background:var(--line)}
  .modal-body{padding:16px 20px 22px;overflow-y:auto;flex:1;min-height:0}
  .mgrid{display:grid;grid-template-columns:120px 1.1fr 1fr 1fr 1fr;
       gap:9px 14px;align-items:center}
  .mgrid .mh{font-size:10px;text-transform:uppercase;letter-spacing:.4px;
       color:var(--muted);font-weight:700;padding-bottom:5px;
       border-bottom:1px solid var(--line)}
  .mloc{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:13px}
  .mloc .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .mbar{display:flex;align-items:center;gap:9px}
  .mtrack{flex:1;height:16px;background:var(--lost);border-radius:4px;
       overflow:hidden;min-width:50px}
  .mfill{height:100%}
  .mval{font-family:var(--mono);font-size:12px;min-width:46px;text-align:right}
  .mna{color:var(--muted);font-family:var(--mono);font-size:11px}

  /* ---- expand / child columns / histogram ---- */
  .col.panel.expandable .colhead{cursor:pointer}
  .col.panel.expandable .colhead:hover{background:#e6eaf1}
  .col.panel.child{background:#fbfbff}
  .col.panel.child .colhead{background:#f1f2fb}
  .chev2{color:var(--muted);font-size:9px;margin-right:4px}
  .col.panel.expandable .colhead .t{padding-right:32px}
  .histbtn{position:absolute;top:5px;right:6px;font-size:10px;padding:1px 6px;
       border:1px solid var(--line);border-radius:5px;background:#fff;
       color:var(--accent);cursor:pointer;line-height:1.3;font-family:var(--sans)}
  .histbtn:hover{background:var(--accent-soft);border-color:var(--accent)}
  .schart{width:100%}
  .pdfbtn{font-family:var(--sans);font-size:12px;padding:5px 11px;border-radius:7px;
       border:1px solid var(--accent);background:var(--accent-soft);color:var(--accent);
       cursor:pointer;white-space:nowrap}
  .pdfbtn:hover{background:var(--accent);color:#fff}
  .slegend{display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap}
  .sl{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
  .slsw{width:13px;height:13px;border-radius:3px;background:#374151;flex:none}

  /* ---- tooltip ---- */
  #tip{position:fixed;z-index:200;pointer-events:none;display:none;
       background:#111827;color:#f9fafb;font-size:12px;line-height:1.45;
       padding:7px 10px;border-radius:7px;box-shadow:0 6px 18px rgba(0,0,0,.3);
       max-width:280px}
  #tip b{color:#fff}
  #tip .k{color:#9ca3af}
  #tip .mono{font-family:var(--mono)}
</style>
</head>
<body>
<header id="bar">
  <div class="barrow">
    <div class="ctl" style="margin-right:6px">
      <h1>Amplicon pipeline summary</h1>
      <div class="sub" id="subline"></div>
    </div>
    <div class="ctl">
      <label>Find sample</label>
      <input type="text" id="search" placeholder="id contains…" style="width:150px">
    </div>
    <div class="ctl">
      <label>Filters</label>
      <div id="filters" style="display:flex;gap:8px;flex-wrap:wrap"></div>
    </div>
    <div class="ctl">
      <label>Sort by</label>
      <select id="sort"></select>
    </div>
    <div class="ctl">
      <label>Bar height</label>
      <div class="rangewrap">
        <input type="range" id="rowh" min="5" max="44" step="1" value="17">
        <span class="rangeval" id="rowhval">17</span>
      </div>
    </div>
    <div class="ctl">
      <label>Palette</label>
      <select id="palette">
        <optgroup label="Sequential (matches PNG)">
          <option value="viridis" selected>viridis</option>
          <option value="plasma">plasma</option>
          <option value="cividis">cividis</option>
          <option value="magma">magma</option>
          <option value="inferno">inferno</option>
          <option value="turbo">turbo</option>
        </optgroup>
        <optgroup label="Categorical">
          <option value="tab10">tab10</option>
          <option value="set2">Set2</option>
          <option value="dark2">Dark2</option>
        </optgroup>
      </select>
    </div>
    <div class="ctl">
      <label>&nbsp;</label>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="count-pill" id="countpill"></span>
        <button id="reset">Reset</button>
        <button class="pdfbtn" id="grid-pdf" title="Export visible grid as PDF">⬇ PDF</button>
      </div>
    </div>
  </div>
  <div class="legend" id="legend"></div>
</header>

<div id="plot"><div class="grid" id="grid"></div></div>

<div id="modal" class="modal-back" style="display:none">
  <div class="modal-card" role="dialog" aria-modal="true">
    <div class="modal-head">
      <div style="min-width:0">
        <div class="modal-id" id="m-id"></div>
        <div class="mstats" id="m-stats"></div>
        <div class="modal-meta" id="m-meta"></div>
      </div>
      <div style="display:flex;gap:8px;align-items:flex-start;flex:none">
        <button class="pdfbtn" id="m-pdf" title="Download as PDF">⬇ PDF</button>
        <button id="m-close" title="Close (Esc)">✕</button>
      </div>
    </div>
    <div class="modal-body" id="m-body"></div>
  </div>
</div>

<div id="hmodal" class="modal-back" style="display:none">
  <div class="modal-card" style="max-width:780px">
    <div class="modal-head">
      <div style="min-width:0">
        <div class="modal-id" id="h-id"></div>
        <div class="mna" id="h-sub" style="margin-top:4px"></div>
      </div>
      <button id="h-close" title="Close (Esc)">✕</button>
    </div>
    <div class="modal-body" id="h-body"></div>
  </div>
</div>

<div id="tip"></div>

<script>
const D = __DATA_JSON__;

// ---------------------------------------------------------------- palettes
const ANCHORS = {
  viridis:["#440154","#3b528b","#21918c","#5ec962","#fde725"],
  plasma: ["#0d0887","#7e03a8","#cc4778","#f89540","#f0f921"],
  cividis:["#00204d","#31446b","#666970","#a69d75","#ffea46"],
  magma:  ["#000004","#3b0f70","#8c2981","#de4968","#fe9f6d","#fcfdbf"],
  inferno:["#000004","#420a68","#932667","#dd513a","#fca50a","#fcffa4"],
  turbo:  ["#30123b","#4146c8","#1ddfa3","#a4fc3c","#fb8022","#7a0403"]
};
const CATEG = {
  tab10:["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
         "#e377c2","#7f7f7f","#bcbd22","#17becf"],
  set2:["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f",
        "#e5c494","#b3b3b3"],
  dark2:["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02",
         "#a6761d","#666666"]
};
const CMAP_LO=0.05, CMAP_HI=0.90;            // mirrors the PNG sampling

function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),
                           parseInt(h.slice(5,7),16)];}
function rgb2hex(r){return "#"+r.map(v=>Math.round(v).toString(16)
                           .padStart(2,"0")).join("");}
function txtOn(hex){const c=hex2rgb(hex);
  return (0.299*c[0]+0.587*c[1]+0.114*c[2])>150?"#1f2933":"#ffffff";}
function rampAt(anchors,t){
  if(t<=0)return anchors[0]; if(t>=1)return anchors[anchors.length-1];
  const x=t*(anchors.length-1), i=Math.floor(x), f=x-i;
  const a=hex2rgb(anchors[i]), b=hex2rgb(anchors[i+1]);
  return rgb2hex(a.map((v,k)=>v+(b[k]-v)*f));
}
function paletteColours(name,n){
  if(CATEG[name]){const c=CATEG[name];return Array.from({length:n},(_,i)=>c[i%c.length]);}
  const anc=ANCHORS[name]||ANCHORS.viridis;
  if(n<=1)return [rampAt(anc,0.5)];
  const out=[]; for(let i=0;i<n;i++)out.push(rampAt(anc,CMAP_LO+(CMAP_HI-CMAP_LO)*i/(n-1)));
  return out;
}

// ---------------------------------------------------------------- columns
const LOCI = D.loci;
const STAGE_LABELS = {
  cutadapt:{pos:"written",  neg:"discarded", unit:"% pairs"},
  bbduk:   {pos:"retained", neg:"lost",      unit:"% reads"},
  bbmerge: {pos:"merged",   neg:"unmerged",  unit:"% pairs"}
};
function buildColumns(){
  const cols=[{type:"counts",stage:"cutadapt",title:"cutadapt",sub:"pairs written"},
              {type:"notfound"}];
  LOCI.forEach((L,li)=>{
    cols.push({type:"pct",stage:"cutadapt",locus:L,li,title:"cutadapt %",
               sub:L,expandable:true});
    if(EXPANDED.has(L)){
      cols.push({type:"pct",stage:"bbduk",locus:L,li,title:"bbduk %",sub:L,child:true});
      cols.push({type:"pct",stage:"bbmerge",locus:L,li,title:"bbmerge %",sub:L,child:true});
    }
  });
  return cols;
}
const EXPANDED = new Set();          // loci whose bbduk/bbmerge are revealed
let COLUMNS = buildColumns();
const NOTFOUND_COLOUR = "#d79b54";   // amber — reads with no primer match

// ---------------------------------------------------------------- state
const META = D.meta;
const state = {
  rowH:17, palette:"viridis", search:"", sort:"__order__",
  filters:{}                                  // col -> Set(selected); absent=all
};
let COLOURS = paletteColours(state.palette, LOCI.length);
let VISIBLE = [];                             // current ordered sample ids

const $ = s=>document.querySelector(s);

function metaVal(s,col){const r=META.rows[s];return r?(r[col]||""):"";}

// ---------------------------------------------------------------- filtering
function uniqueValues(col){
  const m=new Map();
  for(const s of D.samples){const v=metaVal(s,col)||"(blank)";
    m.set(v,(m.get(v)||0)+1);}
  return [...m.entries()].sort((a,b)=>a[0].localeCompare(b[0],undefined,{numeric:true}));
}
function passesFilters(s){
  for(const col in state.filters){
    const sel=state.filters[col]; if(!sel||sel.size===0)continue;
    const v=metaVal(s,col)||"(blank)"; if(!sel.has(v))return false;
  }
  if(state.search){const q=state.search.toLowerCase();
    if(!s.toLowerCase().includes(q))return false;}
  return true;
}
function meanPct(s,stage){
  let sum=0,k=0; for(const L of LOCI){const e=D.stages[stage][s][L];
    if(e&&e[2]!=null){sum+=e[2];k++;}} return k?sum/k:-1;
}
function totalCount(s){
  let t=0; for(const L of LOCI){const e=D.stages.cutadapt[s][L];
    if(e&&e[1]!=null)t+=e[1];} return t;
}
function notFoundOf(s){
  // initial input reads minus the cutadapt loci totals.
  // cutadapt Input_Pairs is the same raw total for every locus, so take the
  // max across loci as the input and subtract the sum of pairs written.
  let inp=0, written=0, any=false;
  for(const L of LOCI){const e=D.stages.cutadapt[s][L];
    if(e&&e[0]!=null){inp=Math.max(inp,e[0]);any=true;}
    if(e&&e[1]!=null)written+=e[1];}
  if(!any)return null;
  const nf=Math.max(0,inp-written);
  return {input:inp,written:written,nf:nf,pct:inp>0?100*nf/inp:0};
}
function computeVisible(){
  let v=D.samples.filter(passesFilters);
  const so=state.sort;
  if(so==="__count_desc__")      v.sort((a,b)=>totalCount(b)-totalCount(a));
  else if(so==="__count_asc__")  v.sort((a,b)=>totalCount(a)-totalCount(b));
  else if(so==="__worst_ca__")   v.sort((a,b)=>meanPct(a,"cutadapt")-meanPct(b,"cutadapt"));
  else if(so==="__worst_bd__")   v.sort((a,b)=>meanPct(a,"bbduk")-meanPct(b,"bbduk"));
  else if(so==="__worst_bm__")   v.sort((a,b)=>meanPct(a,"bbmerge")-meanPct(b,"bbmerge"));
  else if(so!=="__order__")      v.sort((a,b)=>metaVal(a,so)
                                   .localeCompare(metaVal(b,so),undefined,{numeric:true}));
  VISIBLE=v;
}

// ---------------------------------------------------------------- rendering
const PCT_W=120, COUNT_W=230, PAD_L=8, PAD_R=10, HEAD_H=46;
const LBL_W=Math.min(170,Math.max(70,
            8.4*Math.max(...D.samples.map(s=>s.length),6)+18));

function fmtInt(n){return n==null?"—":n.toLocaleString("en-US");}
function fmtAxis(n){n=Math.round(n);
  if(n>=1e6)return (n/1e6).toFixed(n>=1e7?0:1)+"M";
  if(n>=1e3)return (n/1e3).toFixed(n>=1e4?0:1)+"k";
  return ""+n;}
function svgEl(w,h){return `<svg width="${w}" height="${h}" `+
  `viewBox="0 0 ${w} ${h}">`;}

function renderLabels(rh){
  const n=VISIBLE.length, h=n*rh;
  const showText=rh>=9;
  const fs=Math.max(6,Math.min(11,rh-5));
  let s=svgEl(LBL_W,h);
  s+=`<rect class="hl-lbl" x="0" y="-99" width="${LBL_W}" height="${rh}" `+
     `fill="var(--hl)"/>`;
  if(showText){
    for(let i=0;i<n;i++)
      s+=`<text x="${LBL_W-10}" y="${i*rh+rh/2}" text-anchor="end" `+
         `dominant-baseline="central" font-family="var(--mono)" `+
         `font-size="${fs}" fill="var(--ink)">${VISIBLE[i]}</text>`;
  }
  s+="</svg>";
  $("#lblbody").innerHTML=s;
}

function gridLinesPct(w,h){
  let g="";
  for(const p of [25,50,75,100]){
    const x=PAD_L+(w-PAD_L-PAD_R)*p/100;
    g+=`<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="var(--line)" `+
       `stroke-width="1" stroke-dasharray="2 3"/>`;
  }
  return g;
}

function renderPanel(idx){
  const c=COLUMNS[idx], rh=state.rowH, n=VISIBLE.length, h=n*rh;
  const body=document.getElementById("body"+idx);
  if(n===0){body.innerHTML='<div class="empty">no samples</div>';return;}

  if(c.type==="counts"){
    const w=COUNT_W, innerW=w-PAD_L-PAD_R;
    let mx=0; for(const s of VISIBLE)for(const L of LOCI){
      const e=D.stages.cutadapt[s][L]; if(e&&e[1]!=null)mx=Math.max(mx,e[1]);}
    mx=mx||1;
    const nl=LOCI.length, sub=rh/nl;
    let s=svgEl(w,h);
    s+=`<rect class="hl" x="0" y="-99" width="${w}" height="${rh}" fill="var(--hl)"/>`;
    // baseline + max gridline
    s+=`<line x1="${PAD_L}" y1="0" x2="${PAD_L}" y2="${h}" stroke="var(--line)"/>`;
    s+=`<line x1="${PAD_L+innerW}" y1="0" x2="${PAD_L+innerW}" y2="${h}" `+
       `stroke="var(--line)" stroke-dasharray="2 3"/>`;
    for(let i=0;i<n;i++){
      const smp=VISIBLE[i];
      for(let li=0;li<nl;li++){
        const e=D.stages.cutadapt[smp][LOCI[li]];
        const v=(e&&e[1]!=null)?e[1]:0;
        const bw=innerW*v/mx, by=i*rh+li*sub+0.4, bh=Math.max(1,sub-0.8);
        s+=`<rect x="${PAD_L}" y="${by.toFixed(2)}" width="${bw.toFixed(2)}" `+
           `height="${bh.toFixed(2)}" fill="${COLOURS[li]}"/>`;
      }
    }
    s+="</svg>";
    body.innerHTML=s; body.dataset.max=mx; return;
  }

  if(c.type==="notfound"){
    const w=PCT_W, innerW=w-PAD_L-PAD_R;
    const barH=Math.max(2,rh*0.82), off=(rh-barH)/2;
    let s=svgEl(w,h);
    s+=`<rect class="hl" x="0" y="-99" width="${w}" height="${rh}" fill="var(--hl)"/>`;
    s+=gridLinesPct(w,h);
    for(let i=0;i<n;i++){
      const nf=notFoundOf(VISIBLE[i]); const y=i*rh+off;
      if(!nf){
        s+=`<rect x="${PAD_L}" y="${y.toFixed(2)}" width="${innerW}" `+
           `height="${barH.toFixed(2)}" fill="none" stroke="var(--line)" `+
           `stroke-dasharray="2 2"/>`;
        continue;
      }
      const v=Math.max(0,Math.min(100,nf.pct));
      const wv=innerW*v/100, wl=innerW-wv;
      s+=`<rect x="${PAD_L}" y="${y.toFixed(2)}" width="${wv.toFixed(2)}" `+
         `height="${barH.toFixed(2)}" fill="${NOTFOUND_COLOUR}"/>`;
      s+=`<rect x="${(PAD_L+wv).toFixed(2)}" y="${y.toFixed(2)}" `+
         `width="${wl.toFixed(2)}" height="${barH.toFixed(2)}" fill="var(--lost)"/>`;
    }
    s+="</svg>";
    body.innerHTML=s; return;
  }

  // percentage panel (stacked value + remainder)
  const w=PCT_W, innerW=w-PAD_L-PAD_R, st=c.stage, L=c.locus, col=COLOURS[c.li];
  const barH=Math.max(2,rh*0.82), off=(rh-barH)/2;
  let s=svgEl(w,h);
  s+=`<rect class="hl" x="0" y="-99" width="${w}" height="${rh}" fill="var(--hl)"/>`;
  s+=gridLinesPct(w,h);
  for(let i=0;i<n;i++){
    const e=D.stages[st][VISIBLE[i]][L];
    const y=i*rh+off;
    if(!e||e[2]==null){
      s+=`<rect x="${PAD_L}" y="${y.toFixed(2)}" width="${innerW}" `+
         `height="${barH.toFixed(2)}" fill="none" stroke="var(--line)" `+
         `stroke-dasharray="2 2"/>`;
      continue;
    }
    const v=Math.max(0,Math.min(100,e[2]));
    const wv=innerW*v/100, wl=innerW-wv;
    s+=`<rect x="${PAD_L}" y="${y.toFixed(2)}" width="${wv.toFixed(2)}" `+
       `height="${barH.toFixed(2)}" fill="${col}"/>`;
    s+=`<rect x="${(PAD_L+wv).toFixed(2)}" y="${y.toFixed(2)}" `+
       `width="${wl.toFixed(2)}" height="${barH.toFixed(2)}" fill="var(--lost)"/>`;
  }
  s+="</svg>";
  body.innerHTML=s;
}

function renderAll(){
  computeVisible();
  $("#rowhval").textContent=state.rowH;
  $("#countpill").textContent=`${VISIBLE.length} / ${D.samples.length} samples`;
  renderLabels(state.rowH);
  for(let i=0;i<COLUMNS.length;i++)renderPanel(i);
  updateLegend();
}

// ---------------------------------------------------------------- legend
function updateLegend(){
  const lg=$("#legend");
  let h='<div class="lg" style="color:var(--muted)">Loci:</div>';
  LOCI.forEach((L,i)=>{h+=`<div class="lg"><span class="sw" `+
     `style="background:${COLOURS[i]}"></span>${L}</div>`;});
  h+=`<div class="lg"><span class="sw" style="background:${NOTFOUND_COLOUR}">`+
     `</span>primers not found</div>`;
  h+=`<div class="lg"><span class="sw" style="background:var(--lost)"></span>`+
     `lost / discarded</div>`;
  lg.innerHTML=h;
}

// ---------------------------------------------------------------- hover
const tip=$("#tip");
function setHighlight(rowIdx){
  const y=rowIdx*state.rowH;
  document.querySelectorAll("#grid .hl").forEach(r=>r.setAttribute("y",y));
  const l=document.querySelector(".hl-lbl"); if(l)l.setAttribute("y",y);
}
function clearHighlight(){
  document.querySelectorAll("#grid .hl,.hl-lbl").forEach(r=>r.setAttribute("y",-99));
}
function colTip(c,smp){
  const head=`<b class="mono">${smp}</b>`;
  if(c.type==="notfound"){
    const nf=notFoundOf(smp);
    if(!nf)return head+`<div class="k">primers not found: no data</div>`;
    return head+
      `<div style="margin-top:2px"><span class="k">primers not found</span></div>`+
      `<div><b>${nf.pct.toFixed(1)}%</b> <span class="k">of input</span></div>`+
      `<div class="mono k">${fmtInt(nf.nf)} of ${fmtInt(nf.input)} reads · `+
      `${fmtInt(nf.written)} assigned</div>`;
  }
  if(c.type==="counts"){
    let rows="";
    LOCI.forEach((L,i)=>{const e=D.stages.cutadapt[smp][L];
      rows+=`<div><span class="sw" style="display:inline-block;width:9px;height:9px;`+
        `border-radius:2px;background:${COLOURS[i]};margin-right:5px"></span>`+
        `<span class="k">${L}</span> <span class="mono">${e?fmtInt(e[1]):"—"}</span></div>`;});
    return head+`<div class="k" style="margin:3px 0 1px">cutadapt pairs written</div>`+rows;
  }
  const e=D.stages[c.stage][smp][c.locus], lab=STAGE_LABELS[c.stage];
  if(!e||e[2]==null)return head+`<div class="k">${c.stage} · ${c.locus}: no data</div>`;
  const pct=e[2].toFixed(1);
  return head+
    `<div style="margin-top:2px"><span class="k">${c.stage} · ${c.locus}</span></div>`+
    `<div><b>${pct}%</b> <span class="k">${lab.pos}</span></div>`+
    `<div class="mono k">${fmtInt(e[1])} / ${fmtInt(e[0])} in</div>`;
}
function attachHover(bodyEl,colIdx){
  const c=COLUMNS[colIdx];
  bodyEl.addEventListener("mousemove",ev=>{
    if(VISIBLE.length===0)return;
    const rect=bodyEl.getBoundingClientRect();
    const row=Math.floor((ev.clientY-rect.top)/state.rowH);
    if(row<0||row>=VISIBLE.length){tip.style.display="none";clearHighlight();return;}
    setHighlight(row);
    const smp=VISIBLE[row];
    tip.innerHTML=colTip(c,smp);
    tip.style.display="block";
    let tx=ev.clientX+14, ty=ev.clientY+14;
    const tw=tip.offsetWidth, th=tip.offsetHeight;
    if(tx+tw>innerWidth-8)tx=ev.clientX-tw-14;
    if(ty+th>innerHeight-8)ty=ev.clientY-th-14;
    tip.style.left=tx+"px"; tip.style.top=ty+"px";
  });
  bodyEl.addEventListener("mouseleave",()=>{tip.style.display="none";clearHighlight();});
}

// ---------------------------------------------------------------- single sample
function sampleChartSVG(smp){
  const W=Math.max(380,80+LOCI.length*110), H=260, mL=42,mR=14,mT=16,mB=38;
  const pw=W-mL-mR, ph=H-mT-mB;
  const ys=v=>mT+ph*(1-v/100);
  const groupW=pw/LOCI.length, barW=Math.min(26,(groupW-18)/3);
  const stages=[["cutadapt",1.0],["bbduk",0.6],["bbmerge",0.38]];
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" xmlns="http://www.w3.org/2000/svg">`;
  for(let g=0;g<=4;g++){const v=g*25,y=ys(v);
    s+=`<line x1="${mL}" y1="${y}" x2="${W-mR}" y2="${y}" stroke="#dce1e8" `+
       `stroke-dasharray="2 3"/>`;
    s+=`<text x="${mL-7}" y="${y}" text-anchor="end" dominant-baseline="central" `+
       `font-family="monospace" font-size="10" fill="#6b7280">${v}%</text>`;}
  LOCI.forEach((L,li)=>{
    const col=COLOURS[li], gx=mL+groupW*li+groupW/2, x0=gx-(barW*3+8)/2;
    stages.forEach((st,si)=>{
      const e=D.stages[st[0]][smp][L], x=x0+si*(barW+4);
      if(!e||e[2]==null){
        s+=`<rect x="${x.toFixed(1)}" y="${(mT+ph-2).toFixed(1)}" `+
           `width="${barW}" height="2" fill="#dce1e8"/>`;
      }else{
        const v=Math.max(0,Math.min(100,e[2])), bh=ph*v/100;
        s+=`<rect x="${x.toFixed(1)}" y="${(mT+ph-bh).toFixed(1)}" width="${barW}" `+
           `height="${bh.toFixed(1)}" fill="${col}" fill-opacity="${st[1]}"/>`;
        s+=`<text x="${(x+barW/2).toFixed(1)}" y="${(mT+ph-bh-3).toFixed(1)}" `+
           `text-anchor="middle" font-family="monospace" font-size="9" `+
           `fill="#1f2933">${e[2].toFixed(0)}</text>`;
      }
    });
    s+=`<text x="${gx}" y="${mT+ph+17}" text-anchor="middle" `+
       `font-family="monospace" font-size="11" fill="#1f2933">${L}</text>`;
  });
  s+=`<line x1="${mL}" y1="${mT+ph}" x2="${W-mR}" y2="${mT+ph}" stroke="#1f2933"/>`;
  s+="</svg>";
  return s;
}
function sampleCountsSVG(smp){
  const cats=[];
  LOCI.forEach((L,li)=>{const e=D.stages.cutadapt[smp][L];
    cats.push({label:L, value:(e&&e[1]!=null)?e[1]:0, na:!(e&&e[1]!=null),
               col:COLOURS[li]});});
  const W=600, H=260, mL=58,mR=14,mT=18,mB=38;
  const pw=W-mL-mR, ph=H-mT-mB;
  const ymax=Math.max(1,...cats.map(c=>c.value));
  const ys=v=>mT+ph*(1-v/ymax);
  const slotW=pw/cats.length, barW=Math.min(52,slotW-20);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" xmlns="http://www.w3.org/2000/svg">`;
  for(let g=0;g<=4;g++){const yv=ymax*g/4,y=ys(yv);
    s+=`<line x1="${mL}" y1="${y}" x2="${W-mR}" y2="${y}" stroke="#dce1e8" `+
       `stroke-dasharray="2 3"/>`;
    s+=`<text x="${mL-8}" y="${y}" text-anchor="end" dominant-baseline="central" `+
       `font-family="monospace" font-size="10" fill="#6b7280">${fmtAxis(yv)}</text>`;}
  cats.forEach((c,i)=>{
    const cx=mL+slotW*i+slotW/2, x=cx-barW/2, bh=Math.max(0,ph*c.value/ymax);
    s+=`<rect x="${x.toFixed(1)}" y="${(mT+ph-bh).toFixed(1)}" width="${barW.toFixed(1)}" `+
       `height="${bh.toFixed(1)}" fill="${c.col}" fill-opacity="0.9"/>`;
    s+=`<text x="${cx.toFixed(1)}" y="${(mT+ph-bh-3).toFixed(1)}" text-anchor="middle" `+
       `font-family="monospace" font-size="9" fill="#1f2933">`+
       `${c.na?"—":fmtInt(c.value)}</text>`;
    s+=`<text x="${cx.toFixed(1)}" y="${mT+ph+17}" text-anchor="middle" `+
       `font-family="monospace" font-size="11" fill="#1f2933">${c.label}</text>`;
  });
  s+=`<line x1="${mL}" y1="${mT+ph}" x2="${W-mR}" y2="${mT+ph}" stroke="#1f2933"/>`;
  s+="</svg>";
  return s;
}
function sampleAllocHTML(smp){
  const nf=notFoundOf(smp);
  if(!nf||nf.input<=0)return '<div class="mna">no data</div>';
  const input=nf.input, segs=[];
  LOCI.forEach((L,li)=>{const e=D.stages.cutadapt[smp][L];
    const c=(e&&e[1]!=null)?e[1]:0;
    if(c>0)segs.push({label:L,count:c,pct:100*c/input,col:COLOURS[li]});});
  if(nf.nf>0)segs.push({label:"not found",count:nf.nf,
                        pct:100*nf.nf/input,col:NOTFOUND_COLOUR});
  let bar=`<div class="hstack">`;
  segs.forEach(s=>{
    bar+=`<div class="hseg" style="width:${s.pct}%;background:${s.col};color:${txtOn(s.col)}" `+
      `title="${s.label}: ${s.pct.toFixed(1)}% (${fmtInt(s.count)} reads)">`+
      `${s.pct>=8?s.pct.toFixed(0)+"%":""}</div>`;
  });
  bar+=`</div>`;
  const leg=`<div class="hleg">`+segs.map(s=>
    `<span class="hl2"><span class="sw" style="background:${s.col}"></span>`+
    `${s.label} <b>${s.pct.toFixed(1)}%</b></span>`).join("")+`</div>`;
  return bar+leg;
}
function openSample(smp){
  document.getElementById("m-id").textContent=smp;
  const nf=notFoundOf(smp);
  document.getElementById("m-stats").innerHTML= nf?
    `<b>${fmtInt(nf.input)}</b> reads · <b>${(100-nf.pct).toFixed(1)}%</b> primers found`:
    "";
  const row=META.rows[smp]||{};
  const chips=META.columns.map(c=>
    `<span class="chip"><b>${c}</b> ${row[c]||"—"}</span>`).join("");
  document.getElementById("m-meta").innerHTML=
    chips||'<span class="mna">no metadata for this sample</span>';

  const op=[1,.6,.38], names=["cutadapt","bbduk","bbmerge"];
  const leg=`<div class="slegend">`+names.map((nm,i)=>
    `<span class="sl"><span class="slsw" style="opacity:${op[i]}"></span>${nm} %</span>`)
    .join("")+`</div>`;
  document.getElementById("m-body").innerHTML=
    `<div class="chsec"><div class="chh">Read allocation (% of input)</div>`+
    `${sampleAllocHTML(smp)}</div>`+
    `<div class="chsec"><div class="chh">Stage percentages</div>${leg}`+
    `<div class="schart">${sampleChartSVG(smp)}</div></div>`+
    `<div class="chsec"><div class="chh">Read counts per locus</div>`+
    `<div class="schart">${sampleCountsSVG(smp)}</div></div>`;
  document.getElementById("modal").style.display="flex";
}
function closeModal(){document.getElementById("modal").style.display="none";}

// ---------------------------------------------------------------- expand / histogram
function toggleLocus(L){
  if(EXPANDED.has(L))EXPANDED.delete(L); else EXPANDED.add(L);
  COLUMNS=buildColumns();
  buildGrid();
  renderAll();
}
function histStats(sizes,counts){
  const total=counts.reduce((a,b)=>a+b,0)||1;
  let mean=0; for(let i=0;i<sizes.length;i++)mean+=sizes[i]*counts[i]; mean/=total;
  let cum=0, median=sizes[sizes.length-1];
  for(let i=0;i<sizes.length;i++){cum+=counts[i];if(cum>=total/2){median=sizes[i];break;}}
  let mi=0; for(let i=1;i<counts.length;i++)if(counts[i]>counts[mi])mi=i;
  return {total,mean,median,mode:sizes[mi]};
}
function histSVG(sizes,counts,m,col){
  const W=720,H=360,mL=58,mR=16,mT=18,mB=46, pw=W-mL-mR, ph=H-mT-mB;
  const xmin=sizes[0], xmax=sizes[sizes.length-1], span=(xmax-xmin)||1;
  const ymax=Math.max(...counts)||1;
  const xs=v=>mL+pw*(v-xmin)/span, ys=v=>mT+ph*(1-v/ymax);
  const bw=Math.max(0.7,pw/span);
  let s=svgEl(W,H);
  for(let g=0;g<=4;g++){const yv=ymax*g/4, y=ys(yv);
    s+=`<line x1="${mL}" y1="${y}" x2="${W-mR}" y2="${y}" stroke="var(--line)" `+
       `stroke-dasharray="2 3"/>`;
    s+=`<text x="${mL-8}" y="${y}" text-anchor="end" dominant-baseline="central" `+
       `font-family="var(--mono)" font-size="10" fill="var(--muted)">`+
       `${Math.round(yv).toLocaleString()}</text>`;}
  for(let i=0;i<sizes.length;i++){
    const x=xs(sizes[i]), bh=ph*counts[i]/ymax;
    s+=`<rect x="${(x-bw/2).toFixed(2)}" y="${(mT+ph-bh).toFixed(2)}" `+
       `width="${bw.toFixed(2)}" height="${bh.toFixed(2)}" fill="${col}" fill-opacity="0.85"/>`;
  }
  const nt=6;
  for(let t=0;t<=nt;t++){const xv=xmin+span*t/nt, x=xs(xv);
    s+=`<text x="${x}" y="${H-mB+16}" text-anchor="middle" font-family="var(--mono)" `+
       `font-size="10" fill="var(--muted)">${Math.round(xv)}</text>`;}
  s+=`<text x="${mL+pw/2}" y="${H-6}" text-anchor="middle" font-size="11" `+
     `fill="var(--ink)">insert size (bp)</text>`;
  const mk=[["mean",m.mean,"#e2483b"],["median",m.median,"#e08a1e"],
            ["mode",m.mode,"#2a9d54"]];
  mk.forEach((mm,idx)=>{const x=xs(mm[1]);
    s+=`<line x1="${x}" y1="${mT}" x2="${x}" y2="${mT+ph}" stroke="${mm[2]}" `+
       `stroke-width="1.4" stroke-dasharray="5 3"/>`;
    s+=`<text x="${x+4}" y="${mT+12+idx*13}" font-size="10" font-family="var(--mono)" `+
       `fill="${mm[2]}">${mm[0]} ${Math.round(mm[1])}</text>`;});
  s+=`<line x1="${mL}" y1="${mT+ph}" x2="${W-mR}" y2="${mT+ph}" stroke="var(--ink)"/>`;
  s+=`<line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT+ph}" stroke="var(--ink)"/>`;
  s+="</svg>";
  return s;
}
function openHist(L){
  const hd=D.hist&&D.hist[L]; if(!hd)return;
  const m=histStats(hd.sizes,hd.counts);
  document.getElementById("h-id").textContent=L+" — insert-size histogram";
  document.getElementById("h-sub").textContent=
    `${m.total.toLocaleString()} read pairs · mean ${m.mean.toFixed(1)} · `+
    `median ${m.median} · mode ${m.mode} bp`;
  document.getElementById("h-body").innerHTML=
    `<div class="schart">${histSVG(hd.sizes,hd.counts,m,COLOURS[LOCI.indexOf(L)])}</div>`;
  document.getElementById("hmodal").style.display="flex";
}
function closeHist(){document.getElementById("hmodal").style.display="none";}

// ---------------------------------------------------------------- DOM build
function buildGrid(){
  const grid=$("#grid"); grid.innerHTML="";
  // label column
  const lc=document.createElement("div"); lc.className="col labels";
  lc.innerHTML=`<div class="colhead"><div class="t">sample</div>`+
    `<div class="s">click for detail</div></div>`+
    `<div class="colbody" id="lblbody" style="width:${LBL_W}px"></div>`;
  grid.appendChild(lc);
  const lblbody=lc.querySelector("#lblbody");
  const rowAt=ev=>{
    const r=lblbody.getBoundingClientRect();
    return Math.floor((ev.clientY-r.top)/state.rowH);
  };
  lblbody.addEventListener("mousemove",ev=>{
    const row=rowAt(ev);
    if(row>=0&&row<VISIBLE.length)setHighlight(row); else clearHighlight();
  });
  lblbody.addEventListener("mouseleave",clearHighlight);
  lblbody.addEventListener("click",ev=>{
    const row=rowAt(ev);
    if(row>=0&&row<VISIBLE.length)openSample(VISIBLE[row]);
  });
  // panels
  COLUMNS.forEach((c,i)=>{
    const w=c.type==="counts"?COUNT_W:PCT_W;
    const el=document.createElement("div");
    el.className="col panel"+(c.child?" child":"")+(c.expandable?" expandable":"");
    let top,bot;
    if(c.type==="counts"){top=c.title;bot=c.sub;}            // cutadapt / pairs written
    else if(c.type==="notfound"){top="primers";bot="not found · % input";}
    else if(c.expandable){
      const open=EXPANDED.has(c.locus);
      top=`<span class="chev2">${open?"▾":"▸"}</span>${c.locus}`;
      bot=`${c.stage} · ${STAGE_LABELS[c.stage].unit}`;
    } else {top=c.locus;bot=`${c.stage} · ${STAGE_LABELS[c.stage].unit}`;}

    const hasHist=c.expandable && D.hist && D.hist[c.locus];
    const histBtn=hasHist?`<button class="histbtn" title="Insert-size histogram">`+
      `hist</button>`:"";

    el.innerHTML=`<div class="colhead"><div class="t">${top}</div>`+
      `<div class="s">${bot}</div>${histBtn}</div>`+
      `<div class="colbody" id="body${i}" style="width:${w}px"></div>`;
    grid.appendChild(el);

    if(c.expandable){
      const head=el.querySelector(".colhead");
      head.addEventListener("click",ev=>{
        if(ev.target.classList.contains("histbtn"))return;
        toggleLocus(c.locus);
      });
    }
    const hb=el.querySelector(".histbtn");
    if(hb)hb.addEventListener("click",ev=>{ev.stopPropagation();openHist(c.locus);});
    attachHover(el.querySelector(".colbody"),i);
  });
}

// ---------------------------------------------------------------- filters UI
function buildFilters(){
  const host=$("#filters"); host.innerHTML="";
  if(!META.columns.length){
    host.innerHTML='<span class="sub">no metadata loaded</span>'; return;}
  META.columns.forEach(col=>{
    const vals=uniqueValues(col);
    const d=document.createElement("details"); d.className="filter";
    d.innerHTML=`<summary><span>${col}</span>`+
      `<span class="fbadge" data-badge="${col}" style="display:none"></span>`+
      `<span class="chev">▾</span></summary>`;
    const pop=document.createElement("div"); pop.className="pop";
    pop.innerHTML=`<div class="pophead"><button data-all="${col}">All</button>`+
      `<button data-none="${col}">None</button></div><div class="opts"></div>`;
    const opts=pop.querySelector(".opts");
    vals.forEach(([v,k])=>{
      const o=document.createElement("label"); o.className="opt";
      o.innerHTML=`<input type="checkbox" checked data-col="${col}" `+
        `value="${v.replace(/"/g,'&quot;')}"><span class="val">${v}</span>`+
        `<span class="n">${k}</span>`;
      opts.appendChild(o);
    });
    d.appendChild(pop);
    host.appendChild(d);
  });

  host.addEventListener("change",e=>{
    const cb=e.target; if(cb.dataset.col===undefined)return;
    const col=cb.dataset.col;
    const boxes=[...host.querySelectorAll(`input[data-col="${col}"]`)];
    const sel=new Set(boxes.filter(b=>b.checked).map(b=>b.value));
    if(sel.size===boxes.length)delete state.filters[col];
    else state.filters[col]=sel;
    updateBadge(col,boxes,sel);
    renderAll();
  });
  host.addEventListener("click",e=>{
    const all=e.target.dataset?.all, none=e.target.dataset?.none;
    const col=all||none; if(!col)return;
    const boxes=[...host.querySelectorAll(`input[data-col="${col}"]`)];
    boxes.forEach(b=>b.checked=!!all);
    const sel=new Set(boxes.filter(b=>b.checked).map(b=>b.value));
    if(all)delete state.filters[col]; else state.filters[col]=sel;
    updateBadge(col,boxes,sel); renderAll();
  });
}
function updateBadge(col,boxes,sel){
  const b=document.querySelector(`[data-badge="${col}"]`); if(!b)return;
  if(sel.size===boxes.length){b.style.display="none";}
  else{b.style.display="inline-block";b.textContent=sel.size+"/"+boxes.length;}
}

// ---------------------------------------------------------------- sort UI
function buildSort(){
  const sel=$("#sort");
  const opts=[["__order__","Original order"],
    ["__count_desc__","Read count ↓"],["__count_asc__","Read count ↑"],
    ["__worst_ca__","Lowest cutadapt %"],["__worst_bd__","Lowest bbduk %"],
    ["__worst_bm__","Lowest bbmerge %"]];
  META.columns.forEach(c=>opts.push(["meta:"+c,"meta · "+c]));
  sel.innerHTML=opts.map(([v,t])=>`<option value="${v}">${t}</option>`).join("");
  sel.value="__order__";
}

// ---------------------------------------------------------------- PDF export
// Builds a self-contained print page in a hidden iframe → browser Save as PDF.

function printPage(html){
  const iframe=document.createElement("iframe");
  // Must have real dimensions — width:0/height:0 causes a blank first page in
  // most browsers because the layout engine renders nothing before print fires.
  iframe.style.cssText="position:fixed;left:-9999px;top:0;width:800px;height:600px;border:0;visibility:hidden";
  document.body.appendChild(iframe);
  iframe.contentDocument.open();
  iframe.contentDocument.write(html);
  iframe.contentDocument.close();
  // Wait for layout to settle before triggering print
  setTimeout(()=>{
    try{iframe.contentWindow.focus();iframe.contentWindow.print();}
    finally{setTimeout(()=>{if(document.body.contains(iframe))document.body.removeChild(iframe);},5000);}
  },600);
}

function exportSamplePDF(smp){
  const bodyHTML=document.getElementById("m-body").innerHTML;
  const statsHTML=document.getElementById("m-stats").innerHTML;
  const metaHTML=document.getElementById("m-meta").innerHTML;
  const html=`<!DOCTYPE html><html><head><meta charset="utf-8">
<title>${smp} — pipeline summary</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;font-size:12px;color:#1f2933;background:#fff;padding:24px 28px}
h1{font-size:16px;font-weight:700;font-family:monospace;margin-bottom:4px}
.stats{font-family:monospace;font-size:12px;color:#6b7280;margin-bottom:6px}
.stats b{color:#1f2933}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:20px}
.chip{font-size:10px;background:#eef1f5;border:1px solid #dce1e8;border-radius:999px;padding:2px 8px;font-family:monospace}
.chip b{color:#6b7280;font-weight:600;margin-right:2px}
.chsec{margin-bottom:20px}
.chh{font-size:10px;text-transform:uppercase;letter-spacing:.4px;color:#6b7280;font-weight:700;margin-bottom:7px;padding-bottom:4px;border-bottom:1px solid #dce1e8}
.hstack{display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid #dce1e8;margin-bottom:8px}
.hseg{display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:10px;font-weight:600;min-width:0;overflow:hidden;white-space:nowrap}
.hleg{display:flex;flex-wrap:wrap;gap:5px 14px}
.hl2{display:flex;align-items:center;gap:5px;font-family:monospace;font-size:10px;color:#6b7280}
.hl2 b{color:#1f2933;font-weight:650}
.sw{width:10px;height:10px;border-radius:2px;flex:none;display:inline-block}
.slegend{display:flex;gap:14px;margin-bottom:8px;flex-wrap:wrap}
.sl{display:flex;align-items:center;gap:5px;font-size:10px;color:#6b7280}
.slsw{width:11px;height:11px;border-radius:2px;background:#374151;display:inline-block}
svg{display:block;width:100%}
@page{size:A4 portrait;margin:14mm}
</style></head><body>
<h1>${smp}</h1>
<div class="stats">${statsHTML}</div>
<div class="chips">${metaHTML}</div>
${bodyHTML}
</body></html>`;
  printPage(html);
}

function exportGridPDF(){
  const rh=Math.max(state.rowH,10);
  const n=VISIBLE.length;
  if(n===0){alert("No visible samples to export.");return;}

  const INK="#1f2933",MUT="#6b7280",LINE="#dce1e8",LOST="#dde2e8",BG="#ffffff";
  const colW=COLUMNS.map(c=>c.type==="counts"?COUNT_W:PCT_W);
  const totalW=LBL_W+colW.reduce((a,b)=>a+b,0);
  const HEAD=46, totalH=HEAD+n*rh+4;

  let s=`<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${totalH}" `+
        `viewBox="0 0 ${totalW} ${totalH}">`;
  s+=`<rect width="${totalW}" height="${totalH}" fill="${BG}"/>`;

  // header
  s+=`<rect width="${totalW}" height="${HEAD}" fill="#eef1f5"/>`;
  s+=`<text x="${LBL_W/2}" y="16" text-anchor="middle" font-family="monospace" `+
     `font-size="10" font-weight="700" fill="${INK}">sample</text>`;
  let hcx=LBL_W;
  COLUMNS.forEach((c,i)=>{
    const w=colW[i],mid=hcx+w/2;
    let top,bot;
    if(c.type==="counts"){top="cutadapt";bot="pairs written";}
    else if(c.type==="notfound"){top="primers";bot="not found · %";}
    else{top=c.locus;bot=c.stage+" %";}
    s+=`<text x="${mid}" y="16" text-anchor="middle" font-family="monospace" `+
       `font-size="10" font-weight="700" fill="${INK}">${top}</text>`;
    s+=`<text x="${mid}" y="28" text-anchor="middle" font-family="monospace" `+
       `font-size="8" fill="${MUT}">${bot}</text>`;
    hcx+=w;
  });
  s+=`<line x1="0" y1="${HEAD}" x2="${totalW}" y2="${HEAD}" stroke="${LINE}"/>`;

  // counts column scale
  let cmx=0;
  VISIBLE.forEach(smp=>LOCI.forEach(L=>{
    const e=D.stages.cutadapt[smp][L];
    if(e&&e[1]!=null)cmx=Math.max(cmx,e[1]);}));
  cmx=cmx||1;
  const nl=LOCI.length;

  VISIBLE.forEach((smp,ri)=>{
    const y0=HEAD+ri*rh;
    if(ri%2===1)s+=`<rect x="0" y="${y0}" width="${totalW}" height="${rh}" fill="#f9fafb"/>`;
    if(rh>=8){
      const fs=Math.max(6,Math.min(10,rh-4));
      s+=`<text x="${LBL_W-6}" y="${y0+rh/2}" text-anchor="end" `+
         `dominant-baseline="central" font-family="monospace" font-size="${fs}" `+
         `fill="${INK}">${smp}</text>`;
    }
    let cx=LBL_W;
    COLUMNS.forEach((c,ci)=>{
      const w=colW[ci],innerW=w-PAD_L-PAD_R;
      const barH=Math.max(2,rh*0.82),off=(rh-barH)/2;
      if(c.type==="counts"){
        const sub=rh/nl;
        LOCI.forEach((L,li)=>{
          const e=D.stages.cutadapt[smp][L];
          const v=(e&&e[1]!=null)?e[1]:0;
          const bw=innerW*v/cmx,by=y0+li*sub+0.4,bh2=Math.max(1,sub-0.8);
          s+=`<rect x="${cx+PAD_L}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" `+
             `height="${bh2.toFixed(1)}" fill="${COLOURS[li]}"/>`;
        });
      } else if(c.type==="notfound"){
        const nf=notFoundOf(smp);
        if(nf){
          const v=Math.max(0,Math.min(100,nf.pct)),wv=innerW*v/100;
          s+=`<rect x="${cx+PAD_L}" y="${(y0+off).toFixed(1)}" `+
             `width="${wv.toFixed(1)}" height="${barH.toFixed(1)}" fill="${NOTFOUND_COLOUR}"/>`;
          s+=`<rect x="${(cx+PAD_L+wv).toFixed(1)}" y="${(y0+off).toFixed(1)}" `+
             `width="${(innerW-wv).toFixed(1)}" height="${barH.toFixed(1)}" fill="${LOST}"/>`;
        }
      } else {
        const e=D.stages[c.stage][smp][c.locus];
        if(!e||e[2]==null){
          s+=`<rect x="${cx+PAD_L}" y="${(y0+off).toFixed(1)}" width="${innerW}" `+
             `height="${barH.toFixed(1)}" fill="none" stroke="${LINE}" stroke-dasharray="2 2"/>`;
        } else {
          const v=Math.max(0,Math.min(100,e[2])),wv=innerW*v/100;
          s+=`<rect x="${cx+PAD_L}" y="${(y0+off).toFixed(1)}" width="${wv.toFixed(1)}" `+
             `height="${barH.toFixed(1)}" fill="${COLOURS[c.li]}"/>`;
          s+=`<rect x="${(cx+PAD_L+wv).toFixed(1)}" y="${(y0+off).toFixed(1)}" `+
             `width="${(innerW-wv).toFixed(1)}" height="${barH.toFixed(1)}" fill="${LOST}"/>`;
        }
      }
      s+=`<line x1="${cx+w}" y1="${y0}" x2="${cx+w}" y2="${y0+rh}" stroke="${LINE}"/>`;
      cx+=w;
    });
  });
  s+="</svg>";

  let legHtml=`<div style="display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:10px;font-family:monospace;font-size:11px;color:#6b7280">`;
  LOCI.forEach((L,i)=>legHtml+=`<span style="display:flex;align-items:center;gap:5px">`+
    `<span style="width:11px;height:11px;border-radius:3px;background:${COLOURS[i]};display:inline-block"></span>${L}</span>`);
  legHtml+=`<span style="display:flex;align-items:center;gap:5px">`+
    `<span style="width:11px;height:11px;border-radius:3px;background:${NOTFOUND_COLOUR};display:inline-block"></span>primers not found</span>`;
  legHtml+=`<span style="display:flex;align-items:center;gap:5px">`+
    `<span style="width:11px;height:11px;border-radius:3px;background:${LOST};display:inline-block"></span>lost/discarded</span></div>`;

  const title=`Amplicon pipeline summary — ${n} samples · ${LOCI.length} loci`;
  const filterStr=Object.keys(state.filters).length>0?
    Object.entries(state.filters).map(([k,v])=>k+"="+[...v].join("|")).join(", "):"none";
  const html=`<!DOCTYPE html><html><head><meta charset="utf-8">
<title>${title}</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;font-size:12px;background:#fff;padding:16px}
h1{font-size:13px;font-weight:700;margin-bottom:3px;color:#1f2933}
.sub{font-size:10px;color:#6b7280;margin-bottom:12px;font-family:monospace}
svg{display:block;max-width:100%;height:auto}
@page{size:auto;margin:10mm}
</style></head><body>
<h1>${title}</h1>
<div class="sub">${new Date().toLocaleDateString()} · palette: ${state.palette} · bar height: ${state.rowH}px · filters: ${filterStr}</div>
${s}
${legHtml}
</body></html>`;
  printPage(html);
}

// ---------------------------------------------------------------- events
function wire(){
  $("#search").addEventListener("input",e=>{state.search=e.target.value.trim();renderAll();});
  $("#sort").addEventListener("change",e=>{
    let v=e.target.value; state.sort=v.startsWith("meta:")?v.slice(5):v; renderAll();});
  $("#rowh").addEventListener("input",e=>{state.rowH=+e.target.value;renderAll();});
  $("#palette").addEventListener("change",e=>{
    state.palette=e.target.value;
    COLOURS=paletteColours(state.palette,LOCI.length); renderAll();});
  $("#reset").addEventListener("click",()=>{
    state.search=""; state.sort="__order__"; state.rowH=17;
    state.palette="viridis"; state.filters={}; EXPANDED.clear();
    COLOURS=paletteColours("viridis",LOCI.length);
    $("#search").value=""; $("#sort").value="__order__";
    $("#rowh").value=17; $("#palette").value="viridis";
    document.querySelectorAll('#filters input[type=checkbox]').forEach(b=>b.checked=true);
    document.querySelectorAll('[data-badge]').forEach(b=>b.style.display="none");
    COLUMNS=buildColumns(); buildGrid(); renderAll();
  });

  // filters: only one open at a time, and close when clicking outside
  const fhost=$("#filters");
  fhost.querySelectorAll("details.filter").forEach(d=>{
    d.addEventListener("toggle",()=>{
      if(d.open)fhost.querySelectorAll("details.filter[open]")
        .forEach(o=>{if(o!==d)o.open=false;});
    });
  });
  document.addEventListener("click",e=>{
    if($("#modal").style.display!=="none"||$("#hmodal").style.display!=="none")return;
    fhost.querySelectorAll("details.filter[open]")
      .forEach(d=>{if(!d.contains(e.target))d.open=false;});
  });

  // single-sample modal close: button, backdrop, Esc
  $("#m-close").addEventListener("click",closeModal);
  $("#modal").addEventListener("click",e=>{if(e.target.id==="modal")closeModal();});
  $("#m-pdf").addEventListener("click",()=>{
    const smp=document.getElementById("m-id").textContent;
    if(smp)exportSamplePDF(smp);
  });
  // grid PDF
  $("#grid-pdf").addEventListener("click",exportGridPDF);
  // histogram modal close
  $("#h-close").addEventListener("click",closeHist);
  $("#hmodal").addEventListener("click",e=>{if(e.target.id==="hmodal")closeHist();});
  document.addEventListener("keydown",e=>{
    if(e.key==="Escape"){closeModal();closeHist();
      $("#filters").querySelectorAll("details.filter[open]").forEach(d=>d.open=false);}
  });
}

// ---------------------------------------------------------------- init
function init(){
  $("#subline").textContent=
    `${D.samples.length} samples · ${LOCI.length} loci (${LOCI.join(", ")}) `+
    `· stages: cutadapt → bbduk → bbmerge`;
  buildFilters(); buildSort(); buildGrid(); wire(); renderAll();
}
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-locus pipeline TSVs and regenerate all summary plots. "
            "Works with output from both amplicon_pipeline.py and amplicon_pipeline.pbs."
        ),
    )
    parser.add_argument(
        "-b", "--base-path", required=True, metavar="DIR",
        help="Root output directory used for the pipeline run.",
    )
    parser.add_argument(
        "-p", "--primers", default=None, metavar="FILE",
        help="primers.txt path. [default: <base_path>/primers.txt or ./primers.txt]",
    )
    parser.add_argument(
        "-m", "--metadata", default=None, metavar="FILE",
        help="Sample metadata CSV/TSV for the interactive HTML (needs a "
             "sample_id column plus any extra columns). "
             "[default: <base_path>/metadata.csv if present]",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip writing pipeline_summary.html.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)

    if not os.path.isdir(base):
        sys.exit(f"ERROR: base path not found: {base}")

    # Locate primers.txt
    if args.primers:
        primers_path = args.primers
    elif os.path.exists(os.path.join(base, "primers.txt")):
        primers_path = os.path.join(base, "primers.txt")
    elif os.path.exists("primers.txt"):
        primers_path = "primers.txt"
    else:
        sys.exit(
            "ERROR: primers.txt not found. Specify with -p or place in "
            "<base_path> or current directory."
        )

    primers = load_primers(primers_path)
    loci    = [p["locus"] for p in primers]

    print(f"\nBase path : {base}")
    print(f"Primers   : {primers_path}  ({len(primers)} loci: {', '.join(loci)})")

    # ── Load stats for each stage ──────────────────────────────────────────
    print("\nLoading cutadapt stats...")
    ca_samples, ca_stats = load_stage_stats(
        base, "cutadapt", loci,
        ["Input_Pairs", "Pairs_Written", "Pct_Written"],
    )

    print("Loading bbduk stats...")
    bbd_samples, bbd_stats = load_stage_stats(
        base, "bbduk", loci,
        ["Input_Reads", "Result_Reads", "Pct_Remaining"],
    )

    print("Loading bbmerge stats...")
    bbm_samples, bbm_stats = load_stage_stats(
        base, "bbmerge", loci,
        ["Pairs", "Joined", "Pct_Joined"],
    )

    # Use the union of all sample lists, preserving order from cutadapt
    # (the first stage) then adding any that appear only in later stages.
    seen: set[str] = set()
    samples: list[str] = []
    for s in ca_samples + bbd_samples + bbm_samples:
        if s not in seen:
            samples.append(s)
            seen.add(s)

    if not samples:
        sys.exit("ERROR: No samples found in any TSV file.")

    print(f"\nSamples found: {len(samples)}")

    # ── Generate plots ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    pipeline_png = os.path.join(base, "pipeline_summary.png")
    ca_png       = os.path.join(base, "cutadapt_summary.png")
    bbd_bbm_png  = os.path.join(base, "bbduk_bbmerge_summary.png")

    for label, fn, args_ in [
        ("pipeline_summary",      plot_pipeline_summary,
         (samples, primers, ca_stats, bbd_stats, bbm_stats, pipeline_png)),
        ("cutadapt_summary",      plot_cutadapt_summary,
         (samples, primers, ca_stats, ca_png)),
        ("bbduk_bbmerge_summary", plot_bbduk_bbmerge_summary,
         (samples, primers, bbd_stats, bbm_stats, bbd_bbm_png)),
    ]:
        print(f"\n{label}...")
        try:
            fn(*args_)
        except Exception as exc:
            print(f"  WARNING: {label} failed: {exc}")

    print("\nInsert-size histograms...")
    try:
        plot_insert_histograms(primers, base)
    except Exception as exc:
        print(f"  WARNING: insert histogram failed: {exc}")

    # ── Interactive HTML ───────────────────────────────────────────────────
    html_path = os.path.join(base, "pipeline_summary.html")
    if not args.no_html:
        print("\nInteractive HTML...")
        # Locate metadata: explicit -m, else <base>/metadata.csv
        if args.metadata:
            meta_path = args.metadata
        elif os.path.exists(os.path.join(base, "metadata.csv")):
            meta_path = os.path.join(base, "metadata.csv")
        else:
            meta_path = None
            print("  No metadata file (-m / <base>/metadata.csv) — "
                  "HTML will have no metadata filters.")

        try:
            meta = load_metadata(meta_path, samples) if meta_path else None
            write_html_summary(samples, primers, ca_stats, bbd_stats, bbm_stats,
                               meta, base, html_path)
        except Exception as exc:
            print(f"  WARNING: HTML summary failed: {exc}")

    print("\nAll done.")
    print(f"  pipeline_summary.png      -> {pipeline_png}")
    print(f"  cutadapt_summary.png      -> {ca_png}")
    print(f"  bbduk_bbmerge_summary.png -> {bbd_bbm_png}")
    if not args.no_html:
        print(f"  pipeline_summary.html     -> {html_path}")
    for p in primers:
        print(f"  {p['locus']} insert hist        -> "
              f"{os.path.join(base, p['locus'] + '_histogram.png')}")


if __name__ == "__main__":
    main()
