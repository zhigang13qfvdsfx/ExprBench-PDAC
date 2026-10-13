#!/usr/bin/env python
"""Generate a deeply redesigned Nature-style figure set for the PDAC manuscript.

This script intentionally writes to a new output directory and does not overwrite
the previous default or Nature-style figure sets. It uses the same local result
tables, but redesigns the figure grammar to reduce repetitive bar/box panels and
introduce Nature-like multi-panel storytelling: schematic summaries, alluvial
flows, raincloud/violin panels, lollipop plots, dot matrices, forest plots,
network diagrams, and compact heatmaps.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


COL = {
    "blue": "#2E6FBB",
    "sky": "#65B7D4",
    "teal": "#00A087",
    "orange": "#E64B35",
    "gold": "#F2B701",
    "purple": "#7E57C2",
    "magenta": "#B8529E",
    "red": "#B2182B",
    "green": "#4DBD74",
    "ink": "#111827",
    "muted": "#5B6675",
    "line": "#C8D0DA",
    "grid": "#E8EDF3",
    "pale": "#F7F9FC",
}

SUBTYPE = {1: COL["blue"], 2: COL["orange"], 3: COL["teal"]}
RISK = {"Low": COL["teal"], "High": COL["orange"]}
MODEL_COLORS = {
    "Clinical-Cox": "#8DA0CB",
    "Expr-Penalized-Cox": COL["blue"],
    "Expr-DeepSurv": COL["sky"],
    "LateFusion-Cox": COL["teal"],
    "LateFusion-DeepSurv": COL["purple"],
    "Gated-MultiModal": COL["gold"],
}
PATHWAY_COLORS = {
    "ERK MAPK signaling": COL["blue"],
    "Other, kinases": COL["purple"],
    "p53 pathway": COL["orange"],
    "Chromatin other": COL["magenta"],
    "Protein stability and degradation": COL["teal"],
    "DNA replication": COL["red"],
    "PI3K/MTOR signaling": COL["gold"],
}


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.3,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "legend.title_fontsize": 5.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COL["line"],
            "axes.linewidth": 0.55,
            "grid.color": COL["grid"],
            "grid.linewidth": 0.45,
            "pdf.use14corefonts": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


class NatureV2:
    def __init__(self, project_root: Path, tag: str = "nature_v2") -> None:
        self.project_root = project_root
        self.results = project_root / "results"
        self.out = self.results / "figures" / f"composites_{tag}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.panel_out = self.out / "small_panels_vector"
        self.panel_out.mkdir(parents=True, exist_ok=True)
        self.tag = tag

    def p(self, rel: str) -> Path:
        return self.results / rel

    def csv(self, rel: str) -> pd.DataFrame:
        path = self.p(rel)
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def save(self, fig: plt.Figure, name: str) -> None:
        png = self.out / f"{name}.png"
        pdf = self.out / f"{name}.pdf"
        fig.savefig(png, dpi=650, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    def save_panel(self, fig: plt.Figure, name: str) -> None:
        png = self.panel_out / f"{name}.png"
        pdf = self.panel_out / f"{name}.pdf"
        fig.savefig(png, dpi=650, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    def label(self, ax: plt.Axes, letter: str, x: float = -0.09, y: float = 1.06) -> None:
        ax.text(
            x,
            y,
            letter.lower(),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=COL["ink"],
        )

    def clean(self, ax: plt.Axes, grid: str | bool = "y") -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ax.spines.values():
            sp.set_color(COL["line"])
            sp.set_linewidth(0.55)
        ax.tick_params(length=2.4, width=0.55, colors=COL["muted"])
        if grid == "y":
            ax.grid(True, axis="y", color=COL["grid"], linewidth=0.45)
            ax.grid(False, axis="x")
        elif grid == "x":
            ax.grid(True, axis="x", color=COL["grid"], linewidth=0.45)
            ax.grid(False, axis="y")
        elif grid:
            ax.grid(True, color=COL["grid"], linewidth=0.45)
        else:
            ax.grid(False)

    def fmt2(self, ax: plt.Axes, x: bool = True, y: bool = True) -> None:
        if x:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if y:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    def empty(self, ax: plt.Axes, msg: str) -> None:
        ax.axis("off")
        ax.text(0.5, 0.5, textwrap.fill(msg, 48), ha="center", va="center", color=COL["muted"])

    def add_card(self, ax: plt.Axes, x, y, w, h, title, value, color, subtitle="") -> None:
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc="white",
            ec=color,
            lw=0.9,
        )
        ax.add_patch(box)
        ax.add_patch(patches.Rectangle((x, y + h - 0.035), w, 0.035, fc=color, ec="none", alpha=0.95))
        if value:
            ax.text(x + 0.02, y + h - 0.050, title, ha="left", va="top", color=COL["ink"], fontweight="bold", fontsize=5.7)
            ax.text(x + 0.02, y + h - 0.075, value, ha="left", va="top", color=color, fontweight="bold", fontsize=6.8)
            if subtitle:
                ax.text(x + 0.02, y + h - 0.098, subtitle, ha="left", va="top", color=COL["muted"], fontsize=4.8)
        else:
            ax.text(x + 0.02, y + h - 0.056, title, ha="left", va="top", color=COL["ink"], fontweight="bold", fontsize=6.2)
            if subtitle:
                ax.text(x + 0.02, y + h - 0.084, subtitle, ha="left", va="top", color=COL["muted"], fontsize=5.2)

    def arrow(self, ax: plt.Axes, xy1, xy2, color=COL["line"], rad=0.0, lw=1.0) -> None:
        ax.annotate(
            "",
            xy=xy2,
            xytext=xy1,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                shrinkA=4,
                shrinkB=4,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=9,
            ),
        )

    @staticmethod
    def neglog(p: Iterable[float]) -> np.ndarray:
        arr = pd.to_numeric(pd.Series(p), errors="coerce").replace(0, np.nan).astype(float)
        return -np.log10(arr.clip(lower=1e-300))

    @staticmethod
    def short(s: str, n: int = 28) -> str:
        s = str(s).replace("HALLMARK_", "").replace("_", " ")
        return s if len(s) <= n else s[: n - 1] + "..."

    @staticmethod
    def parse_means(txt: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in str(txt).split(";"):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
        return out

    def lollipop(self, ax, labels, values, color=COL["blue"], xerr=None, xlabel="") -> None:
        labels = list(labels)
        values = np.asarray(values, dtype=float)
        y = np.arange(len(labels))
        ax.hlines(y, 0, values, color=COL["line"], lw=1.2)
        if xerr is not None:
            ax.errorbar(values, y, xerr=xerr, fmt="none", ecolor=COL["muted"], lw=0.7, capsize=1.5, zorder=1)
        ax.scatter(values, y, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        self.clean(ax, grid="x")
        self.fmt2(ax, x=True, y=False)

    @staticmethod
    def modality_label(value: str) -> str:
        return (
            str(value)
            .replace("expression", "Expr")
            .replace("methylation", "Meth")
            .replace("mutation", "Mut")
            .replace("clinical", "Clin")
            .replace("all_modalities", "All")
            .replace("_only", "")
            .replace("_", "+")
            .replace("pca", "PCA")
            .replace("top", "Top")
        )

    def draw_clinical_summary(self, ax: plt.Axes, clin: pd.DataFrame) -> None:
        ax.axis("off")
        ax.set_title("Clinical distribution summary")
        if clin.empty:
            self.empty(ax, "Clinical table not available")
            return

        age_ax = ax.inset_axes([0.00, 0.14, 0.48, 0.78])
        sns.histplot(clin["age"].dropna(), bins=14, color=COL["sky"], edgecolor="white", linewidth=0.5, ax=age_ax)
        age_ax.axvline(clin["age"].median(), color=COL["orange"], lw=1.0)
        age_ax.text(
            0.98,
            0.93,
            f"median {clin['age'].median():.2f}",
            transform=age_ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.2,
            color=COL["orange"],
            fontweight="bold",
        )
        age_ax.set_xlabel("age at diagnosis")
        age_ax.set_ylabel("patients")
        self.clean(age_ax)
        self.fmt2(age_ax, x=True, y=True)

        comp_ax = ax.inset_axes([0.58, 0.12, 0.40, 0.80])
        variables = [("sex", "Sex"), ("stage", "Stage"), ("grade", "Grade")]
        bar_palette = [COL["blue"], COL["orange"], COL["teal"], COL["purple"], COL["gold"], COL["sky"]]
        y_positions = np.arange(len(variables))[::-1]
        for y, (col, label) in zip(y_positions, variables):
            counts = clin[col].fillna("Unknown").astype(str).value_counts()
            total = counts.sum() or 1
            left = 0.0
            for i, (cat, n) in enumerate(counts.items()):
                frac = float(n) / total
                color = bar_palette[i % len(bar_palette)]
                comp_ax.barh(y, frac, left=left, color=color, height=0.45, edgecolor="white", linewidth=0.5)
                if frac >= 0.20:
                    comp_ax.text(left + frac / 2, y, f"{cat.replace('STAGE ', '')}\n{frac * 100:.2f}%", ha="center", va="center", fontsize=4.8, color="white", fontweight="bold")
                left += frac
        comp_ax.set_yticks(y_positions)
        comp_ax.set_yticklabels([v[1] for v in variables])
        comp_ax.set_xlim(0, 1)
        comp_ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x * 100:.2f}"))
        comp_ax.set_xlabel("patients (%)")
        self.clean(comp_ax, grid="x")

    def draw_cohort_pca_ellipse(self, ax: plt.Axes, df: pd.DataFrame, title: str, show_legend: bool = True) -> None:
        if df.empty:
            self.empty(ax, "PCA harmonization table not available")
            return
        cohort_col = "cohort" if "cohort" in df.columns else ("dataset" if "dataset" in df.columns else None)
        if cohort_col is None:
            self.empty(ax, "Cohort labels not available")
            return
        cohorts = sorted(df[cohort_col].dropna().unique())
        palette = dict(zip(cohorts, sns.color_palette("tab10", n_colors=max(len(cohorts), 3)).as_hex()))
        for cohort in cohorts:
            sub = df[df[cohort_col].eq(cohort)]
            color = palette[cohort]
            ax.scatter(sub["PC1"], sub["PC2"], s=8, color=color, alpha=0.35, linewidth=0, label=f"{cohort} (n={len(sub)})")
            if len(sub) >= 4:
                cov = np.cov(sub[["PC1", "PC2"]].dropna().to_numpy().T)
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 1e-9)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                width, height = 2.0 * np.sqrt(vals)
                center = (sub["PC1"].mean(), sub["PC2"].mean())
                ax.add_patch(patches.Ellipse(center, width, height, angle=angle, fc=color, ec=color, lw=0.8, alpha=0.12))
                ax.scatter([center[0]], [center[1]], s=34, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if show_legend:
            ax.legend(frameon=False, loc="upper right", fontsize=4.8, markerscale=1.5, handletextpad=0.3)
        self.clean(ax, grid=True)
        self.fmt2(ax)

    def draw_ablation_forest(self, ax: plt.Axes, abl: pd.DataFrame) -> None:
        if abl.empty:
            self.empty(ax, "Ablation summary not available")
            return
        plot = abl.copy()
        plot["label"] = plot["modality_combo"].map(self.modality_label)
        plot = plot.sort_values("cindex_mean", ascending=True)
        base_vals = plot.loc[plot["modality_combo"].eq("clinical_only"), "cindex_mean"]
        baseline = float(base_vals.iloc[0]) if not base_vals.empty else float(plot["cindex_mean"].min())
        y = np.arange(len(plot))
        colors = np.where(plot["cindex_mean"].to_numpy() >= baseline, COL["teal"], COL["orange"])
        ax.axvline(baseline, color=COL["muted"], lw=0.8, ls="--")
        ax.text(baseline, len(plot) - 0.35, f"clinical {baseline:.2f}", ha="center", va="bottom", fontsize=5.0, color=COL["muted"])
        for yi, (_, r) in zip(y, plot.iterrows()):
            ax.plot([baseline, r["cindex_mean"]], [yi, yi], color=COL["line"], lw=1.2, zorder=1)
        ax.errorbar(
            plot["cindex_mean"],
            y,
            xerr=plot["cindex_std"].fillna(0),
            fmt="none",
            ecolor=COL["muted"],
            elinewidth=0.7,
            capsize=1.8,
            zorder=2,
        )
        ax.scatter(plot["cindex_mean"], y, s=36, c=colors, edgecolor="white", linewidth=0.5, zorder=3)
        xmax = float((plot["cindex_mean"] + plot["cindex_std"].fillna(0)).max() + 0.025)
        xmin = float((plot["cindex_mean"] - plot["cindex_std"].fillna(0)).min() - 0.025)
        for yi, (_, r) in zip(y, plot.iterrows()):
            ax.text(xmax, yi, f"{r['cindex_mean']:.2f}", va="center", ha="right", fontsize=5.0, color=COL["ink"])
        ax.set_xlim(xmin, xmax + 0.015)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"])
        ax.set_xlabel("mean C-index +/- SD")
        ax.set_title("Modality ablation forest")
        self.clean(ax, grid="x")
        self.fmt2(ax, x=True, y=False)

    def draw_permutation_slope(self, ax: plt.Axes, fc: pd.DataFrame) -> None:
        if fc.empty:
            self.empty(ax, "Feature contribution table not available")
            return
        plot = fc.copy()
        plot["label"] = plot["block"].map(self.modality_label)
        plot = plot.sort_values("cindex_drop", ascending=True)
        baseline = float(plot["baseline_cindex"].dropna().iloc[0])
        y = np.arange(len(plot))
        ax.axvline(baseline, color=COL["ink"], lw=0.85)
        ax.text(baseline, len(plot) - 0.28, f"baseline {baseline:.2f}", ha="center", va="bottom", fontsize=5.0, color=COL["ink"])
        norm = plt.Normalize(plot["cindex_drop"].min(), plot["cindex_drop"].max() + 1e-9)
        cmap = plt.get_cmap("magma_r")
        for yi, (_, r) in zip(y, plot.iterrows()):
            color = cmap(norm(float(r["cindex_drop"])))
            ax.plot([r["permuted_cindex_mean"], baseline], [yi, yi], color=color, lw=2.0, alpha=0.90)
            ax.scatter([r["permuted_cindex_mean"]], [yi], s=34, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(baseline + 0.004, yi, f"drop {r['cindex_drop']:.2f}", va="center", fontsize=5.0, color=COL["muted"])
        xmin = float(plot["permuted_cindex_mean"].min() - 0.025)
        xmax = float(baseline + 0.035)
        ax.set_xlim(xmin, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"])
        ax.set_xlabel("C-index after block permutation")
        ax.set_title("Permutation contribution slopegraph")
        self.clean(ax, grid="x")
        self.fmt2(ax, x=True, y=False)

    def draw_hub_rank(self, ax: plt.Axes, hubs: pd.DataFrame) -> None:
        if hubs.empty:
            self.empty(ax, "Hub-gene table not available")
            return
        plot = hubs.head(14).iloc[::-1].copy()
        module_vals = sorted(plot["module_id"].dropna().unique())
        module_palette = dict(zip(module_vals, sns.color_palette("Set2", n_colors=max(len(module_vals), 3)).as_hex()))
        y = np.arange(len(plot))
        ax.hlines(y, 0, plot["degree"], color=COL["grid"], lw=1.4)
        ax.scatter(plot["degree"], y, s=44, color=plot["module_id"].map(module_palette), edgecolor="white", linewidth=0.6, zorder=3)
        for yi, (_, r) in zip(y, plot.iterrows()):
            ax.text(r["degree"] + 1.0, yi, f"{float(r['degree']):.2f}", va="center", fontsize=5.0, color=COL["muted"])
        ax.set_yticks(y)
        ax.set_yticklabels(plot["gene"])
        ax.set_xlabel("co-expression degree")
        ax.set_title("Hub-gene rank map")
        self.clean(ax, grid="x")
        self.fmt2(ax, x=True, y=False)

    def draw_dmp_heatmap(self, ax: plt.Axes, dm: pd.DataFrame) -> None:
        if dm.empty:
            self.empty(ax, "Differential methylation table not available")
            return
        plot = dm.sort_values("fdr").groupby("subtype").head(5).copy()
        ordered_features = plot.sort_values(["subtype", "fdr"])["feature"].drop_duplicates().tolist()
        mat = plot.pivot_table(index="feature", columns="subtype", values="log2fc", aggfunc="first").reindex(ordered_features).fillna(0)
        sns.heatmap(
            mat,
            cmap=sns.diverging_palette(240, 20, as_cmap=True),
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 4.6},
            yticklabels=True,
            linewidths=0.25,
            linecolor="white",
            cbar_kws=dict(label="delta beta", shrink=0.62, pad=0.01),
            ax=ax,
        )
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_title("Differential methylation probe map")
        ax.set_xlabel("subtype")
        ax.set_ylabel("")

    def draw_drug_evidence_matrix(self, ax: plt.Axes, top: pd.DataFrame, overlap: pd.DataFrame) -> None:
        if top.empty:
            self.empty(ax, "Drug prioritization table not available")
            return
        merge_cols = ["drug", "overlap_marker_genes_n"]
        ov = overlap[merge_cols].drop_duplicates("drug") if set(merge_cols).issubset(overlap.columns) else pd.DataFrame(columns=merge_cols)
        plot = top.head(10).merge(ov, on="drug", how="left")
        plot["overlap_marker_genes_n"] = plot["overlap_marker_genes_n"].fillna(0)
        metrics = [
            ("Subtype\n-logFDR", "neglog10_fdr"),
            ("Risk\n-logFDR", "neglog10_risk_fdr"),
            ("Subtype\neffect", "effect_size"),
            ("Risk\neffect", "risk_effect_size"),
            ("Mechanism", "mechanistic_score"),
            ("Marker\noverlap", "overlap_marker_genes_n"),
        ]
        actual = plot[[col for _, col in metrics]].astype(float)
        normed = actual.copy()
        for col in actual.columns:
            vmin, vmax = actual[col].min(), actual[col].max()
            normed[col] = (actual[col] - vmin) / (vmax - vmin + 1e-9)
        annot = actual.applymap(lambda v: f"{v:.2f}")
        normed.index = plot["drug"]
        normed.columns = [label for label, _ in metrics]
        annot.index = normed.index
        annot.columns = normed.columns
        sns.heatmap(
            normed,
            cmap=sns.light_palette(COL["blue"], as_cmap=True),
            vmin=0,
            vmax=1,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 4.8},
            linewidths=0.25,
            linecolor="white",
            cbar_kws=dict(label="column-normalized evidence", shrink=0.60, pad=0.01),
            ax=ax,
        )
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_title("Drug-target evidence matrix")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)

    def raincloud(self, ax, df: pd.DataFrame, x: str, y: str, palette: dict | list | None = None, orient="h") -> None:
        if orient == "h":
            sns.violinplot(data=df, y=y, x=x, hue=y if isinstance(palette, dict) else None, palette=palette, inner=None, cut=0, linewidth=0, alpha=0.45, ax=ax, legend=False)
            sns.boxplot(data=df, y=y, x=x, width=0.18, showfliers=False, color="white", boxprops=dict(edgecolor=COL["ink"], linewidth=0.6), medianprops=dict(color=COL["ink"], linewidth=0.7), whiskerprops=dict(color=COL["ink"], linewidth=0.6), capprops=dict(color=COL["ink"], linewidth=0.6), ax=ax)
            sns.stripplot(data=df, y=y, x=x, hue=y if isinstance(palette, dict) else None, palette=palette, size=1.8, alpha=0.45, jitter=0.22, ax=ax, legend=False)
        else:
            sns.violinplot(data=df, x=x, y=y, hue=x if isinstance(palette, dict) else None, palette=palette, inner=None, cut=0, linewidth=0, alpha=0.45, ax=ax, legend=False)
            sns.boxplot(data=df, x=x, y=y, width=0.18, showfliers=False, color="white", boxprops=dict(edgecolor=COL["ink"], linewidth=0.6), medianprops=dict(color=COL["ink"], linewidth=0.7), ax=ax)
            sns.stripplot(data=df, x=x, y=y, hue=x if isinstance(palette, dict) else None, palette=palette, size=1.7, alpha=0.42, jitter=0.22, ax=ax, legend=False)
        self.clean(ax)

    def draw_alluvial(self, ax, flows: pd.DataFrame, source_col="source", target_col="target", value_col="value") -> None:
        ax.axis("off")
        if flows.empty:
            self.empty(ax, "Alluvial source table not available")
            return
        flows = flows.copy()
        flows[value_col] = pd.to_numeric(flows[value_col], errors="coerce").fillna(0)
        left = flows.groupby(source_col)[value_col].sum().sort_index()
        right = flows.groupby(target_col)[value_col].sum().sort_values(ascending=False)
        total = float(left.sum()) or 1.0

        def blocks(vals):
            y = 1.0
            coords = {}
            for k, v in vals.items():
                h = 0.82 * v / total
                y -= h
                coords[k] = [y, y + h]
                y -= 0.018
            return coords

        left_c = blocks(left)
        right_c = blocks(right)
        right_offsets = {k: v[0] for k, v in right_c.items()}
        left_offsets = {k: v[0] for k, v in left_c.items()}
        for i, (k, (y0, y1)) in enumerate(left_c.items()):
            c = SUBTYPE.get(int(str(k).split("_")[-1]) if str(k).split("_")[-1].isdigit() else i + 1, COL["blue"])
            ax.add_patch(patches.Rectangle((0.02, y0), 0.08, y1 - y0, fc=c, ec="none", alpha=0.95))
            ax.text(0.0, (y0 + y1) / 2, str(k).replace("Discovered_", "S"), ha="right", va="center", fontsize=5.5)
        for i, (k, (y0, y1)) in enumerate(right_c.items()):
            ax.add_patch(patches.Rectangle((0.90, y0), 0.08, y1 - y0, fc=sns.color_palette("Set2", len(right_c))[i], ec="none", alpha=0.95))
            ax.text(1.0, (y0 + y1) / 2, str(k).replace("Moffitt_", ""), ha="left", va="center", fontsize=5.5)

        for _, r in flows.iterrows():
            s, t, v = r[source_col], r[target_col], float(r[value_col])
            if v <= 0:
                continue
            h = 0.82 * v / total
            sy0 = left_offsets[s]
            ty0 = right_offsets[t]
            left_offsets[s] += h
            right_offsets[t] += h
            sy1, ty1 = sy0 + h, ty0 + h
            c = SUBTYPE.get(int(str(s).split("_")[-1]) if str(s).split("_")[-1].isdigit() else 1, COL["blue"])
            verts = [
                (0.10, sy0), (0.35, sy0), (0.65, ty0), (0.90, ty0),
                (0.90, ty1), (0.65, ty1), (0.35, sy1), (0.10, sy1),
                (0.10, sy0),
            ]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts, codes), fc=c, ec="none", alpha=0.28))
        ax.set_xlim(-0.06, 1.14)
        ax.set_ylim(0.05, 1.03)

    def figure1(self) -> None:
        facts = json.loads((self.results / "manuscript" / "results_key_facts.json").read_text(encoding="utf-8"))
        fig = plt.figure(figsize=(7.2, 4.35))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.02, 0.94, "a", fontsize=8, fontweight="bold", color=COL["ink"])
        ax.text(0.08, 0.94, "ExprBench-PDAC framework for transferable PDAC stratification", fontsize=8.5, fontweight="bold", color=COL["ink"])

        modalities = [
            ("RNA-seq", "20,512", COL["blue"]),
            ("DNA methylation", "22,601", COL["teal"]),
            ("Mutation", "10,014", COL["orange"]),
            ("Clinical", "167 pts", COL["purple"]),
        ]
        for i, (t, v, c) in enumerate(modalities):
            self.add_card(ax, 0.05, 0.70 - i * 0.15, 0.20, 0.12, t, v, c, "harmonized features")
            self.arrow(ax, (0.25, 0.755 - i * 0.15), (0.39, 0.51), c, rad=0.12 - 0.08 * i, lw=1.2)

        hub = patches.Circle((0.50, 0.50), 0.13, fc="#F4F7FB", ec=COL["ink"], lw=0.85)
        ax.add_patch(hub)
        ax.text(0.50, 0.535, "Model\nbenchmark", ha="center", va="center", fontsize=9, fontweight="bold", color=COL["ink"])
        ax.text(0.50, 0.493, "regularized expression\nvs deep/multimodal comparators", ha="center", va="center", fontsize=5.7, color=COL["muted"])
        ax.text(0.50, 0.445, f"C-index {facts.get('best_model', {}).get('cindex_mean', 0.615):.3f}", ha="center", va="center", fontsize=6.5, color=COL["blue"], fontweight="bold")

        outputs = [
            ("3-subtype map", "log-rank P=5.77e-4", COL["blue"], 0.76, 0.75),
            ("Immune contexture", "cold-to-inflamed states", COL["teal"], 0.78, 0.58),
            ("Drug-response transfer", "80 models; top PLX-4720", COL["orange"], 0.78, 0.40),
            ("Single-cell anchor", "9,948 cells", COL["purple"], 0.76, 0.22),
        ]
        for title, sub, color, x, y in outputs:
            self.arrow(ax, (0.61, 0.50), (x, y + 0.04), color, rad=0.12 - (y - 0.5) * 0.12, lw=1.2)
            self.add_card(ax, x, y, 0.19, 0.12, title, "", color, sub)
        ax.text(0.05, 0.08, "Locked risk score transferred to GSE62452, GSE28735 and GSE71729; figures use public de-identified cohorts.", fontsize=5.8, color=COL["muted"])
        self.save(fig, "Figure1")

    def figure2(self) -> None:
        flow = self.csv("01_tcga_prep/figure_ready/Figure2A_flow_counts.csv")
        upset = self.csv("01_tcga_prep/figure_ready/Figure2B_upset_table.csv")
        clin = self.csv("01_tcga_prep/figure_ready/Figure2C_clinical_summary.csv")
        muts = self.csv("01_tcga_prep/figure_ready/Figure2I_top_mutation_frequencies.csv")
        pca_b = self.csv("07_external_validation/harmonization_visuals/pca_before.csv")
        pca_a = self.csv("07_external_validation/harmonization_visuals/pca_after.csv")
        fig = plt.figure(figsize=(7.2, 6.2))
        gs = GridSpec(3, 4, figure=fig, hspace=0.62, wspace=0.65)
        ax = fig.add_subplot(gs[0, 0:2])
        self.label(ax, "a")
        flow = flow.copy()
        flow["label"] = flow["step"].str.replace("_", "\n")
        x = np.arange(len(flow))
        ax.plot(x, flow["n_patients"], color=COL["blue"], lw=1.2)
        ax.scatter(x, flow["n_patients"], s=85, color=[COL["blue"], COL["sky"], COL["teal"], COL["orange"], COL["purple"]], edgecolor="white", linewidth=0.8, zorder=3)
        for xi, n in zip(x, flow["n_patients"]):
            ax.text(xi, n + 5, f"{int(n)}", ha="center", fontsize=5.8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(flow["label"], rotation=0)
        ax.set_ylabel("patients")
        ax.set_title("Cohort filtering as a compact cascade")
        self.clean(ax)

        ax = fig.add_subplot(gs[0, 2:4])
        self.label(ax, "b")
        if not upset.empty:
            bool_cols = ["has_expression", "has_methylation", "has_mutation", "has_clinical"]
            combos = upset.groupby(bool_cols).size().reset_index(name="n").sort_values("n", ascending=False).head(8)
            y0 = np.arange(len(bool_cols))
            for j, row in combos.reset_index(drop=True).iterrows():
                for i, col in enumerate(bool_cols):
                    ax.scatter(j, i, s=34, color=COL["ink"] if row[col] else "#D7DDE6", zorder=3)
                active = [i for i, col in enumerate(bool_cols) if row[col]]
                if len(active) > 1:
                    ax.plot([j, j], [min(active), max(active)], color=COL["ink"], lw=0.75, zorder=2)
                ax.bar(j, row["n"] / combos["n"].max() * 0.55, bottom=4.0, color=COL["blue"], width=0.58)
                ax.text(j, 4.6, str(int(row["n"])), ha="center", fontsize=5.2)
            ax.set_yticks(y0)
            ax.set_yticklabels([c.replace("has_", "") for c in bool_cols])
            ax.set_ylim(-0.6, 4.9)
            ax.set_xticks([])
            ax.set_title("Multi-omics intersection dot matrix")
            self.clean(ax, grid=False)

        ax = fig.add_subplot(gs[1, 0:2])
        self.label(ax, "c")
        self.draw_clinical_summary(ax, clin)

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "d")
        muts = muts.head(16).iloc[::-1]
        mut_count_col = "mutated_patients" if "mutated_patients" in muts.columns else "count"
        self.lollipop(ax, muts["gene"], muts[mut_count_col], COL["orange"], xlabel="mutated patients")
        ax.set_title("Driver mutation rank as lollipop")

        sub = gs[2, :].subgridspec(1, 2, wspace=0.25)
        for i, (df, title, letter) in enumerate([(pca_b, "Before harmonization", "e"), (pca_a, "After harmonization", "f")]):
            ax = fig.add_subplot(sub[0, i])
            self.label(ax, letter)
            self.draw_cohort_pca_ellipse(ax, df, title, show_legend=i == 1)
        self.save(fig, "Figure2")

    def figure3(self) -> None:
        fold = self.csv("02_benchmark/benchmark_fold_level_metrics.csv")
        auc = self.csv("02_benchmark/time_auc_summary.csv")
        cal = self.csv("02_benchmark/calibration_summary.csv")
        dca = self.csv("02_benchmark/dca_summary.csv")
        abl = self.csv("02_benchmark/ablation_summary.csv")
        fc = self.csv("02_benchmark/feature_contribution_summary.csv")
        fig = plt.figure(figsize=(7.2, 6.25))
        gs = GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.72)
        ax = fig.add_subplot(gs[0, 0:2])
        self.label(ax, "a")
        order = fold.groupby("model")["cindex"].median().sort_values(ascending=False).index.tolist()
        fold["model"] = pd.Categorical(fold["model"], categories=order, ordered=True)
        self.raincloud(ax, fold.sort_values("model"), "cindex", "model", MODEL_COLORS, orient="h")
        ax.set_title("Repeated-CV discrimination raincloud")
        ax.set_xlabel("C-index")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[0, 2:4])
        self.label(ax, "b")
        for model, sub in auc.groupby("model"):
            c = MODEL_COLORS.get(model, COL["muted"])
            sub = sub.sort_values("time")
            ax.plot(sub["time"], sub["auc_mean"], marker="o", ms=3, lw=1.0, color=c, label=model)
            ax.fill_between(sub["time"], sub["auc_mean"] - sub["auc_std"], sub["auc_mean"] + sub["auc_std"], color=c, alpha=0.10, linewidth=0)
        ax.set_title("Time-dependent AUC trajectories")
        ax.set_xlabel("months")
        ax.set_ylabel("AUC")
        ax.legend(frameon=False, ncol=2, loc="lower right")
        self.clean(ax)

        ax = fig.add_subplot(gs[1, 0:2])
        self.label(ax, "c")
        if not cal.empty:
            for h, sub in cal.groupby("horizon_month"):
                ax.plot(sub["mean_pred"], sub["observed_rate"], marker="o", ms=2.8, lw=0.9, label=f"{int(h)} mo")
        ax.plot([0, 1], [0, 1], ls="--", color=COL["line"], lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Calibration by horizon")
        ax.set_xlabel("mean predicted risk")
        ax.set_ylabel("observed event rate")
        ax.legend(frameon=False)
        self.clean(ax, grid=True)

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "d")
        for model, sub in dca.groupby("model"):
            ax.plot(sub["threshold"], sub["net_benefit"], lw=1.0, color=MODEL_COLORS.get(model, COL["muted"]), label=model)
        ax.axhline(0, color=COL["line"], lw=0.6)
        ax.set_title("Decision-curve net benefit")
        ax.set_xlabel("risk threshold")
        ax.set_ylabel("net benefit")
        self.clean(ax)

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "e")
        self.draw_ablation_forest(ax, abl)

        ax = fig.add_subplot(gs[2, 2:4])
        self.label(ax, "f")
        self.draw_permutation_slope(ax, fc)
        self.save(fig, "Figure3")

    def figure4(self) -> None:
        umap = self.csv("03_subtypes/latent_umap_coordinates.csv")
        metrics = self.csv("03_subtypes/subtype_model_selection_metrics.csv")
        clinical = self.csv("01_tcga_prep/clinical_matrix.csv")
        assign = self.csv("03_subtypes/subtype_assignments_final.csv")
        sankey = self.csv("04_classical_mapping/sankey_table.csv")
        clinic_assoc = self.csv("03_subtypes/subtype_clinicopathologic_summary.csv")
        fig = plt.figure(figsize=(7.2, 6.25))
        gs = GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.72)

        ax = fig.add_subplot(gs[0:2, 0:2])
        self.label(ax, "a")
        sns.scatterplot(data=umap, x="UMAP1", y="UMAP2", hue="subtype", style="risk_group", palette=SUBTYPE, s=22, linewidth=0.25, edgecolor="white", alpha=0.9, ax=ax)
        ax.set_title("Latent molecular map")
        ax.legend(frameon=False, ncol=2, loc="best")
        self.clean(ax, grid=True)

        ax = fig.add_subplot(gs[0, 2], projection="polar")
        self.label(ax, "b")
        norm = metrics.copy()
        for col in ["silhouette", "gap", "stability_mean_ari"]:
            norm[col] = (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min() + 1e-9)
        norm["survival"] = self.neglog(norm["logrank_p"])
        norm["survival"] = (norm["survival"] - norm["survival"].min()) / (norm["survival"].max() - norm["survival"].min() + 1e-9)
        radar_cols = ["silhouette", "gap", "stability_mean_ari", "survival"]
        theta = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False)
        for _, r in norm.iterrows():
            vals = r[radar_cols].to_numpy(float)
            vals = np.r_[vals, vals[0]]
            th = np.r_[theta, theta[0]]
            ax.plot(th, vals, marker="o", ms=2.2, lw=0.8, color=SUBTYPE.get(int(r["k"]) - 1, COL["muted"]), label=f"k={int(r['k'])}")
            ax.fill(th, vals, color=SUBTYPE.get(int(r["k"]) - 1, COL["muted"]), alpha=0.06)
        ax.set_xticks(theta)
        ax.set_xticklabels(["sil", "gap", "ARI", "surv"])
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.grid(color="#E6EAF0", lw=0.45)
        ax.set_title("Cluster evidence radar")
        ax.legend(frameon=False, loc="upper right")
        ax.spines["polar"].set_color("#D7DDE6")

        ax = fig.add_subplot(gs[0, 3])
        self.label(ax, "c")
        km = assign.merge(clinical[["PATIENT_ID", "OS_TIME", "OS_EVENT"]], on="PATIENT_ID", how="left").dropna(subset=["OS_TIME", "OS_EVENT"])
        fitter = KaplanMeierFitter()
        for subtype, sub in km.groupby("subtype"):
            fitter.fit(sub["OS_TIME"], event_observed=sub["OS_EVENT"], label=f"S{subtype}")
            fitter.plot(ax=ax, ci_show=False, lw=1.0, color=SUBTYPE.get(int(subtype), COL["muted"]))
        ax.set_title("Survival by subtype")
        ax.set_xlabel("months")
        ax.set_ylabel("survival")
        ax.legend(frameon=False)
        self.clean(ax)

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "d")
        self.draw_alluvial(ax, sankey[sankey["system"].eq("Moffitt")], "source", "target", "value")
        ax.set_title("Alluvial mapping to Moffitt classes")

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "e")
        plot = assign.copy()
        plot["subtype_label"] = "S" + plot["subtype"].astype(str)
        self.raincloud(ax, plot, "risk_score", "subtype_label", {f"S{k}": v for k, v in SUBTYPE.items()}, orient="h")
        ax.set_title("Risk-score distribution by subtype")
        ax.set_xlabel("locked risk score")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[2, 2:4])
        self.label(ax, "f")
        clinic_assoc = clinic_assoc.copy()
        clinic_assoc["neglog_fdr"] = self.neglog(clinic_assoc["fdr"])
        sns.scatterplot(data=clinic_assoc, x="effect_size", y="feature", size="neglog_fdr", hue="type", palette=[COL["blue"], COL["orange"]], sizes=(35, 160), ax=ax)
        ax.set_title("Clinicopathologic association map")
        ax.set_xlabel("effect size")
        ax.set_ylabel("")
        ax.legend(frameon=False, loc="lower right")
        self.clean(ax, grid="x")
        self.save(fig, "Figure4")

    def figure5(self) -> None:
        deg = self.csv("05_biology/deg_results.csv")
        markers = self.csv("05_biology/marker_gene_table.csv")
        gsva = self.csv("05_biology/gsva_summary.csv")
        mut = self.csv("05_biology/mutation_subtype_comparison.csv")
        edges = self.csv("05_biology/network_edges.csv")
        hubs = self.csv("05_biology/hub_genes.csv")
        coupling = self.csv("05_biology/methylation_expression_coupling.csv")
        fig = plt.figure(figsize=(7.2, 6.45))
        gs = GridSpec(3, 4, figure=fig, hspace=0.76, wspace=0.78)

        ax = fig.add_subplot(gs[0, 0:2])
        self.label(ax, "a")
        plot = deg.copy()
        plot["neglog"] = self.neglog(plot["fdr"])
        plot = plot.sample(min(len(plot), 9000), random_state=3)
        sig = plot["fdr"] < 0.01
        ax.scatter(plot.loc[~sig, "log2fc"], plot.loc[~sig, "neglog"], s=2, color="#C9CED6", alpha=0.40, linewidth=0)
        ax.scatter(plot.loc[sig, "log2fc"], plot.loc[sig, "neglog"], s=4, c=plot.loc[sig, "subtype"].map(SUBTYPE), alpha=0.65, linewidth=0)
        top = deg.assign(neglog=self.neglog(deg["fdr"])).sort_values("fdr").head(8)
        ax.axvline(0, color=COL["line"], lw=0.6)
        ax.set_ylim(0, max(60, float(top["neglog"].max()) * 1.06))
        # Stack the most significant volcano labels in the unused upper-left
        # space so the gene names remain readable after downscaling into DOCX.
        y_stack = np.linspace(ax.get_ylim()[1] - 4.0, ax.get_ylim()[1] - 22.0, len(top))
        x_text = ax.get_xlim()[0] + 0.35
        for y_text, (_, r) in zip(y_stack, top.iterrows()):
            ax.annotate(
                str(r["feature"]),
                xy=(r["log2fc"], min(r["neglog"], ax.get_ylim()[1] - 1.5)),
                xytext=(x_text, y_text),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=4.7,
                color=COL["ink"],
                arrowprops=dict(arrowstyle="-", color=COL["line"], lw=0.35, shrinkA=1, shrinkB=1),
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.82),
            )
        ax.set_title("Differential-expression volcano cloud")
        ax.set_xlabel("log2 fold-change")
        ax.set_ylabel("-log10(FDR)")
        self.clean(ax)

        ax = fig.add_subplot(gs[0, 2:4])
        self.label(ax, "b")
        top_markers = markers.sort_values("fdr").groupby("subtype").head(6)
        mat = top_markers.pivot_table(index="feature", columns="subtype", values="log2fc", aggfunc="max").fillna(0)
        sns.heatmap(mat, cmap=sns.diverging_palette(240, 20, as_cmap=True), center=0, ax=ax, cbar_kws=dict(label="log2FC", shrink=0.55), linewidths=0.25, linecolor="white")
        ax.set_title("Subtype marker heatmap")
        ax.set_xlabel("subtype")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[1, 0:2])
        self.label(ax, "c")
        path = gsva.sort_values("fdr").groupby("subtype").head(6).copy()
        path["pathway"] = path["gene_set"].map(lambda x: self.short(x, 34))
        path["neglog"] = self.neglog(path["fdr"])
        sns.scatterplot(data=path, x="delta", y="pathway", hue="subtype", size="neglog", palette=SUBTYPE, sizes=(15, 95), ax=ax)
        ax.axvline(0, color=COL["line"], lw=0.6)
        ax.set_xlim(float(path["delta"].min()) - 0.25, float(path["delta"].max()) + 1.35)
        ax.set_title("Pathway effect-size dot plot")
        ax.set_xlabel("inside vs outside subtype score")
        ax.set_ylabel("")
        ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.52), borderaxespad=0.0)
        self.clean(ax, grid="x")

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "d")
        top_genes = mut.sort_values("fdr").groupby("subtype").head(8)
        mat = top_genes.pivot_table(index="gene", columns="subtype", values="mut_freq_in", aggfunc="mean").fillna(0)
        sns.heatmap(mat, cmap=sns.light_palette(COL["orange"], as_cmap=True), ax=ax, cbar_kws=dict(label="mutation frequency", shrink=0.55), linewidths=0.25, linecolor="white")
        ax.set_title("Subtype mutation-frequency matrix")
        ax.set_xlabel("subtype")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "e")
        self.draw_hub_rank(ax, hubs)

        ax = fig.add_subplot(gs[2, 2:4])
        self.label(ax, "f")
        dm = self.csv("05_biology/dm_all_subtypes.csv")
        self.draw_dmp_heatmap(ax, dm)
        self.save(fig, "Figure5")

    def figure6(self) -> None:
        means = self.csv("06_immune/immune_feature_means_by_subtype.csv")
        stats = self.csv("06_immune/immune_feature_statistics.csv")
        cold = self.csv("06_immune/cold_hot_summary.csv")
        matrix = self.csv("06_immune/immune_feature_matrix.csv")
        assign = self.csv("03_subtypes/subtype_assignments_final.csv")
        fig = plt.figure(figsize=(7.2, 6.1))
        gs = GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.74)

        ax = fig.add_subplot(gs[0, 0:2])
        self.label(ax, "a")
        feats = ["T_CELL_INFLAMED", "CYTOTOXIC", "IFNG_SIGNATURE", "STROMAL_SIGNATURE", "HYPOXIA_SIGNATURE", "TIDE_DYSFUNCTION_PROXY", "TIDE_EXCLUSION_PROXY", "RISK_SCORE"]
        mat = means.set_index("subtype")[feats].T
        sns.heatmap(mat, cmap=sns.diverging_palette(240, 20, as_cmap=True), center=0, ax=ax, cbar_kws=dict(label="mean z-score", shrink=0.55), linewidths=0.25, linecolor="white")
        ax.set_title("Subtype immune-state heatmap")
        ax.set_xlabel("subtype")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[0, 2:4])
        self.label(ax, "b")
        state = cold[cold["grouping"].eq("subtype")].copy()
        piv = state.pivot(index="group", columns="immune_state", values="count").fillna(0)
        piv = piv[[c for c in ["Cold", "Intermediate", "Hot"] if c in piv.columns]]
        bottom = np.zeros(len(piv))
        x = np.arange(len(piv))
        colors = {"Cold": COL["sky"], "Intermediate": COL["gold"], "Hot": COL["orange"]}
        for col in piv.columns:
            ax.bar(x, piv[col], bottom=bottom, color=colors.get(col, COL["muted"]), label=col, width=0.62)
            bottom += piv[col].to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in piv.index])
        ax.set_ylabel("patients")
        ax.set_title("Cold-to-hot composition")
        ax.legend(frameon=False, ncol=3)
        self.clean(ax)

        for j, (pattern, title, color) in enumerate([("CHECKPOINT__", "Immune checkpoint lollipop", COL["teal"]), ("HLA__", "Antigen-presentation lollipop", COL["blue"])]):
            ax = fig.add_subplot(gs[1, j * 2:(j + 1) * 2])
            self.label(ax, chr(ord("c") + j))
            sub = stats[stats["feature"].str.contains(pattern, regex=False)].sort_values("fdr").head(10)
            self.lollipop(ax, sub["feature"].str.replace(pattern, "", regex=False), self.neglog(sub["fdr"]), color, xlabel="-log10(FDR)")
            ax.set_title(title)

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "e")
        plot = stats[stats["grouping"].eq("risk_group")].copy()
        plot["neglog"] = self.neglog(plot["fdr"])
        sns.scatterplot(data=plot, x="effect_size", y="neglog", hue="feature", palette="tab20", s=36, legend=False, ax=ax)
        top = plot.sort_values("fdr").head(8)
        ax.set_xlim(float(plot["effect_size"].min()) - 0.01, max(0.55, float(top["effect_size"].max()) + 0.09))
        y_stack = np.linspace(float(top["neglog"].max()) - 0.08, max(2.95, float(top["neglog"].max()) - 2.1), len(top))
        x_text = ax.get_xlim()[1] - 0.115
        for y_text, (_, r) in zip(y_stack, top.iterrows()):
            label = str(r["feature"]).replace("CHECKPOINT__", "").replace("_PROXY", "")[:17]
            ax.annotate(
                label,
                xy=(r["effect_size"], r["neglog"]),
                xytext=(x_text, y_text),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=4.6,
                color=COL["ink"],
                arrowprops=dict(arrowstyle="-", color=COL["line"], lw=0.35, shrinkA=1, shrinkB=1),
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.82),
            )
        ax.set_title("Risk-associated immune feature volcano")
        ax.set_xlabel("effect size")
        ax.set_ylabel("-log10(FDR)")
        self.clean(ax)

        ax = fig.add_subplot(gs[2, 2:4])
        self.label(ax, "f")
        merged = matrix.merge(assign[["PATIENT_ID", "subtype"]], on="PATIENT_ID", how="left")
        long = merged.melt(id_vars=["PATIENT_ID", "subtype"], value_vars=["T_CELL_INFLAMED", "CYTOTOXIC", "IFNG_SIGNATURE", "STROMAL_SIGNATURE"], var_name="signature", value_name="score")
        sns.violinplot(data=long, x="signature", y="score", hue="subtype", palette=SUBTYPE, cut=0, linewidth=0.4, inner=None, ax=ax)
        ax.set_xticklabels([t.get_text().replace("_", "\n") for t in ax.get_xticklabels()])
        ax.set_title("Patient-level immune signature distributions")
        ax.set_xlabel("")
        ax.legend(frameon=False, ncol=3, loc="upper right")
        self.clean(ax)
        self.save(fig, "Figure6")

    def figure7(self) -> None:
        ext = self.csv("07_external_validation/external_summary.csv")
        forest = self.csv("07_external_validation/forest_plot_table.csv")
        meta = self.csv("07_external_validation/meta_summary.csv")
        fig = plt.figure(figsize=(7.2, 5.9))
        gs = GridSpec(3, 4, figure=fig, hspace=0.74, wspace=0.72)

        ax = fig.add_subplot(gs[0, 0:2])
        self.label(ax, "a")
        x = np.arange(len(ext))
        ax.scatter(x, ext["n_expression"], s=90, color=COL["blue"], label="expression")
        ax.scatter(x, ext["n_survival"], s=90, color=COL["orange"], label="survival")
        for xi, row in ext.iterrows():
            ax.plot([xi, xi], [row["n_survival"], row["n_expression"]], color=COL["line"], lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(ext["cohort"], rotation=25, ha="right")
        ax.set_ylabel("samples")
        ax.set_title("External cohort transfer map")
        ax.legend(frameon=False)
        self.clean(ax)

        for j, cohort in enumerate(["GSE62452", "GSE28735"]):
            ax = fig.add_subplot(gs[0, 2 + j])
            self.label(ax, chr(ord("b") + j))
            clin = self.csv(f"07_external_validation/{cohort}_clinical_processed.csv")
            pred = self.csv(f"07_external_validation/student_model_predictions_{cohort}.csv")
            km = clin.merge(pred, on="PATIENT_ID").dropna(subset=["OS_TIME", "OS_EVENT"])
            fitter = KaplanMeierFitter()
            for group, sub in km.groupby("risk_group_locked"):
                fitter.fit(sub["OS_TIME"], sub["OS_EVENT"], label=group)
                fitter.plot(ax=ax, ci_show=False, lw=1.0, color=RISK.get(group, COL["muted"]))
            ax.set_title(cohort)
            ax.set_xlabel("months")
            ax.set_ylabel("survival")
            ax.legend(frameon=False)
            self.clean(ax)

        ax = fig.add_subplot(gs[1, 0:2])
        self.label(ax, "d")
        fp = forest.copy()
        fp["log_hr"] = np.log(fp["hr"])
        fp["log_low"] = np.log(fp["hr_ci_low"])
        fp["log_high"] = np.log(fp["hr_ci_high"])
        y = np.arange(len(fp))
        ax.errorbar(fp["log_hr"], y, xerr=[fp["log_hr"] - fp["log_low"], fp["log_high"] - fp["log_hr"]], fmt="o", color=COL["blue"], ecolor=COL["ink"], lw=0.8, capsize=2)
        if not meta.empty:
            ax.scatter(np.log(meta.loc[0, "pooled_hr"]), len(fp), marker="D", s=42, color=COL["orange"], zorder=3)
            ax.set_yticks(list(y) + [len(fp)])
            ax.set_yticklabels(fp["cohort"].tolist() + ["pooled"])
        else:
            ax.set_yticks(y)
            ax.set_yticklabels(fp["cohort"])
        ax.axvline(0, ls="--", color=COL["line"], lw=0.8)
        ax.set_xlabel("log hazard ratio")
        ax.set_title("Validation hazard-ratio forest")
        self.clean(ax, grid="x")

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "e")
        for cohort in ["GSE62452", "GSE28735"]:
            cal = self.csv(f"07_external_validation/calibration_external_{cohort}.csv")
            if not cal.empty:
                pred_col = "mean_predicted_score" if "mean_predicted_score" in cal.columns else "predicted_score_mean"
                ax.scatter(cal[pred_col], cal["observed_event_rate"], s=28, label=cohort, alpha=0.85)
        ax.plot([0, 1], [0, 1], ls="--", color=COL["line"], lw=0.8)
        ax.set_xlabel("mean predicted score")
        ax.set_ylabel("observed event rate")
        ax.set_title("External calibration cloud")
        ax.legend(frameon=False)
        self.clean(ax, grid=True)

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "f")
        rows = []
        for cohort in ext["cohort"].tolist():
            pred = self.csv(f"07_external_validation/student_model_predictions_{cohort}.csv")
            if not pred.empty:
                pred["cohort"] = cohort
                rows.append(pred)
        dist = pd.concat(rows, ignore_index=True)
        sns.violinplot(data=dist, x="cohort", y="student_risk_score", color=COL["sky"], inner=None, cut=0, linewidth=0, ax=ax)
        sns.stripplot(data=dist.sample(min(len(dist), 450), random_state=4), x="cohort", y="student_risk_score", hue="risk_group_locked", palette=RISK, size=1.8, alpha=0.55, ax=ax, legend=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_title("Transferred risk-score distributions")
        ax.set_xlabel("")
        self.clean(ax)

        ax = fig.add_subplot(gs[2, 2:4])
        self.label(ax, "g")
        subtype_rows = []
        for cohort in ext["cohort"].tolist():
            st = self.csv(f"07_external_validation/subtype_transfer_{cohort}.csv")
            if not st.empty:
                st["cohort"] = cohort
                subtype_rows.append(st)
        if subtype_rows:
            st = pd.concat(subtype_rows)
            col = "predicted_subtype" if "predicted_subtype" in st.columns else ("subtype" if "subtype" in st.columns else st.columns[-1])
            piv = pd.crosstab(st["cohort"], st[col])
            piv.plot(kind="bar", stacked=True, color=[SUBTYPE.get(int(c), COL["muted"]) if str(c).isdigit() else COL["muted"] for c in piv.columns], ax=ax, width=0.66)
            ax.legend(frameon=False, title="subtype", ncol=3)
        ax.set_title("Subtype transfer composition")
        ax.set_xlabel("")
        ax.set_ylabel("samples")
        self.clean(ax)
        self.save(fig, "Figure7")

    def figure8(self) -> None:
        top = self.csv("08_drug/top_compounds_final.csv")
        risk = self.csv("08_drug/riskgroup_drug_comparison.csv")
        pathsum = self.csv("08_drug/drug_target_pathway_summary.csv")
        overlap = self.csv("08_drug/drug_marker_overlap.csv")
        fig = plt.figure(figsize=(7.2, 7.15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.92, wspace=0.92)

        ax = fig.add_subplot(gs[0, 0:2])
        self.label(ax, "a")
        plot = top.head(30).copy()
        pathways = plot["target_pathway"].fillna("Other").astype(str).unique().tolist()
        pathway_palette = {
            name: PATHWAY_COLORS.get(name, color)
            for name, color in zip(pathways, sns.color_palette("tab20", n_colors=max(len(pathways), 3)).as_hex())
        }
        sns.scatterplot(data=plot, x="neglog10_fdr", y="neglog10_risk_fdr", size="priority_score", hue="target_pathway", palette=pathway_palette, sizes=(22, 145), alpha=0.86, edgecolor="white", linewidth=0.4, legend=False, ax=ax)
        for _, r in plot.head(6).iterrows():
            ax.text(r["neglog10_fdr"] + 0.05, r["neglog10_risk_fdr"] + 0.05, self.short(r["drug"], 12), fontsize=5.1)
        ax.set_title("Drug-transfer evidence space")
        ax.set_xlabel("subtype association, -log10(FDR)")
        ax.set_ylabel("risk association, -log10(FDR)")
        ax.text(0.02, 0.96, "dot size: priority score; color: target pathway", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=COL["muted"])
        self.clean(ax)

        ax = fig.add_subplot(gs[0, 2:4])
        self.label(ax, "b")
        rank = top.head(12).iloc[::-1]
        self.lollipop(ax, rank["drug"], rank["priority_score"], COL["blue"], xlabel="priority score")
        ax.set_title("Top compound rank lollipop")

        ax = fig.add_subplot(gs[1, 0:2])
        self.label(ax, "c")
        heat_rows = []
        for _, r in top.head(12).iterrows():
            vals = self.parse_means(r["group_means"])
            vals["drug"] = r["drug"]
            heat_rows.append(vals)
        hm = pd.DataFrame(heat_rows).set_index("drug")[[c for c in ["1", "2", "3"] if c in pd.DataFrame(heat_rows).columns]]
        sns.heatmap(hm, cmap=sns.diverging_palette(240, 20, as_cmap=True), center=hm.stack().median(), linewidths=0.25, linecolor="white", cbar_kws=dict(label="lnIC50", shrink=0.52, pad=0.01), ax=ax)
        ax.set_title("Subtype-specific predicted response")
        ax.set_xlabel("subtype")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "d")
        pathway = top.groupby("target_pathway").agg(priority=("priority_score", "max"), n=("drug", "count"), mechanistic=("mechanistic_score", "mean")).reset_index().sort_values("priority", ascending=False).head(12)
        pathway = pathway.sort_values("mechanistic")
        y = np.arange(len(pathway))
        sizes = 28 + pathway["priority"].to_numpy() / max(pathway["priority"].max(), 1) * 145
        colors = plt.get_cmap("viridis")((pathway["n"].to_numpy() - pathway["n"].min()) / (pathway["n"].max() - pathway["n"].min() + 1e-9))
        ax.scatter(pathway["mechanistic"], y, s=sizes, c=colors, edgecolor="white", linewidth=0.45, alpha=0.9)
        for yi, (_, r) in enumerate(pathway.iterrows()):
            ax.plot([0, r["mechanistic"]], [yi, yi], color=COL["grid"], lw=0.6, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([self.short(v, 24) for v in pathway["target_pathway"]])
        ax.set_title("Mechanistic pathway bubble map")
        ax.set_xlabel("mean mechanistic score")
        ax.set_ylabel("")
        ax.text(0.98, 0.04, "size: max priority\ncolor: compound count", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.1, color=COL["muted"])
        self.clean(ax, grid="x")

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "e")
        r = risk.sort_values("fdr").head(14).iloc[::-1]
        self.lollipop(ax, r["drug"], self.neglog(r["fdr"]), COL["orange"], xlabel="-log10(FDR)")
        ax.set_title("Risk-group response separation")

        ax = fig.add_subplot(gs[2, 2:4])
        self.label(ax, "f")
        self.draw_drug_evidence_matrix(ax, top, overlap)
        self.save(fig, "Figure8")

    def figure9(self) -> None:
        sc = self.csv("09_singlecell/sc_cell_annotations.csv")
        key = self.csv("09_singlecell/key_gene_expression_by_celltype.csv")
        fig = plt.figure(figsize=(7.2, 6.4))
        gs = GridSpec(3, 4, figure=fig, hspace=0.76, wspace=0.78)

        ax = fig.add_subplot(gs[0:2, 0:2])
        self.label(ax, "a")
        cols = ["bulk_Subtype_1", "bulk_Subtype_2", "bulk_Subtype_3", "nCount_RNA", "nFeature_RNA", "percent.mt"]
        sub = sc.sample(min(len(sc), 5200), random_state=5).copy()
        X = sub[cols].fillna(0).to_numpy()
        coords = PCA(n_components=2, random_state=5).fit_transform(StandardScaler().fit_transform(X))
        sub["PC1"], sub["PC2"] = coords[:, 0], coords[:, 1]
        top_types = sub["cell_type_auto"].value_counts().head(10).index
        sub["cell_type_plot"] = np.where(sub["cell_type_auto"].isin(top_types), sub["cell_type_auto"], "Other")
        sns.scatterplot(data=sub, x="PC1", y="PC2", hue="cell_type_plot", s=5, alpha=0.72, linewidth=0, palette="tab20", legend=False, ax=ax)
        ax.set_title("Single-cell signature-state map")
        ax.text(0.02, 0.96, "colors: top cell states", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=COL["muted"])
        self.clean(ax, grid=True)

        ax = fig.add_subplot(gs[0, 2:4])
        self.label(ax, "b")
        counts = sc["cell_type_auto"].value_counts().head(12).iloc[::-1]
        y = np.arange(len(counts))
        colors = sns.color_palette("tab20", len(counts)).as_hex()
        ax.hlines(y, 0, counts.values, color=COL["grid"], lw=0.8)
        ax.scatter(counts.values, y, s=35 + counts.values / counts.max() * 210, c=colors, edgecolor="white", linewidth=0.6)
        for yi, (ct, n) in enumerate(counts.items()):
            ax.text(n + counts.max() * 0.025, yi, f"{n}", va="center", fontsize=5.0, color=COL["muted"])
        ax.set_yticks(y)
        ax.set_yticklabels([self.short(v, 17) for v in counts.index])
        ax.set_xlim(0, counts.max() * 1.22)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("cells")
        ax.set_title("Cell-type abundance bubble-lollipop")
        self.clean(ax, grid="x")

        ax = fig.add_subplot(gs[1, 2:4])
        self.label(ax, "c")
        sig = sc.melt(id_vars=["cell_type_auto"], value_vars=["bulk_Subtype_1", "bulk_Subtype_2", "bulk_Subtype_3"], var_name="signature", value_name="score")
        keep = sc["cell_type_auto"].value_counts().head(8).index
        sig = sig[sig["cell_type_auto"].isin(keep)]
        sns.boxplot(data=sig, x="cell_type_auto", y="score", hue="signature", showfliers=False, width=0.72, palette=[SUBTYPE[1], SUBTYPE[2], SUBTYPE[3]], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_title("Subtype signatures across cell states")
        ax.set_xlabel("")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=5.5, handlelength=1.2, columnspacing=0.8)
        self.clean(ax)

        ax = fig.add_subplot(gs[2, 0:2])
        self.label(ax, "d")
        numeric = key.set_index("gene").select_dtypes(include=[np.number])
        genes = numeric.max(axis=1).sort_values(ascending=False).head(12).index
        mat = numeric.loc[genes]
        sns.heatmap(mat, cmap="mako", ax=ax, cbar_kws=dict(label="mean expression", shrink=0.55), linewidths=0.25, linecolor="white")
        ax.set_title("Cell-type marker expression heatmap")
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax = fig.add_subplot(gs[2, 2])
        self.label(ax, "e")
        pb = sc.groupby("orig.ident")[["bulk_Subtype_1", "bulk_Subtype_2", "bulk_Subtype_3"]].mean()
        sns.heatmap(pb, cmap=sns.diverging_palette(240, 20, as_cmap=True), center=0, ax=ax, cbar=False, linewidths=0.25, linecolor="white")
        ax.set_title("Pseudo-bulk signature")
        ax.set_xlabel("")
        ax.set_ylabel("sample")

        ax = fig.add_subplot(gs[2, 3])
        self.label(ax, "f")
        comp = pd.crosstab(sc["DiseaseState"], sc["cell_type_auto"])
        comp = comp[comp.sum().sort_values(ascending=False).head(7).index]
        comp.div(comp.sum(axis=1), axis=0).plot(kind="bar", stacked=True, ax=ax, width=0.7, colormap="tab20")
        ax.set_title("Disease-state composition")
        ax.set_ylabel("fraction")
        ax.set_xlabel("")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        self.clean(ax)
        self.save(fig, "Figure9")

    def supplementary_s1(self) -> None:
        flow = self.csv("01_tcga_prep/figure_ready/Supplementary_Figure_S1B_filtering_cascade.csv")
        filt = self.csv("01_tcga_prep/figure_ready/Supplementary_Figure_S1C_feature_filtering.csv")
        event = self.csv("01_tcga_prep/figure_ready/Supplementary_Figure_S1D_event_distribution.csv")
        pca_b = self.csv("07_external_validation/harmonization_visuals/pca_before.csv")
        pca_a = self.csv("07_external_validation/harmonization_visuals/pca_after.csv")
        fig = plt.figure(figsize=(7.2, 4.9))
        gs = GridSpec(2, 4, figure=fig, hspace=0.62, wspace=0.7)
        ax = fig.add_subplot(gs[0, 0:2]); self.label(ax, "a")
        if not flow.empty:
            col = "n_patients" if "n_patients" in flow.columns else flow.select_dtypes("number").columns[-1]
            lab = "step" if "step" in flow.columns else flow.columns[0]
            self.lollipop(ax, flow[lab].astype(str).str.replace("_", "\n"), flow[col], COL["teal"], xlabel="patients")
        ax.set_title("Filtering cascade")
        ax = fig.add_subplot(gs[0, 2:4]); self.label(ax, "b")
        if not filt.empty:
            xcol = filt.columns[0]
            ycol = filt.select_dtypes("number").columns[-1]
            sns.barplot(data=filt, x=xcol, y=ycol, color=COL["blue"], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_title("Feature filtering")
        self.clean(ax)
        ax = fig.add_subplot(gs[1, 0:2]); self.label(ax, "c")
        if not event.empty:
            val_cols = [c for c in event.columns if "month" in c.lower() or "time" in c.lower()]
            if val_cols:
                for c in val_cols[:2]:
                    sns.kdeplot(event[c].dropna(), fill=True, alpha=0.30, ax=ax, label=c.replace("_", " "))
        ax.set_title("Event-time density")
        ax.legend(frameon=False)
        self.clean(ax)
        for j, (df, title) in enumerate([(pca_b, "Before"), (pca_a, "After")]):
            ax = fig.add_subplot(gs[1, 2 + j]); self.label(ax, chr(ord("d") + j))
            if not df.empty:
                sns.scatterplot(data=df, x="PC1", y="PC2", hue="cohort" if "cohort" in df.columns else None, s=8, linewidth=0, alpha=0.75, ax=ax, legend=False)
            ax.set_title(f"Harmonization {title.lower()}")
            self.clean(ax, grid=True)
        self.save(fig, "Supplementary_Figure_S1")

    def supplementary_s2(self) -> None:
        fold = self.csv("02_benchmark/benchmark_fold_level_metrics.csv")
        hp = self.csv("02_benchmark/benchmark_selected_hyperparams.csv")
        summary = self.csv("02_benchmark/benchmark_summary.csv")
        fig = plt.figure(figsize=(7.2, 4.8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.68, wspace=0.72)
        ax = fig.add_subplot(gs[0, 0:2]); self.label(ax, "a")
        self.raincloud(ax, fold, "mean_auc", "model", MODEL_COLORS, orient="h")
        ax.set_title("Mean AUC repeated-CV raincloud")
        ax = fig.add_subplot(gs[0, 2:4]); self.label(ax, "b")
        if "seed" in fold.columns:
            seed_mean = fold.groupby(["seed", "model"])["cindex"].mean().reset_index()
            sns.lineplot(data=seed_mean, x="seed", y="cindex", hue="model", palette=MODEL_COLORS, marker="o", lw=0.9, ms=3, ax=ax)
            ax.legend(frameon=False, ncol=2)
        ax.set_title("Seed-level robustness")
        self.clean(ax)
        ax = fig.add_subplot(gs[1, 0:2]); self.label(ax, "c")
        summ = summary.sort_values("composite_score")
        self.lollipop(ax, summ["model"], summ["composite_score"], COL["purple"], xlabel="composite score")
        ax.set_title("Composite model ranking")
        ax = fig.add_subplot(gs[1, 2:4]); self.label(ax, "d")
        if not hp.empty:
            nums = hp.select_dtypes(include=[np.number])
            if not nums.empty:
                sns.heatmap(nums.corr(), cmap=sns.diverging_palette(240, 20, as_cmap=True), center=0, ax=ax, cbar_kws=dict(shrink=0.55))
            else:
                self.empty(ax, "No numeric hyperparameters")
        ax.set_title("Hyperparameter correlation")
        self.save(fig, "Supplementary_Figure_S2")

    def supplementary_s3(self) -> None:
        stats = self.csv("06_immune/immune_feature_statistics.csv")
        train = self.csv("08_drug/drug_model_training_stats.csv")
        ext = self.csv("07_external_validation/external_summary.csv")
        sc = self.csv("09_singlecell/sc_cell_annotations.csv")
        fig = plt.figure(figsize=(7.2, 4.9))
        gs = GridSpec(2, 4, figure=fig, hspace=0.68, wspace=0.72)
        ax = fig.add_subplot(gs[0, 0:2]); self.label(ax, "a")
        extra = stats.sort_values("fdr").head(14).iloc[::-1]
        self.lollipop(ax, extra["feature"].str.replace("_PROXY", "").str.replace("CHECKPOINT__", ""), self.neglog(extra["fdr"]), COL["blue"], xlabel="-log10(FDR)")
        ax.set_title("Additional immune signatures")
        ax = fig.add_subplot(gs[0, 2:4]); self.label(ax, "b")
        if not train.empty:
            sns.scatterplot(data=train, x="train_n", y="cv_mse", hue="target_pathway" if "target_pathway" in train.columns else None, s=30, linewidth=0.25, edgecolor="white", ax=ax, legend=False)
        ax.set_title("Drug model training diagnostics")
        ax.set_xlabel("training cell lines")
        ax.set_ylabel("CV MSE")
        self.clean(ax)
        ax = fig.add_subplot(gs[1, 0:2]); self.label(ax, "c")
        if not ext.empty:
            sns.scatterplot(data=ext, x="n_expression", y="c_index", hue="cohort", s=55, ax=ax)
            ax.set_ylim(0.48, 0.62)
        ax.set_title("External performance by retained samples")
        ax.set_xlabel("expression samples")
        self.clean(ax)
        ax = fig.add_subplot(gs[1, 2:4]); self.label(ax, "d")
        if not sc.empty:
            qc = sc[["nCount_RNA", "nFeature_RNA", "percent.mt", "percent.ribo", "DiseaseState"]].sample(min(len(sc), 1800), random_state=7)
            sns.scatterplot(data=qc, x="nCount_RNA", y="nFeature_RNA", hue="DiseaseState", s=8, alpha=0.45, linewidth=0, ax=ax)
            ax.set_xscale("log")
        ax.set_title("Single-cell QC state space")
        self.clean(ax)
        self.save(fig, "Supplementary_Figure_S3")

    def export_target_panels(self) -> None:
        targets = [
            ("Figure2_C", (3.4, 2.2), lambda ax: self.draw_clinical_summary(ax, self.csv("01_tcga_prep/figure_ready/Figure2C_clinical_summary.csv"))),
            ("Figure2_F", (3.3, 2.2), lambda ax: self.draw_cohort_pca_ellipse(ax, self.csv("07_external_validation/harmonization_visuals/pca_after.csv"), "After harmonization", show_legend=True)),
            ("Figure3_E", (3.4, 2.2), lambda ax: self.draw_ablation_forest(ax, self.csv("02_benchmark/ablation_summary.csv"))),
            ("Figure3_F", (3.4, 2.2), lambda ax: self.draw_permutation_slope(ax, self.csv("02_benchmark/feature_contribution_summary.csv"))),
            ("Figure5_E", (3.2, 2.25), lambda ax: self.draw_hub_rank(ax, self.csv("05_biology/hub_genes.csv"))),
            ("Figure5_F", (3.4, 2.45), lambda ax: self.draw_dmp_heatmap(ax, self.csv("05_biology/dm_all_subtypes.csv"))),
            ("Figure8_F", (3.6, 2.55), lambda ax: self.draw_drug_evidence_matrix(ax, self.csv("08_drug/top_compounds_final.csv"), self.csv("08_drug/drug_marker_overlap.csv"))),
        ]
        for name, size, drawer in targets:
            fig, ax = plt.subplots(figsize=size)
            drawer(ax)
            self.save_panel(fig, name)

    def run(self) -> None:
        setup_style()
        self.figure1()
        self.figure2()
        self.figure3()
        self.figure4()
        self.figure5()
        self.figure6()
        self.figure7()
        self.figure8()
        self.figure9()
        self.supplementary_s1()
        self.supplementary_s2()
        self.supplementary_s3()
        self.export_target_panels()
        self.zip_outputs()

    def zip_outputs(self) -> None:
        zip_path = self.out / f"Pancreas_Figure_{self.tag}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(self.out.rglob("*.png")) + sorted(self.out.rglob("*.pdf")):
                if not p.name.startswith("_") and p.name != zip_path.name:
                    zf.write(p, p.relative_to(self.out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="pdac_project")
    parser.add_argument("--tag", default="nature_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    NatureV2(Path(args.project_root), tag=args.tag).run()


if __name__ == "__main__":
    main()
