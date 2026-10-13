#!/usr/bin/env python3
"""
Prompt 5: Immune microenvironment analysis by subtype/risk group.

Supports:
- CIBERSORT/xCell/ESTIMATE score integration from precomputed files
- ssGSEA-like immune signature scores
- checkpoint gene profiling
- cytolytic score
- TIDE-compatible proxy features
- stromal/hypoxia signatures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pdac_immune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)
    if "PATIENT_ID" in df.columns and (df.index.name is None or df.index.name.upper() != "PATIENT_ID"):
        df = df.set_index("PATIENT_ID")
    df.index = df.index.astype(str)
    return df


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    clean = np.where(np.isfinite(pvals), pvals, 1.0)
    order = np.argsort(clean)
    ranks = np.arange(1, n + 1)
    adj = np.empty(n, dtype=float)
    adj[order] = clean[order] * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order[::-1]])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj_sorted, 0.0, 1.0)
    return out


def parse_gmt(path: Path) -> Dict[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            gene_sets[p[0]] = [g for g in p[2:] if g]
    return gene_sets


def score_gene_sets(expr: pd.DataFrame, gene_sets: Dict[str, List[str]]) -> pd.DataFrame:
    z = expr.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) if c.std(ddof=0) > 0 else 1.0), axis=0)
    upper_map = {g.upper(): g for g in z.columns}

    rows = {}
    for name, genes in gene_sets.items():
        mapped = [upper_map[g.upper()] for g in genes if g.upper() in upper_map]
        if len(mapped) < 3:
            continue
        rows[name] = z[mapped].mean(axis=1)

    out = pd.DataFrame(rows, index=z.index)
    out.index.name = "PATIENT_ID"
    return out


def compare_feature_by_group(feature: pd.Series, group: pd.Series) -> Dict[str, float]:
    df = pd.DataFrame({"value": pd.to_numeric(feature, errors="coerce"), "group": group.astype(str)})
    df = df.dropna()
    if df["group"].nunique() < 2:
        return {"p_value": np.nan, "effect_size": np.nan, "test": "NA"}

    groups = [v["value"].to_numpy(dtype=float) for _, v in df.groupby("group")]

    if len(groups) == 2:
        stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        n1, n2 = len(groups[0]), len(groups[1])
        u = float(stat)
        effect = abs((2 * u) / (n1 * n2) - 1) if n1 > 0 and n2 > 0 else np.nan  # rank-biserial
        return {"p_value": float(p), "effect_size": float(effect), "test": "Mann-Whitney"}

    stat, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (stat - k + 1) / (n - k) if n > k else np.nan
    return {"p_value": float(p), "effect_size": float(max(eta2, 0.0) if np.isfinite(eta2) else np.nan), "test": "Kruskal-Wallis"}


def built_in_signatures() -> Dict[str, List[str]]:
    return {
        "T_CELL_INFLAMED": ["CD8A", "CD8B", "GZMB", "IFNG", "CXCL9", "CXCL10", "PRF1"],
        "CYTOTOXIC": ["GZMA", "GZMB", "PRF1", "NKG7"],
        "IFNG_SIGNATURE": ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1"],
        "TIDE_DYSFUNCTION_PROXY": ["PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2"],
        "TIDE_EXCLUSION_PROXY": ["COL1A1", "COL1A2", "FAP", "TGFB1", "CXCL12"],
        "STROMAL_SIGNATURE": ["COL1A1", "COL3A1", "FAP", "ACTA2", "TAGLN"],
        "HYPOXIA_SIGNATURE": ["HIF1A", "VEGFA", "SLC2A1", "CA9", "LDHA", "ENO1"],
    }


def merge_external_scores(base: pd.DataFrame, external_file: Optional[str], prefix: str) -> pd.DataFrame:
    if not external_file:
        return base
    path = Path(external_file)
    if not path.exists():
        return base

    ext = read_matrix(path)
    ext = ext.add_prefix(prefix + "__")
    common = sorted(set(base.index) & set(ext.index))
    if not common:
        return base

    merged = base.loc[common].join(ext.loc[common], how="left")
    return merged


def append_gap_note(gap_path: Optional[str], note: str) -> None:
    if not gap_path:
        return
    path = Path(gap_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(note.rstrip() + "\n")


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "immune_analysis.log")

    try:
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")
        expr = read_matrix(Path(args.expression)).apply(pd.to_numeric, errors="coerce")

        sub = pd.read_csv(args.subtypes)
        if "PATIENT_ID" not in sub.columns or "subtype" not in sub.columns:
            raise ValueError("Subtype file must contain PATIENT_ID and subtype columns.")
        subtype = sub.set_index("PATIENT_ID")["subtype"].astype(str)

        common = sorted(set(expr.index) & set(subtype.index))
        if len(common) < 20:
            raise ValueError("Too few overlapping patients between expression and subtype assignments.")

        expr = expr.loc[common]
        subtype = subtype.loc[common]

        scores = pd.DataFrame(index=common)

        # Built-in signatures and optional external GMT signatures
        signatures = built_in_signatures()
        if args.immune_gmt and Path(args.immune_gmt).exists():
            signatures.update(parse_gmt(Path(args.immune_gmt)))

        sig_scores = score_gene_sets(expr, signatures)
        scores = scores.join(sig_scores, how="left")

        # Cytolytic score
        g1 = expr.get("GZMA", pd.Series(np.nan, index=expr.index))
        g2 = expr.get("PRF1", pd.Series(np.nan, index=expr.index))
        scores["CYTOLYTIC_SCORE"] = np.sqrt(np.clip(g1, 0, None) * np.clip(g2, 0, None))

        # Checkpoint genes
        checkpoints = [g.strip() for g in args.checkpoint_genes.split(",") if g.strip()]
        for g in checkpoints:
            if g in expr.columns:
                scores[f"CHECKPOINT__{g}"] = expr[g]

        # HLA / antigen presentation panel
        hla_genes = [g.strip() for g in args.hla_genes.split(",") if g.strip()]
        for g in hla_genes:
            if g in expr.columns:
                scores[f"HLA__{g}"] = expr[g]

        # External CIBERSORT/xCell/ESTIMATE inputs (optional)
        if not args.cibersort_file or not Path(args.cibersort_file).exists():
            append_gap_note(
                args.gap_report,
                "- Immune module: official CIBERSORT matrix unavailable; reporting signature/proxy-based immune composition.",
            )
        if not args.xcell_file or not Path(args.xcell_file).exists():
            append_gap_note(
                args.gap_report,
                "- Immune module: official xCell matrix unavailable; reporting signature/proxy-based stromal/immune scores.",
            )
        if not args.estimate_file or not Path(args.estimate_file).exists():
            append_gap_note(
                args.gap_report,
                "- Immune module: official ESTIMATE scores unavailable; reporting compatible stromal/hypoxia proxies.",
            )
        scores = merge_external_scores(scores, args.cibersort_file, "CIBERSORT")
        scores = merge_external_scores(scores, args.xcell_file, "XCELL")
        scores = merge_external_scores(scores, args.estimate_file, "ESTIMATE")

        # Optional risk-group integration
        risk_group = None
        if args.risk_scores and Path(args.risk_scores).exists():
            risk = read_matrix(Path(args.risk_scores))
            common2 = sorted(set(scores.index) & set(risk.index))
            if common2:
                scores = scores.loc[common2]
                subtype = subtype.loc[common2]
                risk = risk.loc[common2]
                risk_col = risk.columns[0]
                med = float(risk[risk_col].median())
                risk_group = pd.Series(np.where(risk[risk_col] >= med, "High", "Low"), index=common2)
                scores["RISK_SCORE"] = risk[risk_col]

        # Statistics by subtype
        stat_rows = []
        for feat in scores.columns:
            cmp_sub = compare_feature_by_group(scores[feat], subtype)
            stat_rows.append(
                {
                    "feature": feat,
                    "grouping": "subtype",
                    "test": cmp_sub["test"],
                    "p_value": cmp_sub["p_value"],
                    "effect_size": cmp_sub["effect_size"],
                }
            )

            if risk_group is not None and feat != "RISK_SCORE":
                cmp_risk = compare_feature_by_group(scores[feat], risk_group)
                stat_rows.append(
                    {
                        "feature": feat,
                        "grouping": "risk_group",
                        "test": cmp_risk["test"],
                        "p_value": cmp_risk["p_value"],
                        "effect_size": cmp_risk["effect_size"],
                    }
                )

        stat_df = pd.DataFrame(stat_rows)
        if not stat_df.empty:
            stat_df["fdr"] = bh_fdr(stat_df["p_value"].fillna(1.0).to_numpy())
            stat_df = stat_df.sort_values(["grouping", "fdr", "p_value"]).reset_index(drop=True)

        scores.index.name = "PATIENT_ID"
        scores.to_csv(out_dir / "immune_feature_matrix.csv")
        stat_df.to_csv(out_dir / "immune_feature_statistics.csv", index=False)
        scores.to_csv(out_dir / "immune_signature_scores.csv")
        stat_df.to_csv(out_dir / "immune_comparison_stats.csv", index=False)

        chk_mask = stat_df["feature"].astype(str).str.startswith("CHECKPOINT__") if not stat_df.empty else pd.Series([], dtype=bool)
        hla_mask = stat_df["feature"].astype(str).str.startswith("HLA__") if not stat_df.empty else pd.Series([], dtype=bool)
        evasion_mask = (
            stat_df["feature"].astype(str).str.contains("TIDE_DYSFUNCTION_PROXY|TIDE_EXCLUSION_PROXY|STROMAL_SIGNATURE|HYPOXIA_SIGNATURE", regex=True)
            if not stat_df.empty
            else pd.Series([], dtype=bool)
        )
        cyt_mask = (
            stat_df["feature"].astype(str).str.contains("CYTOLYTIC_SCORE|IFNG_SIGNATURE|T_CELL_INFLAMED", regex=True)
            if not stat_df.empty
            else pd.Series([], dtype=bool)
        )
        stat_df.loc[chk_mask].to_csv(out_dir / "checkpoint_expression_stats.csv", index=False)
        stat_df.loc[hla_mask].to_csv(out_dir / "hla_antigen_stats.csv", index=False)
        stat_df.loc[cyt_mask].to_csv(out_dir / "cytolytic_ifng_tcell_scores.csv", index=False)
        stat_df.loc[evasion_mask].to_csv(out_dir / "immune_evasion_proxy_stats.csv", index=False)

        # Cold vs hot summary using inflamed and exclusion proxies.
        inflamed = pd.to_numeric(scores.get("T_CELL_INFLAMED", pd.Series(np.nan, index=scores.index)), errors="coerce")
        exclusion = pd.to_numeric(scores.get("TIDE_EXCLUSION_PROXY", pd.Series(np.nan, index=scores.index)), errors="coerce")
        inflamed_cut = float(inflamed.median(skipna=True)) if inflamed.notna().any() else np.nan
        exclusion_cut = float(exclusion.median(skipna=True)) if exclusion.notna().any() else np.nan
        cold_hot = pd.DataFrame(index=scores.index)
        cold_hot["subtype"] = subtype
        if risk_group is not None:
            cold_hot["risk_group"] = risk_group
        cold_hot["T_CELL_INFLAMED"] = inflamed
        cold_hot["TIDE_EXCLUSION_PROXY"] = exclusion
        cold_hot["immune_state"] = np.where(
            (inflamed >= inflamed_cut) & (exclusion < exclusion_cut),
            "Hot",
            np.where((inflamed < inflamed_cut) & (exclusion >= exclusion_cut), "Cold", "Intermediate"),
        )
        ch_rows = []
        tab_sub = pd.crosstab(cold_hot["subtype"], cold_hot["immune_state"])
        if tab_sub.shape[0] >= 2 and tab_sub.shape[1] >= 2:
            chi2, p, _, _ = stats.chi2_contingency(tab_sub)
        else:
            p = np.nan
        for st, row in tab_sub.iterrows():
            for state, v in row.items():
                ch_rows.append({"grouping": "subtype", "group": str(st), "immune_state": str(state), "count": int(v), "p_value": float(p) if np.isfinite(p) else np.nan})

        if risk_group is not None:
            tab_risk = pd.crosstab(cold_hot["risk_group"], cold_hot["immune_state"])
            if tab_risk.shape[0] >= 2 and tab_risk.shape[1] >= 2:
                chi2, p2, _, _ = stats.chi2_contingency(tab_risk)
            else:
                p2 = np.nan
            for rg, row in tab_risk.iterrows():
                for state, v in row.items():
                    ch_rows.append({"grouping": "risk_group", "group": str(rg), "immune_state": str(state), "count": int(v), "p_value": float(p2) if np.isfinite(p2) else np.nan})

        cold_hot.to_csv(out_dir / "cold_hot_patient_level.csv")
        pd.DataFrame(ch_rows).to_csv(out_dir / "cold_hot_summary.csv", index=False)

        # Group means for publication-style heatmap
        subtype_means = scores.groupby(subtype).mean(numeric_only=True)
        subtype_means.to_csv(out_dir / "immune_feature_means_by_subtype.csv")

        top_features = stat_df[stat_df["grouping"] == "subtype"].head(args.top_n_plot)["feature"].tolist()
        top_features = [f for f in top_features if f in scores.columns]
        if top_features:
            hm = subtype_means[top_features].T
            plt.figure(figsize=(10, max(5, 0.25 * hm.shape[0])))
            sns.heatmap(hm, cmap="coolwarm", center=0)
            plt.title("Immune features by subtype (top significant)")
            plt.tight_layout()
            plt.savefig(fig_dir / "immune_heatmap_by_subtype.png", dpi=300)
            plt.close()

        # Boxplots for selected features
        for feat in top_features[: min(8, len(top_features))]:
            plot_df = pd.DataFrame({"value": scores[feat], "subtype": subtype}).dropna()
            plt.figure(figsize=(7, 5))
            sns.boxplot(data=plot_df, x="subtype", y="value")
            sns.stripplot(data=plot_df, x="subtype", y="value", color="black", alpha=0.35, size=3)
            plt.title(feat)
            plt.tight_layout()
            plt.savefig(fig_dir / f"boxplot_{feat.replace('/', '_')}.png", dpi=300)
            plt.close()

        if risk_group is not None:
            risk_means = scores.groupby(risk_group).mean(numeric_only=True)
            risk_means.to_csv(out_dir / "immune_feature_means_by_risk_group.csv")

        logger.info("Immune analysis complete.")
        return 0

    except Exception as exc:
        logger.exception("Immune analysis failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDAC immune microenvironment analysis.")
    p.add_argument("--expression", required=True, help="Expression matrix parquet/csv.")
    p.add_argument("--subtypes", required=True, help="Subtype assignment CSV with PATIENT_ID, subtype.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--risk-scores", default=None, help="Optional benchmark risk score matrix.")
    p.add_argument("--immune-gmt", default=None, help="Optional immune signature GMT.")

    p.add_argument("--cibersort-file", default=None, help="Optional precomputed CIBERSORT score matrix.")
    p.add_argument("--xcell-file", default=None, help="Optional precomputed xCell score matrix.")
    p.add_argument("--estimate-file", default=None, help="Optional precomputed ESTIMATE score matrix.")

    p.add_argument(
        "--checkpoint-genes",
        default="PDCD1,CD274,CTLA4,LAG3,TIGIT,HAVCR2,IDO1",
        help="Comma-separated checkpoint genes to profile.",
    )
    p.add_argument(
        "--hla-genes",
        default="HLA-A,HLA-B,HLA-C,B2M,TAP1,TAP2,NLRC5,PSMB8,PSMB9",
        help="Comma-separated HLA/antigen presentation genes to profile.",
    )
    p.add_argument("--gap-report", default=None, help="Optional gap report markdown path.")
    p.add_argument("--top-n-plot", type=int, default=25)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
