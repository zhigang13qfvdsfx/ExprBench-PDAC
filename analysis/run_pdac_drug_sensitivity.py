#!/usr/bin/env python3
"""
Prompt 6: Drug sensitivity prediction and prioritization.

Workflow:
- Train drug-wise ElasticNet models on reference pharmacogenomics data
- Predict drug response for PDAC cohort
- Compare predicted response across subtype/risk groups
- Rank compounds by effect size, significance, and mechanistic relevance
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
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pdac_drug")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
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


def compare_by_group(values: pd.Series, groups: pd.Series) -> Dict[str, float]:
    df = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups.astype(str)}).dropna()
    if df["group"].nunique() < 2:
        return {"p_value": np.nan, "effect_size": np.nan, "test": "NA"}

    arrays = [v["value"].to_numpy(dtype=float) for _, v in df.groupby("group")]
    if len(arrays) == 2:
        stat, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        n1, n2 = len(arrays[0]), len(arrays[1])
        u = float(stat)
        eff = abs((2 * u) / (n1 * n2) - 1) if n1 > 0 and n2 > 0 else np.nan
        return {"p_value": float(p), "effect_size": float(eff), "test": "Mann-Whitney"}

    stat, p = stats.kruskal(*arrays)
    n = sum(len(v) for v in arrays)
    k = len(arrays)
    eta2 = (stat - k + 1) / (n - k) if n > k else np.nan
    return {"p_value": float(p), "effect_size": float(max(eta2, 0.0) if np.isfinite(eta2) else np.nan), "test": "Kruskal-Wallis"}


def build_group_series(subtype_file: Path, risk_scores_file: Optional[Path]) -> pd.Series:
    sub = pd.read_csv(subtype_file)
    if "PATIENT_ID" not in sub.columns or "subtype" not in sub.columns:
        raise ValueError("Subtype file must include PATIENT_ID and subtype columns.")

    g = sub.set_index("PATIENT_ID")["subtype"].astype(str)

    if risk_scores_file is not None and risk_scores_file.exists():
        risk = read_matrix(risk_scores_file)
        common = sorted(set(g.index) & set(risk.index))
        if common:
            g = g.loc[common]
            risk = risk.loc[common]
            risk_col = risk.columns[0]
            med = float(risk[risk_col].median())
            risk_group = pd.Series(np.where(risk[risk_col] >= med, "HighRisk", "LowRisk"), index=common)
            g = pd.Series([f"S{g.loc[idx]}_{risk_group.loc[idx]}" for idx in common], index=common)

    return g


def parse_drug_annotation(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug", "target_pathway", "target_genes"])
    ann = pd.read_csv(path)
    ann.columns = [c.strip().lower() for c in ann.columns]
    for col in ["drug", "target_pathway", "target_genes"]:
        if col not in ann.columns:
            ann[col] = ""
    return ann[["drug", "target_pathway", "target_genes"]]


def parse_marker_genes(path: Optional[Path], fdr_cutoff: float) -> set:
    if path is None or not path.exists():
        return set()
    df = pd.read_csv(path)
    feature_col = "feature" if "feature" in df.columns else ("gene" if "gene" in df.columns else None)
    if feature_col is None:
        return set()
    if "fdr" in df.columns:
        df = df[df["fdr"] <= fdr_cutoff]
    return set(df[feature_col].astype(str).tolist())


def load_subtype_series(subtype_file: Path) -> pd.Series:
    sub = pd.read_csv(subtype_file)
    if "PATIENT_ID" not in sub.columns or "subtype" not in sub.columns:
        raise ValueError("Subtype file must include PATIENT_ID and subtype columns.")
    return sub.set_index("PATIENT_ID")["subtype"].astype(str)


def load_risk_group(risk_scores_file: Optional[Path]) -> Optional[pd.Series]:
    if risk_scores_file is None or (not risk_scores_file.exists()):
        return None
    risk = read_matrix(risk_scores_file)
    risk_col = risk.columns[0]
    med = float(pd.to_numeric(risk[risk_col], errors="coerce").median(skipna=True))
    grp = pd.Series(np.where(pd.to_numeric(risk[risk_col], errors="coerce") >= med, "High", "Low"), index=risk.index.astype(str))
    return grp


def compare_drug_matrix_by_group(pred_df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for drug in pred_df.columns:
        cmp = compare_by_group(pred_df[drug], groups)
        means = pred_df.groupby(groups)[drug].mean()
        row = {
            "drug": drug,
            "test": cmp["test"],
            "p_value": cmp["p_value"],
            "effect_size": cmp["effect_size"],
            "group_means": "; ".join([f"{k}:{v:.4f}" for k, v in means.items()]),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["p_value"].fillna(1.0).to_numpy())
        out = out.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    return out


def select_shared_genes(
    pdac_expr: pd.DataFrame,
    ref_expr: pd.DataFrame,
    max_shared_genes: int,
) -> List[str]:
    shared = sorted(set(pdac_expr.columns) & set(ref_expr.columns))
    if max_shared_genes <= 0 or len(shared) <= max_shared_genes:
        return shared
    var = ref_expr[shared].var(axis=0, numeric_only=True).sort_values(ascending=False)
    return var.head(max_shared_genes).index.astype(str).tolist()


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "drug_sensitivity.log")

    try:
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")
        pdac_expr = read_matrix(Path(args.expression)).apply(pd.to_numeric, errors="coerce")
        ref_expr = read_matrix(Path(args.ref_expression)).apply(pd.to_numeric, errors="coerce")
        ref_resp = read_matrix(Path(args.ref_response)).apply(pd.to_numeric, errors="coerce")

        subtype_groups = load_subtype_series(Path(args.subtypes))
        risk_groups = load_risk_group(Path(args.risk_scores) if args.risk_scores else None)

        common_pat = sorted(set(pdac_expr.index) & set(subtype_groups.index))
        if len(common_pat) < 20:
            raise ValueError("Too few overlapping PDAC samples between expression and group labels.")

        pdac_expr = pdac_expr.loc[common_pat]
        subtype_groups = subtype_groups.loc[common_pat]
        if risk_groups is not None:
            risk_groups = risk_groups.reindex(common_pat)

        shared_genes = select_shared_genes(pdac_expr, ref_expr, args.max_shared_genes)
        if len(shared_genes) < args.min_shared_genes:
            raise ValueError(
                f"Only {len(shared_genes)} shared genes between PDAC and reference expression; need at least {args.min_shared_genes}."
            )

        pdac_x = pdac_expr[shared_genes].to_numpy(dtype=float)
        ref_x = ref_expr[shared_genes].to_numpy(dtype=float)

        # Standardize with reference statistics for transfer-like prediction.
        scaler = StandardScaler()
        ref_x = scaler.fit_transform(ref_x)
        pdac_x = scaler.transform(pdac_x)

        drug_preds: Dict[str, np.ndarray] = {}
        model_stats: List[Dict[str, float]] = []

        common_ref = sorted(set(ref_expr.index) & set(ref_resp.index))
        ref_resp = ref_resp.loc[common_ref]
        ref_expr_aligned = ref_expr.loc[common_ref, shared_genes]
        ref_x_aligned = scaler.transform(ref_expr_aligned.to_numpy(dtype=float))

        candidate_drugs = list(ref_resp.columns)
        if args.max_drugs > 0 and len(candidate_drugs) > args.max_drugs:
            nonna = ref_resp[candidate_drugs].notna().sum(axis=0).sort_values(ascending=False)
            candidate_drugs = nonna.head(args.max_drugs).index.astype(str).tolist()

        for idx_drug, drug in enumerate(candidate_drugs, start=1):
            y = pd.to_numeric(ref_resp[drug], errors="coerce")
            ok = y.notna().to_numpy()
            if ok.sum() < args.min_train_samples:
                continue

            x_train = ref_x_aligned[ok]
            y_train = y.to_numpy(dtype=float)[ok]

            try:
                if args.fast_mode:
                    enet = ElasticNet(
                        alpha=args.fast_alpha,
                        l1_ratio=args.fast_l1_ratio,
                        max_iter=3000,
                        random_state=args.seed,
                    )
                    enet.fit(x_train, y_train)
                    pred = enet.predict(pdac_x)
                    alpha_val = float(args.fast_alpha)
                    l1_val = float(args.fast_l1_ratio)
                    cv_mse = float(np.mean((enet.predict(x_train) - y_train) ** 2))
                else:
                    enet = ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9],
                        n_alphas=40,
                        cv=max(2, args.cv_folds),
                        max_iter=4000,
                        random_state=args.seed,
                    )
                    enet.fit(x_train, y_train)
                    pred = enet.predict(pdac_x)
                    alpha_val = float(enet.alpha_)
                    l1_val = float(enet.l1_ratio_)
                    cv_mse = float(np.min(np.mean(enet.mse_path_, axis=1)))
                drug_preds[drug] = pred
                model_stats.append(
                    {
                        "drug": drug,
                        "train_n": int(ok.sum()),
                        "alpha": alpha_val,
                        "l1_ratio": l1_val,
                        "cv_mse": cv_mse,
                    }
                )
                if idx_drug % 10 == 0:
                    logger.info("Trained %d/%d drug models", idx_drug, len(candidate_drugs))
            except Exception as exc:
                logger.warning("Drug %s model failed: %s", drug, exc)

        if not drug_preds:
            raise RuntimeError("No drug model could be fit. Check reference files.")

        pred_df = pd.DataFrame(drug_preds, index=common_pat)
        pred_df.index.name = "PATIENT_ID"
        pred_df.to_csv(out_dir / "predicted_drug_response_matrix.csv")

        pd.DataFrame(model_stats).to_csv(out_dir / "drug_model_training_stats.csv", index=False)

        # Group comparisons
        cmp_sub = compare_drug_matrix_by_group(pred_df, subtype_groups)
        cmp_sub.to_csv(out_dir / "subtype_drug_comparison.csv", index=False)
        if risk_groups is not None and risk_groups.notna().sum() >= 20 and risk_groups.nunique(dropna=True) >= 2:
            cmp_risk = compare_drug_matrix_by_group(pred_df.loc[risk_groups.dropna().index], risk_groups.dropna())
        else:
            cmp_risk = pd.DataFrame(columns=["drug", "test", "p_value", "effect_size", "group_means", "fdr"])
        cmp_risk.to_csv(out_dir / "riskgroup_drug_comparison.csv", index=False)

        cmp_df = cmp_sub.copy()
        if not cmp_risk.empty:
            cmp_df = cmp_df.merge(
                cmp_risk[["drug", "p_value", "fdr", "effect_size"]].rename(
                    columns={
                        "p_value": "risk_p_value",
                        "fdr": "risk_fdr",
                        "effect_size": "risk_effect_size",
                    }
                ),
                on="drug",
                how="left",
            )

        # Mechanistic relevance score
        ann = parse_drug_annotation(Path(args.drug_annotation) if args.drug_annotation else None)
        marker_genes = parse_marker_genes(Path(args.marker_genes) if args.marker_genes else None, args.marker_fdr)

        ann_map = ann.set_index("drug") if not ann.empty else pd.DataFrame()
        relevance = []
        overlap_rows = []
        for drug in cmp_df["drug"]:
            score = 0.0
            pathway = ""
            targets = ""
            overlap = 0
            if not ann_map.empty and drug in ann_map.index:
                pathway = str(ann_map.loc[drug, "target_pathway"])
                targets = str(ann_map.loc[drug, "target_genes"])
                tg = [g.strip() for g in targets.replace(";", ",").split(",") if g.strip()]
                if marker_genes and tg:
                    overlap = len(set([g.upper() for g in tg]) & set([g.upper() for g in marker_genes]))
                    score += overlap
            relevance.append({"drug": drug, "target_pathway": pathway, "target_genes": targets, "mechanistic_score": score})
            overlap_rows.append({"drug": drug, "target_genes": targets, "overlap_marker_genes_n": int(overlap)})

        rel_df = pd.DataFrame(relevance)
        cmp_df = cmp_df.merge(rel_df, on="drug", how="left")
        overlap_df = pd.DataFrame(overlap_rows)
        overlap_df.to_csv(out_dir / "drug_marker_overlap.csv", index=False)
        rel_df.to_csv(out_dir / "drug_target_pathway_summary.csv", index=False)

        # Prioritization score: larger effect, smaller p, higher mechanistic score
        cmp_df["neglog10_fdr"] = -np.log10(np.clip(cmp_df["fdr"].fillna(1.0), 1e-300, 1.0))
        cmp_df["neglog10_risk_fdr"] = -np.log10(np.clip(cmp_df.get("risk_fdr", pd.Series(1.0, index=cmp_df.index)).fillna(1.0), 1e-300, 1.0))
        eff = cmp_df["effect_size"].fillna(0.0)
        risk_eff = cmp_df.get("risk_effect_size", pd.Series(0.0, index=cmp_df.index)).fillna(0.0)
        cmp_df["priority_score"] = (
            (eff - eff.mean()) / (eff.std(ddof=0) if eff.std(ddof=0) > 0 else 1.0)
            + (cmp_df["neglog10_fdr"] - cmp_df["neglog10_fdr"].mean())
            / (cmp_df["neglog10_fdr"].std(ddof=0) if cmp_df["neglog10_fdr"].std(ddof=0) > 0 else 1.0)
            + (risk_eff - risk_eff.mean()) / (risk_eff.std(ddof=0) if risk_eff.std(ddof=0) > 0 else 1.0)
            + (cmp_df["neglog10_risk_fdr"] - cmp_df["neglog10_risk_fdr"].mean())
            / (cmp_df["neglog10_risk_fdr"].std(ddof=0) if cmp_df["neglog10_risk_fdr"].std(ddof=0) > 0 else 1.0)
            + cmp_df["mechanistic_score"].fillna(0.0)
        )

        cmp_df = cmp_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        cmp_df.to_csv(out_dir / "drug_group_comparison.csv", index=False)
        cmp_df.to_csv(out_dir / "drug_ranking.csv", index=False)
        cmp_df.head(args.top_n).to_csv(out_dir / "prioritized_compounds_top.csv", index=False)
        cmp_df.head(args.top_n).to_csv(out_dir / "top_compounds_final.csv", index=False)

        # Group mean heatmap (top compounds)
        top_drugs = cmp_df.head(args.top_n)["drug"].tolist()
        gm = pred_df[top_drugs].groupby(subtype_groups).mean().T
        plt.figure(figsize=(10, max(5, 0.28 * gm.shape[0])))
        sns.heatmap(gm, cmap="coolwarm", center=gm.to_numpy().mean())
        plt.title("Predicted drug sensitivity by subtype (top prioritized)")
        plt.tight_layout()
        plt.savefig(fig_dir / "drug_sensitivity_heatmap_top.png", dpi=300)
        plt.close()

        # Effect-size vs significance
        plt.figure(figsize=(7, 6))
        plot_df = cmp_df.copy()
        sns.scatterplot(data=plot_df, x="effect_size", y="neglog10_fdr")
        top = plot_df.head(min(20, len(plot_df)))
        for _, row in top.iterrows():
            plt.text(row["effect_size"], row["neglog10_fdr"], str(row["drug"]), fontsize=7)
        plt.xlabel("Effect size")
        plt.ylabel("-log10(FDR)")
        plt.title("Drug prioritization landscape")
        plt.tight_layout()
        plt.savefig(fig_dir / "drug_prioritization_scatter.png", dpi=300)
        plt.close()

        logger.info("Drug sensitivity analysis complete.")
        return 0

    except Exception as exc:
        logger.exception("Drug sensitivity analysis failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDAC drug sensitivity prediction and prioritization.")
    p.add_argument("--expression", required=True, help="PDAC expression matrix parquet/csv.")
    p.add_argument("--subtypes", required=True, help="Subtype assignment CSV (PATIENT_ID, subtype).")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--ref-expression", required=True, help="Reference cell-line expression matrix.")
    p.add_argument("--ref-response", required=True, help="Reference drug-response matrix (same sample IDs as ref expression).")

    p.add_argument("--risk-scores", default=None, help="Optional risk score matrix for combined subtype/risk grouping.")
    p.add_argument("--drug-annotation", default=None, help="Optional drug annotation CSV (drug,target_pathway,target_genes).")
    p.add_argument("--marker-genes", default=None, help="Optional DE table for mechanistic overlap scoring.")

    p.add_argument("--marker-fdr", type=float, default=0.05)
    p.add_argument("--min-shared-genes", type=int, default=500)
    p.add_argument("--max-shared-genes", type=int, default=3000)
    p.add_argument("--min-train-samples", type=int, default=30)
    p.add_argument("--max-drugs", type=int, default=160)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--fast-mode", action="store_true", help="Use fixed-parameter ElasticNet without inner CV for speed.")
    p.add_argument("--fast-alpha", type=float, default=0.01)
    p.add_argument("--fast-l1-ratio", type=float, default=0.1)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
