#!/usr/bin/env python3
"""
Prompt 3: PDAC subtype discovery from integrated representation.

- Compares k=2,3,4 (customizable) by consensus clustering
- Computes silhouette, gap statistic, and stability (pairwise ARI)
- Performs survival comparison (log-rank)
- Exports assignments, metrics, and figure outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pdac_subtypes")
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


def choose_representation_file(path: Path) -> Path:
    if path.exists():
        return path
    alt_csv = path.with_suffix(".csv")
    if alt_csv.exists():
        return alt_csv
    raise FileNotFoundError(f"Representation not found: {path} or {alt_csv}")


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


def within_cluster_dispersion(x: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels):
        members = x[labels == c]
        if members.shape[0] <= 1:
            continue
        center = members.mean(axis=0, keepdims=True)
        total += float(np.sum((members - center) ** 2))
    return max(total, 1e-12)


def gap_statistic(x: np.ndarray, k: int, n_refs: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(x)
    wk = within_cluster_dispersion(x, labels)

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ref_logs = []
    for i in range(n_refs):
        xr = rng.uniform(mins, maxs, size=x.shape)
        km_r = KMeans(n_clusters=k, n_init=10, random_state=seed + i + 1)
        lb_r = km_r.fit_predict(xr)
        wk_r = within_cluster_dispersion(xr, lb_r)
        ref_logs.append(np.log(wk_r))

    ref_logs = np.asarray(ref_logs, dtype=float)
    gap = float(np.mean(ref_logs) - np.log(wk))
    sk = float(np.std(ref_logs, ddof=1) * np.sqrt(1.0 + 1.0 / n_refs)) if n_refs > 1 else 0.0
    return gap, sk


def consensus_clustering(
    x: np.ndarray,
    k: int,
    repeats: int,
    subsample_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    m = max(k + 1, int(round(n * subsample_fraction)))
    m = min(m, n)

    cocluster = np.zeros((n, n), dtype=float)
    cooccur = np.zeros((n, n), dtype=float)

    all_full_labels: List[np.ndarray] = []

    for r in range(repeats):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        x_sub = x[idx]

        km = KMeans(n_clusters=k, n_init=20, random_state=seed + r)
        labels_sub = km.fit_predict(x_sub)

        cooccur[np.ix_(idx, idx)] += 1.0

        for c in range(k):
            members = idx[labels_sub == c]
            if members.size == 0:
                continue
            cocluster[np.ix_(members, members)] += 1.0

        full_dist = cdist(x, km.cluster_centers_)
        full_labels = np.argmin(full_dist, axis=1)
        all_full_labels.append(full_labels)

    consensus = np.divide(cocluster, cooccur, out=np.zeros_like(cocluster), where=cooccur > 0)
    np.fill_diagonal(consensus, 1.0)

    dist = 1.0 - consensus
    try:
        agg = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        agg = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    final_labels = agg.fit_predict(dist)

    ari_vals = []
    for i in range(len(all_full_labels)):
        for j in range(i + 1, len(all_full_labels)):
            ari_vals.append(adjusted_rand_score(all_full_labels[i], all_full_labels[j]))

    stability_mean = float(np.mean(ari_vals)) if ari_vals else np.nan
    stability_std = float(np.std(ari_vals)) if ari_vals else np.nan
    return final_labels, consensus, stability_mean, stability_std


def survival_compare(
    clinical_df: pd.DataFrame,
    patient_ids: Sequence[str],
    labels: np.ndarray,
    endpoint: str,
) -> Dict[str, float]:
    endpoint = endpoint.upper()
    t_col = f"{endpoint}_TIME"
    e_col = f"{endpoint}_EVENT"
    if t_col not in clinical_df.columns or e_col not in clinical_df.columns:
        return {"logrank_p": np.nan, "n_events": np.nan}

    clin = clinical_df.reindex(patient_ids)
    t = pd.to_numeric(clin[t_col], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(clin[e_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(t) & np.isfinite(e)
    t = t[ok]
    e = e[ok].astype(int)
    g = labels[ok]

    try:
        from lifelines.statistics import multivariate_logrank_test

        res = multivariate_logrank_test(t, g, e)
        pval = float(res.p_value)
    except Exception:
        pval = np.nan

    return {"logrank_p": pval, "n_events": int(e.sum())}


def make_latent_coordinates(rep: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, str]:
    x = rep.to_numpy(dtype=float)
    method = "pca"
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=2, random_state=seed)
        emb = reducer.fit_transform(x)
        method = "umap"
    except Exception:
        pca = PCA(n_components=2, random_state=seed)
        emb = pca.fit_transform(x)

    out = pd.DataFrame({"PATIENT_ID": rep.index.astype(str), "DIM1": emb[:, 0], "DIM2": emb[:, 1]})
    return out, method


def clinicopathologic_summary_table(clinical_df: pd.DataFrame, assign_df: pd.DataFrame) -> pd.DataFrame:
    labels = assign_df.set_index("PATIENT_ID")["subtype"].astype(int)
    clin = clinical_df.reindex(labels.index).copy()
    sub = labels

    preferred = [
        "AGE",
        "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS",
        "SEX",
        "GENDER",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "AJCC_PATHOLOGIC_STAGE",
        "PATHOLOGIC_STAGE",
        "TUMOR_STAGE",
        "HISTOLOGICAL_GRADE",
        "NEOPLASM_HISTOLOGIC_GRADE",
        "GRADE",
    ]
    available = [c for c in preferred if c in clin.columns]
    if not available:
        available = list(clin.columns[: min(12, clin.shape[1])])

    rows: List[Dict[str, object]] = []
    for col in available:
        s = clin[col]
        numeric = pd.to_numeric(s, errors="coerce")
        n_unique_num = numeric.dropna().nunique()

        if n_unique_num >= 8:
            groups = []
            for k in sorted(sub.unique()):
                vals = numeric[sub == k].dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    groups.append(vals)
            if len(groups) >= 2:
                try:
                    stat, p = stats.kruskal(*groups)
                    n = sum(len(g) for g in groups)
                    kk = len(groups)
                    eta2 = (stat - kk + 1) / (n - kk) if n > kk else np.nan
                except Exception:
                    stat, p, eta2 = np.nan, np.nan, np.nan
                row: Dict[str, object] = {
                    "feature": col,
                    "type": "numeric",
                    "test": "Kruskal-Wallis",
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "effect_size": float(max(eta2, 0.0) if np.isfinite(eta2) else np.nan),
                }
                for k in sorted(sub.unique()):
                    vals = numeric[sub == k].dropna()
                    row[f"subtype_{int(k)}_median"] = float(vals.median()) if len(vals) else np.nan
                    row[f"subtype_{int(k)}_n"] = int(vals.shape[0])
                rows.append(row)
            continue

        cat = s.astype(str).replace({"nan": np.nan, "None": np.nan})
        tab = pd.crosstab(cat, sub, dropna=True)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        try:
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            n = tab.to_numpy().sum()
            min_dim = min(tab.shape[0] - 1, tab.shape[1] - 1)
            cramers_v = np.sqrt((chi2 / n) / min_dim) if n > 0 and min_dim > 0 else np.nan
        except Exception:
            p, cramers_v = np.nan, np.nan

        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "test": "Chi-square",
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "effect_size": float(cramers_v) if np.isfinite(cramers_v) else np.nan,
                "levels": "|".join(tab.index.astype(str).tolist()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["p_value"].fillna(1.0).to_numpy())
        out = out.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    return out


def subtype_risk_table(assign_df: pd.DataFrame) -> pd.DataFrame:
    if "risk_score" not in assign_df.columns:
        return pd.DataFrame(columns=["comparison", "test", "p_value", "fdr", "effect_size"])

    df = assign_df.copy()
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df = df.dropna(subset=["risk_score"])
    if df.empty:
        return pd.DataFrame(columns=["comparison", "test", "p_value", "fdr", "effect_size"])

    rows: List[Dict[str, object]] = []
    groups = [g["risk_score"].to_numpy(dtype=float) for _, g in df.groupby("subtype")]
    if len(groups) >= 2:
        try:
            stat, p = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            k = len(groups)
            eta2 = (stat - k + 1) / (n - k) if n > k else np.nan
        except Exception:
            p, eta2 = np.nan, np.nan
        rows.append(
            {
                "comparison": "overall_subtype",
                "test": "Kruskal-Wallis",
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "effect_size": float(max(eta2, 0.0) if np.isfinite(eta2) else np.nan),
            }
        )

    uniq = sorted(df["subtype"].unique())
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a = df.loc[df["subtype"] == uniq[i], "risk_score"].to_numpy(dtype=float)
            b = df.loc[df["subtype"] == uniq[j], "risk_score"].to_numpy(dtype=float)
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                effect = abs((2.0 * float(u)) / (len(a) * len(b)) - 1.0)
            except Exception:
                p, effect = np.nan, np.nan
            rows.append(
                {
                    "comparison": f"Subtype_{int(uniq[i])}_vs_{int(uniq[j])}",
                    "test": "Mann-Whitney",
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "effect_size": float(effect) if np.isfinite(effect) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["p_value"].fillna(1.0).to_numpy())
        out = out.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    return out


def plot_consensus_heatmap(consensus: np.ndarray, labels: np.ndarray, out_file: Path, title: str) -> None:
    order = np.argsort(labels)
    mat = consensus[np.ix_(order, order)]
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_km_curve(
    clinical_df: pd.DataFrame,
    patient_ids: Sequence[str],
    labels: np.ndarray,
    endpoint: str,
    out_file: Path,
) -> None:
    endpoint = endpoint.upper()
    t_col = f"{endpoint}_TIME"
    e_col = f"{endpoint}_EVENT"
    if t_col not in clinical_df.columns or e_col not in clinical_df.columns:
        return

    try:
        from lifelines import KaplanMeierFitter
    except Exception:
        return

    clin = clinical_df.reindex(patient_ids)
    t = pd.to_numeric(clin[t_col], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(clin[e_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(e)
    t = t[ok]
    e = e[ok].astype(int)
    g = labels[ok]

    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    for subtype in sorted(np.unique(g)):
        m = g == subtype
        if m.sum() == 0:
            continue
        kmf.fit(t[m], event_observed=e[m], label=f"Subtype {int(subtype)+1} (n={int(m.sum())})")
        kmf.plot(ci_show=False)

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title(f"{endpoint} Kaplan-Meier by subtype")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    consensus_dir = out_dir / "consensus_matrices"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "subtype_discovery.log")

    try:
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")
        rep_path = choose_representation_file(Path(args.representation))
        rep = read_matrix(rep_path)
        logger.info("Representation: %s | shape=%s", rep_path, rep.shape)

        clinical = read_matrix(Path(args.clinical))

        common = sorted(set(rep.index) & set(clinical.index))
        if len(common) < 20:
            raise ValueError("Too few overlapping patients between representation and clinical matrix.")

        rep = rep.loc[common]
        x = rep.to_numpy(dtype=float)

        k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
        if not k_values:
            raise ValueError("No valid k values provided.")

        metrics_rows: List[Dict[str, float]] = []
        assignment_frames: List[pd.DataFrame] = []

        for k in k_values:
            logger.info("Consensus clustering for k=%d", k)
            labels, consensus, stab_mean, stab_std = consensus_clustering(
                x,
                k=k,
                repeats=args.repeats,
                subsample_fraction=args.subsample_fraction,
                seed=args.seed + k,
            )

            sil = float(silhouette_score(x, labels)) if len(np.unique(labels)) > 1 else np.nan
            gap, gap_sk = gap_statistic(x, k=k, n_refs=args.gap_refs, seed=args.seed + 100 + k)
            surv = survival_compare(clinical, common, labels, args.endpoint)

            metrics_rows.append(
                {
                    "k": k,
                    "silhouette": sil,
                    "gap": gap,
                    "gap_sk": gap_sk,
                    "stability_mean_ari": stab_mean,
                    "stability_std_ari": stab_std,
                    "logrank_p": surv["logrank_p"],
                    "events": surv["n_events"],
                }
            )

            assign_df = pd.DataFrame(
                {
                    "PATIENT_ID": common,
                    "k": k,
                    "subtype": labels.astype(int) + 1,
                }
            )
            assignment_frames.append(assign_df)
            assign_df.to_csv(out_dir / f"subtype_assignments_k{k}.csv", index=False)

            np.save(out_dir / f"consensus_matrix_k{k}.npy", consensus)
            pd.DataFrame(consensus, index=common, columns=common).to_csv(consensus_dir / f"consensus_matrix_k{k}.csv")
            plot_consensus_heatmap(consensus, labels, fig_dir / f"consensus_heatmap_k{k}.png", f"Consensus matrix (k={k})")
            plot_km_curve(clinical, common, labels, args.endpoint, fig_dir / f"km_curve_k{k}.png")

        metrics_df = pd.DataFrame(metrics_rows).sort_values("k").reset_index(drop=True)

        # choose best k by combined score
        score_df = metrics_df.copy()
        for col in ["silhouette", "gap", "stability_mean_ari"]:
            v = score_df[col]
            std = v.std(ddof=0)
            score_df[col + "_z"] = (v - v.mean()) / (std if std > 0 else 1.0)

        p = score_df["logrank_p"].replace(0, 1e-300)
        score_df["logrank_score"] = -np.log10(p)
        v = score_df["logrank_score"]
        std = v.std(ddof=0)
        score_df["logrank_score_z"] = (v - v.mean()) / (std if std > 0 else 1.0)

        score_df["combined_score"] = (
            score_df["silhouette_z"]
            + score_df["gap_z"]
            + score_df["stability_mean_ari_z"]
            + score_df["logrank_score_z"]
        )
        score_df = score_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        best_k = int(score_df.iloc[0]["k"])

        metrics_df.to_csv(out_dir / "subtype_model_selection_metrics.csv", index=False)
        score_df.to_csv(out_dir / "subtype_model_selection_scores.csv", index=False)
        metrics_df.to_csv(out_dir / "cluster_selection_summary.csv", index=False)
        metrics_df[["k", "stability_mean_ari", "stability_std_ari"]].to_csv(out_dir / "stability_summary.csv", index=False)

        all_assign = pd.concat(assignment_frames, axis=0, ignore_index=True)
        all_assign.to_csv(out_dir / "subtype_assignments_all_k.csv", index=False)

        final_assign = all_assign[all_assign["k"] == best_k].copy()
        final_assign = final_assign.sort_values("PATIENT_ID")

        if args.risk_scores and Path(args.risk_scores).exists():
            risk = read_matrix(Path(args.risk_scores))
            common2 = sorted(set(final_assign["PATIENT_ID"]) & set(risk.index))
            if common2:
                final_assign = final_assign.set_index("PATIENT_ID").loc[common2].reset_index()
                risk_sub = risk.loc[common2]
                # pick first risk column by default
                risk_col = risk_sub.columns[0]
                q50 = float(risk_sub[risk_col].median())
                final_assign["risk_score"] = risk_sub[risk_col].to_numpy(dtype=float)
                final_assign["risk_group"] = np.where(final_assign["risk_score"] >= q50, "High", "Low")

        final_assign.to_csv(out_dir / "subtype_assignments_final.csv", index=False)

        clinic_summary = clinicopathologic_summary_table(clinical, final_assign[["PATIENT_ID", "subtype"]])
        clinic_summary.to_csv(out_dir / "subtype_clinicopathologic_summary.csv", index=False)

        risk_cmp = subtype_risk_table(final_assign)
        risk_cmp.to_csv(out_dir / "subtype_risk_comparison.csv", index=False)

        latent_xy, method = make_latent_coordinates(rep.loc[final_assign["PATIENT_ID"].tolist()], seed=args.seed)
        latent_xy = latent_xy.merge(final_assign[["PATIENT_ID", "subtype"]], on="PATIENT_ID", how="left")
        if "risk_group" in final_assign.columns:
            latent_xy = latent_xy.merge(final_assign[["PATIENT_ID", "risk_group"]], on="PATIENT_ID", how="left")
        latent_xy = latent_xy.rename(columns={"DIM1": "UMAP1", "DIM2": "UMAP2"})
        latent_xy["embedding_method"] = method
        latent_xy.to_csv(out_dir / "latent_umap_coordinates.csv", index=False)

        with (out_dir / "subtype_selection.json").open("w", encoding="utf-8") as f:
            json.dump({"best_k": best_k, "endpoint": args.endpoint, "embedding_method": method}, f, indent=2)

        # overview plot
        plt.figure(figsize=(8, 5))
        plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o", label="Silhouette")
        plt.plot(metrics_df["k"], metrics_df["gap"], marker="s", label="Gap")
        plt.plot(metrics_df["k"], metrics_df["stability_mean_ari"], marker="^", label="Stability (ARI)")
        plt.xlabel("k")
        plt.ylabel("Metric value")
        plt.title("Subtype model selection metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "subtype_model_selection_metrics.png", dpi=300)
        plt.close()

        # Latent embedding plot with subtype labels.
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=latent_xy, x="UMAP1", y="UMAP2", hue="subtype", s=28, linewidth=0)
        plt.title(f"Latent embedding by subtype ({method.upper()})")
        plt.tight_layout()
        plt.savefig(fig_dir / "latent_embedding_subtype.png", dpi=300)
        plt.close()

        logger.info("Subtype discovery complete. best_k=%d", best_k)
        return 0

    except Exception as exc:
        logger.exception("Subtype discovery failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subtype discovery from integrated representation.")
    parser.add_argument(
        "--representation",
        required=True,
        help="Path to integrated representation (e.g., benchmark/artifacts/best_integrated_representation.parquet).",
    )
    parser.add_argument("--clinical", required=True, help="Path to clinical_matrix.parquet/csv from data prep.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--endpoint", default="OS", choices=["OS", "PFS"])

    parser.add_argument("--k-values", default="2,3,4")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--subsample-fraction", type=float, default=0.8)
    parser.add_argument("--gap-refs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--risk-scores", default=None, help="Optional risk score file from benchmark artifacts.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
