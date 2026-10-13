#!/usr/bin/env python3
"""
Prompt 4: Biology interpretation by PDAC subtype.

Modules:
- Differential expression
- GSVA/GSEA (GMT-driven)
- Mutation landscape comparison
- Differential methylation
- Methylation-expression correlation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pdac_biology")
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


def parse_gmt(gmt_file: Path) -> Dict[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    with gmt_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]
            if genes:
                gene_sets[name] = genes
    return gene_sets


def built_in_hallmark_sets() -> Dict[str, List[str]]:
    return {
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": ["VIM", "SNAI1", "TWIST1", "ZEB1", "FN1", "COL1A1", "ITGA5"],
        "HALLMARK_KRAS_SIGNALING_UP": ["KRAS", "DUSP6", "ETV4", "ETV5", "SPRY2", "SPRED1", "FOSL1"],
        "HALLMARK_G2M_CHECKPOINT": ["MKI67", "TOP2A", "CCNB1", "CDC20", "AURKA", "BUB1", "CDK1"],
        "HALLMARK_E2F_TARGETS": ["MCM2", "MCM3", "MCM4", "PCNA", "CDK2", "E2F1", "RRM2"],
        "HALLMARK_HYPOXIA": ["HIF1A", "VEGFA", "SLC2A1", "CA9", "LDHA", "ENO1", "PGK1"],
        "HALLMARK_IL6_JAK_STAT3_SIGNALING": ["IL6", "JAK1", "STAT3", "SOCS3", "OSMR", "CEBPB", "BCL3"],
        "HALLMARK_INTERFERON_GAMMA_RESPONSE": ["IFNG", "STAT1", "CXCL9", "CXCL10", "IRF1", "GBP1", "HLA-DRA"],
        "HALLMARK_APOPTOSIS": ["BAX", "BCL2", "CASP3", "CASP8", "FAS", "TP53", "PMAIP1"],
        "HALLMARK_ANGIOGENESIS": ["VEGFA", "KDR", "ANGPT2", "TEK", "FLT1", "PDGFB", "THBS1"],
        "HALLMARK_TGF_BETA_SIGNALING": ["TGFB1", "SMAD2", "SMAD3", "SMAD4", "SERPINE1", "COL1A1", "FN1"],
    }


def differential_table(
    matrix: pd.DataFrame,
    labels: pd.Series,
    subtype: int,
    min_non_na: int = 8,
) -> pd.DataFrame:
    in_group = labels == subtype
    out_group = labels != subtype

    rows = []
    for feat in matrix.columns:
        a = pd.to_numeric(matrix.loc[in_group, feat], errors="coerce").dropna()
        b = pd.to_numeric(matrix.loc[out_group, feat], errors="coerce").dropna()

        if len(a) < min_non_na or len(b) < min_non_na:
            continue

        stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        fc = mean_a - mean_b

        rows.append(
            {
                "feature": feat,
                "subtype": int(subtype),
                "mean_in": mean_a,
                "mean_out": mean_b,
                "log2fc": fc,
                "t_stat": float(stat),
                "p_value": float(p),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr"] = bh_fdr(out["p_value"].to_numpy())
    out = out.sort_values(["fdr", "p_value"], ascending=[True, True]).reset_index(drop=True)
    return out


def volcano_plot(df: pd.DataFrame, title: str, out_file: Path, top_n: int = 15) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["neglog10_p"] = -np.log10(np.clip(plot_df["p_value"], 1e-300, 1.0))
    plot_df["sig"] = (plot_df["fdr"] < 0.05) & (np.abs(plot_df["log2fc"]) > 0.5)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="log2fc", y="neglog10_p", hue="sig", palette={True: "#d62728", False: "#999999"}, s=25)
    plt.title(title)
    plt.xlabel("Difference (in - out)")
    plt.ylabel("-log10(p)")

    top = plot_df.nsmallest(top_n, "fdr")
    for _, row in top.iterrows():
        plt.text(row["log2fc"], row["neglog10_p"], str(row["feature"]), fontsize=7)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def run_gene_set_scoring(expr: pd.DataFrame, gene_sets: Dict[str, List[str]]) -> pd.DataFrame:
    # GSVA-like mean z-score enrichment per sample.
    z = expr.copy()
    z = z.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) if c.std(ddof=0) > 0 else 1.0), axis=0)

    score = {}
    genes_upper = {g.upper(): g for g in z.columns}
    for gs, genes in gene_sets.items():
        mapped = [genes_upper[g.upper()] for g in genes if g.upper() in genes_upper]
        if len(mapped) < 5:
            continue
        score[gs] = z[mapped].mean(axis=1)

    out = pd.DataFrame(score, index=z.index)
    out.index.name = "PATIENT_ID"
    return out


def gene_set_differential(scores: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for subtype in sorted(labels.unique()):
        in_group = labels == subtype
        out_group = labels != subtype
        for gs in scores.columns:
            a = pd.to_numeric(scores.loc[in_group, gs], errors="coerce").dropna()
            b = pd.to_numeric(scores.loc[out_group, gs], errors="coerce").dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "subtype": int(subtype),
                    "gene_set": gs,
                    "mean_in": float(a.mean()),
                    "mean_out": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "t_stat": float(stat),
                    "p_value": float(p),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr"] = bh_fdr(out["p_value"].to_numpy())
    out = out.sort_values(["subtype", "fdr", "p_value"]).reset_index(drop=True)
    return out


def mutation_subtype_comparison(mut: pd.DataFrame, labels: pd.Series, min_mut_count: int = 5) -> pd.DataFrame:
    rows = []
    cand_genes = mut.columns[mut.sum(axis=0) >= min_mut_count]

    for subtype in sorted(labels.unique()):
        in_group = labels == subtype
        out_group = labels != subtype

        for gene in cand_genes:
            a = mut.loc[in_group, gene].astype(int)
            b = mut.loc[out_group, gene].astype(int)

            table = np.array(
                [
                    [int((a > 0).sum()), int((a == 0).sum())],
                    [int((b > 0).sum()), int((b == 0).sum())],
                ]
            )

            try:
                odds, p = stats.fisher_exact(table)
            except Exception:
                odds, p = np.nan, np.nan

            rows.append(
                {
                    "subtype": int(subtype),
                    "gene": gene,
                    "mut_freq_in": float((a > 0).mean()),
                    "mut_freq_out": float((b > 0).mean()),
                    "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr"] = bh_fdr(out["p_value"].fillna(1.0).to_numpy())
    out = out.sort_values(["subtype", "fdr", "p_value"]).reset_index(drop=True)
    return out


def methyl_expression_correlation(
    meth: pd.DataFrame,
    expr: pd.DataFrame,
    probe_gene_map: Optional[pd.DataFrame],
    max_pairs: int,
) -> pd.DataFrame:
    rows = []

    if probe_gene_map is not None and not probe_gene_map.empty:
        if "probe" not in probe_gene_map.columns or "gene" not in probe_gene_map.columns:
            raise ValueError("probe-gene map file must have columns: probe,gene")

        pairs = probe_gene_map[["probe", "gene"]].dropna().drop_duplicates()
        pairs = pairs[pairs["probe"].isin(meth.columns) & pairs["gene"].isin(expr.columns)]
        if len(pairs) > max_pairs:
            pairs = pairs.head(max_pairs)

        for _, row in pairs.iterrows():
            probe = row["probe"]
            gene = row["gene"]
            x = pd.to_numeric(meth[probe], errors="coerce")
            y = pd.to_numeric(expr[gene], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 20:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            rows.append({"probe": probe, "gene": gene, "rho": float(rho), "p_value": float(p), "n": int(mask.sum())})

    else:
        # Fallback: direct feature-name overlap (useful when methylation features are gene-level).
        overlap = [g for g in meth.columns if g in expr.columns]
        overlap = overlap[:max_pairs]
        for g in overlap:
            x = pd.to_numeric(meth[g], errors="coerce")
            y = pd.to_numeric(expr[g], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 20:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            rows.append({"probe": g, "gene": g, "rho": float(rho), "p_value": float(p), "n": int(mask.sum())})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr"] = bh_fdr(out["p_value"].to_numpy())
    out = out.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    return out


def mutation_pairwise_analysis(mut: pd.DataFrame, min_mut_count: int = 8, max_genes: int = 80) -> pd.DataFrame:
    mut_bin = (mut.fillna(0) > 0).astype(int)
    freq = mut_bin.sum(axis=0).sort_values(ascending=False)
    genes = freq[freq >= min_mut_count].head(max_genes).index.tolist()

    rows = []
    for i in range(len(genes)):
        g1 = genes[i]
        v1 = mut_bin[g1].to_numpy(dtype=int)
        for j in range(i + 1, len(genes)):
            g2 = genes[j]
            v2 = mut_bin[g2].to_numpy(dtype=int)
            a = int(((v1 == 1) & (v2 == 1)).sum())
            b = int(((v1 == 1) & (v2 == 0)).sum())
            c = int(((v1 == 0) & (v2 == 1)).sum())
            d = int(((v1 == 0) & (v2 == 0)).sum())
            try:
                odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except Exception:
                odds, p = np.nan, np.nan
            relation = "co_occurrence"
            if np.isfinite(odds) and odds < 1:
                relation = "mutual_exclusivity"
            rows.append(
                {
                    "gene_a": g1,
                    "gene_b": g2,
                    "co_mutated_n": a,
                    "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "relation": relation,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr"] = bh_fdr(out["p_value"].fillna(1.0).to_numpy())
    out = out.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    return out


def network_module_analysis(
    expr: pd.DataFrame,
    marker_genes: List[str],
    corr_threshold: float = 0.35,
    max_genes: int = 220,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expr_u = expr.loc[:, ~expr.columns.duplicated()].copy()
    genes = [g for g in marker_genes if g in expr_u.columns]
    genes = list(dict.fromkeys(genes))
    genes = genes[:max_genes]
    if len(genes) < 5:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = expr_u[genes].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman", min_periods=max(8, int(0.4 * x.shape[0])))

    edge_rows = []
    for i in range(len(genes)):
        g1 = genes[i]
        for j in range(i + 1, len(genes)):
            g2 = genes[j]
            rho = corr.loc[g1, g2]
            if pd.isna(rho):
                continue
            if abs(float(rho)) < corr_threshold:
                continue
            edge_rows.append({"source": g1, "target": g2, "weight": float(rho), "abs_weight": float(abs(rho))})

    if not edge_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    edges = pd.DataFrame(edge_rows).sort_values("abs_weight", ascending=False).reset_index(drop=True)
    g = nx.Graph()
    for _, r in edges.iterrows():
        g.add_edge(r["source"], r["target"], weight=float(r["abs_weight"]))

    communities = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    mod_map: Dict[str, int] = {}
    for idx, comm in enumerate(communities, start=1):
        for node in comm:
            mod_map[str(node)] = idx

    degree = dict(g.degree())
    modules = pd.DataFrame(
        {
            "gene": list(degree.keys()),
            "module_id": [int(mod_map.get(k, -1)) for k in degree.keys()],
            "degree": [int(v) for v in degree.values()],
        }
    ).sort_values(["module_id", "degree"], ascending=[True, False])

    hub_rows = []
    for mid, sub in modules.groupby("module_id"):
        top = sub.sort_values("degree", ascending=False).head(10)
        for _, r in top.iterrows():
            hub_rows.append({"module_id": int(mid), "gene": r["gene"], "degree": int(r["degree"])})
    hubs = pd.DataFrame(hub_rows)

    return edges, modules.reset_index(drop=True), hubs


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "biology_interpretation.log")

    try:
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")
        expr = read_matrix(Path(args.expression)).apply(pd.to_numeric, errors="coerce")
        meth = read_matrix(Path(args.methylation)).apply(pd.to_numeric, errors="coerce")
        mut = read_matrix(Path(args.mutation)).apply(pd.to_numeric, errors="coerce").fillna(0)

        sub = pd.read_csv(args.subtypes)
        if "PATIENT_ID" not in sub.columns or "subtype" not in sub.columns:
            raise ValueError("Subtype assignment file must contain PATIENT_ID and subtype columns.")
        labels = sub.set_index("PATIENT_ID")["subtype"].astype(int)

        common = sorted(set(expr.index) & set(meth.index) & set(mut.index) & set(labels.index))
        if len(common) < 20:
            raise ValueError("Too few overlapping patients across expression/methylation/mutation/subtype files.")

        expr = expr.loc[common]
        meth = meth.loc[common]
        mut = mut.loc[common]
        labels = labels.loc[common]

        # Differential expression
        de_tables = []
        for st in sorted(labels.unique()):
            de = differential_table(expr, labels, st, min_non_na=args.min_group_samples)
            de.to_csv(out_dir / f"de_subtype_{st}.csv", index=False)
            volcano_plot(de, f"DE subtype {st} vs rest", fig_dir / f"de_volcano_subtype_{st}.png")
            de_tables.append(de)

        de_all = pd.concat(de_tables, axis=0, ignore_index=True) if de_tables else pd.DataFrame()
        de_all.to_csv(out_dir / "de_all_subtypes.csv", index=False)

        # Marker table from DE results
        marker_rows = []
        if not de_all.empty:
            for st, sub_de in de_all.groupby("subtype"):
                up = sub_de.sort_values(["fdr", "log2fc"], ascending=[True, False]).head(40).copy()
                up["direction"] = "up"
                down = sub_de.sort_values(["fdr", "log2fc"], ascending=[True, True]).head(40).copy()
                down["direction"] = "down"
                marker_rows.extend([up, down])
        marker_df = pd.concat(marker_rows, axis=0, ignore_index=True) if marker_rows else pd.DataFrame()
        marker_df.to_csv(out_dir / "marker_gene_table.csv", index=False)
        de_all.to_csv(out_dir / "deg_results.csv", index=False)

        # GSVA/GSEA-like scoring from GMT or built-in hallmark proxies
        gs = {}
        if args.gmt and Path(args.gmt).exists():
            gs = parse_gmt(Path(args.gmt))
        if not gs:
            gs = built_in_hallmark_sets()

        scores = run_gene_set_scoring(expr, gs)
        scores.to_csv(out_dir / "gsva_like_scores.csv")
        gs_diff = gene_set_differential(scores, labels) if not scores.empty else pd.DataFrame()
        gs_diff.to_csv(out_dir / "gsva_summary.csv", index=False)
        hall = gs_diff[gs_diff["gene_set"].astype(str).str.contains("HALLMARK", case=False, na=False)].copy() if not gs_diff.empty else pd.DataFrame()
        if hall.empty and not gs_diff.empty:
            hall = gs_diff.copy()
        hall.to_csv(out_dir / "hallmark_summary.csv", index=False)

        if not gs_diff.empty:
            top = gs_diff[gs_diff["fdr"] < 0.05].head(40)
            if not top.empty:
                pivot = top.pivot_table(index="gene_set", columns="subtype", values="delta", fill_value=0)
                plt.figure(figsize=(10, max(5, 0.25 * len(pivot))))
                sns.heatmap(pivot, cmap="coolwarm", center=0)
                plt.title("Top differential gene-set scores")
                plt.tight_layout()
                plt.savefig(fig_dir / "gene_set_heatmap.png", dpi=300)
                plt.close()

        # Optional true pre-ranked GSEA per subtype if gseapy is installed.
        if args.gmt and Path(args.gmt).exists() and not de_all.empty:
            try:
                import gseapy as gp

                gsea_rows = []
                for st in sorted(labels.unique()):
                    de = de_all[de_all["subtype"] == st].copy()
                    rnk = de[["feature", "t_stat"]].dropna().sort_values("t_stat", ascending=False)
                    if rnk.shape[0] < 20:
                        continue
                    pre = gp.prerank(
                        rnk=rnk,
                        gene_sets=str(args.gmt),
                        min_size=10,
                        max_size=500,
                        permutation_num=500,
                        outdir=None,
                        seed=args.seed,
                        verbose=False,
                    )
                    res = pre.res2d.copy()
                    res["subtype"] = st
                    gsea_rows.append(res.reset_index().rename(columns={"index": "term"}))

                if gsea_rows:
                    gsea_all = pd.concat(gsea_rows, axis=0, ignore_index=True)
                    gsea_all.to_csv(out_dir / "gsea_prerank_results.csv", index=False)
            except Exception as exc:
                logger.warning("gseapy prerank was skipped: %s", exc)

        # Mutation landscape comparison
        mut_cmp = mutation_subtype_comparison(mut, labels, min_mut_count=args.min_mut_count)
        mut_cmp.to_csv(out_dir / "mutation_subtype_comparison.csv", index=False)
        mut_cmp.to_csv(out_dir / "mutation_enrichment_results.csv", index=False)
        mut_pair = mutation_pairwise_analysis(mut, min_mut_count=max(5, args.min_mut_count), max_genes=70)
        mut_pair.to_csv(out_dir / "mutation_pairwise_results.csv", index=False)

        if not mut_cmp.empty:
            top_mut = mut_cmp.sort_values("fdr").head(30)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top_mut, x="gene", y="odds_ratio", hue="subtype")
            plt.xticks(rotation=75)
            plt.title("Top subtype-enriched mutations")
            plt.tight_layout()
            plt.savefig(fig_dir / "mutation_subtype_barplot.png", dpi=300)
            plt.close()

        # Differential methylation
        meth_tables = []
        for st in sorted(labels.unique()):
            dm = differential_table(meth, labels, st, min_non_na=args.min_group_samples)
            dm.to_csv(out_dir / f"dm_subtype_{st}.csv", index=False)
            volcano_plot(dm, f"Differential methylation subtype {st} vs rest", fig_dir / f"dm_volcano_subtype_{st}.png")
            meth_tables.append(dm)

        dm_all = pd.concat(meth_tables, axis=0, ignore_index=True) if meth_tables else pd.DataFrame()
        dm_all.to_csv(out_dir / "dm_all_subtypes.csv", index=False)
        dm_all.to_csv(out_dir / "methylation_diff_results.csv", index=False)

        # Methylation-expression correlation
        pg_map = None
        if args.probe_gene_map and Path(args.probe_gene_map).exists():
            pg_map = pd.read_csv(args.probe_gene_map)
            pg_map.columns = [c.strip().lower() for c in pg_map.columns]

        mec = methyl_expression_correlation(meth, expr, pg_map, max_pairs=args.max_corr_pairs)
        mec.to_csv(out_dir / "methylation_expression_correlations.csv", index=False)
        mec.to_csv(out_dir / "methylation_expression_coupling.csv", index=False)

        if not mec.empty:
            top_corr = mec.nsmallest(20, "fdr")
            plt.figure(figsize=(8, 5))
            sns.barplot(data=top_corr, x="gene", y="rho")
            plt.xticks(rotation=75)
            plt.title("Top methylation-expression correlations")
            plt.tight_layout()
            plt.savefig(fig_dir / "methylation_expression_top_correlations.png", dpi=300)
            plt.close()

        if not mut_pair.empty:
            pair_top = mut_pair[mut_pair["fdr"] < 0.1].head(40)
            if not pair_top.empty:
                pair_top = pair_top.copy()
                pair_top["pair"] = pair_top["gene_a"] + " / " + pair_top["gene_b"]
                plt.figure(figsize=(10, max(5, 0.22 * len(pair_top))))
                sns.barplot(data=pair_top, x="odds_ratio", y="pair", hue="relation")
                plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
                plt.title("Significant mutation co-occurrence/exclusivity")
                plt.tight_layout()
                plt.savefig(fig_dir / "mutation_pairwise_top.png", dpi=300)
                plt.close()

        # Correlation network/module analysis from subtype marker genes.
        marker_for_network = []
        if not marker_df.empty:
            marker_for_network = marker_df.sort_values("fdr").head(240)["feature"].astype(str).tolist()
        edges, modules, hubs = network_module_analysis(expr, marker_for_network, corr_threshold=args.network_corr_threshold)
        edges.to_csv(out_dir / "network_edges.csv", index=False)
        modules.to_csv(out_dir / "network_modules.csv", index=False)
        hubs.to_csv(out_dir / "hub_genes.csv", index=False)

        if not hubs.empty:
            top_hubs = hubs.sort_values(["module_id", "degree"], ascending=[True, False]).head(40)
            plt.figure(figsize=(9, max(5, 0.22 * len(top_hubs))))
            sns.barplot(data=top_hubs, x="degree", y="gene", hue="module_id")
            plt.title("Top hub genes by network module")
            plt.tight_layout()
            plt.savefig(fig_dir / "network_hub_genes.png", dpi=300)
            plt.close()

        logger.info("Biology interpretation complete.")
        return 0

    except Exception as exc:
        logger.exception("Biology interpretation failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDAC subtype biology interpretation.")
    parser.add_argument("--expression", required=True, help="Path to expression matrix parquet/csv.")
    parser.add_argument("--methylation", required=True, help="Path to methylation matrix parquet/csv.")
    parser.add_argument("--mutation", required=True, help="Path to mutation matrix parquet/csv.")
    parser.add_argument("--subtypes", required=True, help="Path to subtype_assignments_final.csv.")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--gmt", default=None, help="Optional gene-set GMT file for GSVA/GSEA modules.")
    parser.add_argument("--probe-gene-map", default=None, help="Optional CSV with columns probe,gene for correlation.")

    parser.add_argument("--min-group-samples", type=int, default=8)
    parser.add_argument("--min-mut-count", type=int, default=5)
    parser.add_argument("--max-corr-pairs", type=int, default=50000)
    parser.add_argument("--network-corr-threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
