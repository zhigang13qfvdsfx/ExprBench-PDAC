#!/usr/bin/env python3
"""
Prompt 4: Classical PDAC subtype mapping (Moffitt / Bailey / Collisson).

This module maps discovered PDAC subtypes to established expression subtype systems
with a robust score-based assignment and exports agreement/enrichment tables.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    logger = logging.getLogger("pdac_classical_mapping")
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


def get_classical_signatures() -> Dict[str, Dict[str, List[str]]]:
    # Compact, literature-inspired marker sets for robust mapping with missing-gene tolerance.
    return {
        "Moffitt": {
            "Classical": ["GATA6", "HNF1A", "HNF4A", "EPCAM", "KRT19", "MUC1", "KLF5", "MSLN"],
            "Basal_like": ["KRT5", "KRT6A", "KRT14", "S100A2", "TP63", "LAMC2", "ITGA6", "KRT17"],
        },
        "Bailey": {
            "Squamous": ["KRT5", "KRT6A", "KRT14", "TP63", "S100A2", "LAMC2", "ITGA6"],
            "Pancreatic_Progenitor": ["GATA6", "HNF1A", "HNF4A", "PDX1", "FOXA2", "FOXA3", "EPCAM"],
            "ADEX": ["RBPJL", "NR5A2", "CEL", "CPA1", "PRSS1", "REG1A", "AMY2A"],
            "Immunogenic": ["CD3D", "CD3E", "CD79A", "MS4A1", "CXCL13", "HLA-DRA", "LCK"],
        },
        "Collisson": {
            "Classical": ["GATA6", "EPCAM", "KRT19", "MUC1", "HNF1A", "HNF4A"],
            "QM_PDA": ["VIM", "SNAI1", "TWIST1", "ZEB1", "FN1", "CDH2", "COL1A1"],
            "Exocrine_like": ["CEL", "CPA1", "PRSS1", "PNLIP", "CTRB2", "AMY2A"],
        },
    }


def zscore_by_gene(expr: pd.DataFrame) -> pd.DataFrame:
    x = expr.apply(pd.to_numeric, errors="coerce").copy()
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0).replace(0, np.nan)
    z = (x - mu) / sd
    return z.fillna(0.0)


def score_system(z_expr: pd.DataFrame, sigs: Dict[str, List[str]], min_genes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    upper_map = {g.upper(): g for g in z_expr.columns}
    scores: Dict[str, pd.Series] = {}
    overlap_rows: List[Dict[str, object]] = []

    for subtype_name, genes in sigs.items():
        mapped = [upper_map[g.upper()] for g in genes if g.upper() in upper_map]
        overlap_rows.append(
            {
                "classical_subtype": subtype_name,
                "signature_gene_count": int(len(genes)),
                "mapped_gene_count": int(len(mapped)),
                "mapped_gene_ratio": float(len(mapped) / max(len(genes), 1)),
                "mapped_genes": "|".join(mapped),
            }
        )
        if len(mapped) >= min_genes:
            scores[subtype_name] = z_expr[mapped].mean(axis=1)
        else:
            scores[subtype_name] = pd.Series(np.nan, index=z_expr.index)

    score_df = pd.DataFrame(scores, index=z_expr.index)
    overlap_df = pd.DataFrame(overlap_rows)
    return score_df, overlap_df


def survival_overlay_table(
    clinical: pd.DataFrame,
    labels: pd.Series,
    endpoint: str,
) -> pd.DataFrame:
    endpoint = endpoint.upper()
    t_col = f"{endpoint}_TIME"
    e_col = f"{endpoint}_EVENT"
    if t_col not in clinical.columns or e_col not in clinical.columns:
        return pd.DataFrame(columns=["group", "n", "events", "median_time", "logrank_p"])

    idx = sorted(set(clinical.index) & set(labels.index))
    clin = clinical.loc[idx]
    g = labels.loc[idx].astype(str)
    t = pd.to_numeric(clin[t_col], errors="coerce")
    e = pd.to_numeric(clin[e_col], errors="coerce")
    ok = t.notna() & e.notna()
    t = t[ok]
    e = e[ok].astype(int)
    g = g[ok]

    pval = np.nan
    try:
        from lifelines.statistics import multivariate_logrank_test

        pval = float(multivariate_logrank_test(t, g, e).p_value)
    except Exception:
        pval = np.nan

    rows = []
    for cls, sub in pd.DataFrame({"time": t, "event": e, "group": g}).groupby("group"):
        rows.append(
            {
                "group": cls,
                "n": int(sub.shape[0]),
                "events": int(sub["event"].sum()),
                "median_time": float(sub["time"].median()),
                "logrank_p": pval,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "classical_mapping.log")

    try:
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")

        expr = read_matrix(Path(args.expression)).apply(pd.to_numeric, errors="coerce")
        sub = pd.read_csv(args.subtypes)
        if "PATIENT_ID" not in sub.columns or "subtype" not in sub.columns:
            raise ValueError("Subtype assignment file must include PATIENT_ID and subtype columns.")
        sub = sub[["PATIENT_ID", "subtype"]].copy()
        sub["PATIENT_ID"] = sub["PATIENT_ID"].astype(str)

        common = sorted(set(expr.index) & set(sub["PATIENT_ID"]))
        if len(common) < 20:
            raise ValueError("Too few overlapping patients between expression and discovered subtypes.")

        expr = expr.loc[common]
        sub = sub.set_index("PATIENT_ID").loc[common]

        z_expr = zscore_by_gene(expr)
        signatures = get_classical_signatures()

        assign = pd.DataFrame(index=common)
        assign["discovered_subtype"] = sub["subtype"].astype(int)

        overlap_frames: List[pd.DataFrame] = []
        sankey_rows: List[Dict[str, object]] = []
        enrich_rows: List[Dict[str, object]] = []
        survival_rows: List[pd.DataFrame] = []

        clinical = None
        if args.clinical and Path(args.clinical).exists():
            clinical = read_matrix(Path(args.clinical))

        for system, sigs in signatures.items():
            score_df, overlap_df = score_system(z_expr, sigs, min_genes=args.min_genes_per_signature)
            overlap_df.insert(0, "system", system)
            overlap_frames.append(overlap_df)

            for c in score_df.columns:
                assign[f"{system}__score__{c}"] = score_df[c]

            call = score_df.idxmax(axis=1)
            call = call.fillna("Unassigned")
            assign[f"{system}_subtype"] = call

            tab = pd.crosstab(assign["discovered_subtype"], assign[f"{system}_subtype"], dropna=False)
            tab.to_csv(out_dir / f"contingency_{system}.csv")

            for d in tab.index:
                for c in tab.columns:
                    sankey_rows.append(
                        {
                            "system": system,
                            "source": f"Discovered_{int(d)}",
                            "target": f"{system}_{c}",
                            "value": int(tab.loc[d, c]),
                        }
                    )

            # Global enrichment test
            if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                try:
                    chi2, p_global, _, _ = stats.chi2_contingency(tab)
                    n = tab.to_numpy().sum()
                    mind = min(tab.shape[0] - 1, tab.shape[1] - 1)
                    cv = np.sqrt((chi2 / n) / mind) if n > 0 and mind > 0 else np.nan
                except Exception:
                    p_global, cv = np.nan, np.nan
                enrich_rows.append(
                    {
                        "system": system,
                        "discovered_subtype": "ALL",
                        "classical_subtype": "ALL",
                        "count": int(tab.to_numpy().sum()),
                        "odds_ratio": np.nan,
                        "p_value": float(p_global) if np.isfinite(p_global) else np.nan,
                        "effect_size": float(cv) if np.isfinite(cv) else np.nan,
                        "test": "Chi-square",
                    }
                )

            # Cell-level enrichment with Fisher exact
            row_tot = tab.sum(axis=1)
            col_tot = tab.sum(axis=0)
            n_all = int(tab.to_numpy().sum())
            for d in tab.index:
                for c in tab.columns:
                    a = int(tab.loc[d, c])
                    b = int(row_tot.loc[d] - a)
                    cc = int(col_tot.loc[c] - a)
                    d2 = int(n_all - a - b - cc)
                    if min(a, b, cc, d2) < 0:
                        continue
                    try:
                        odds, p = stats.fisher_exact([[a, b], [cc, d2]], alternative="two-sided")
                    except Exception:
                        odds, p = np.nan, np.nan
                    enrich_rows.append(
                        {
                            "system": system,
                            "discovered_subtype": int(d),
                            "classical_subtype": str(c),
                            "count": a,
                            "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                            "p_value": float(p) if np.isfinite(p) else np.nan,
                            "effect_size": np.nan,
                            "test": "FisherExact",
                        }
                    )

            if clinical is not None:
                surv = survival_overlay_table(clinical, assign[f"{system}_subtype"], endpoint=args.endpoint)
                if not surv.empty:
                    surv.insert(0, "system", system)
                    survival_rows.append(surv)

            # Plot contingency heatmap
            plt.figure(figsize=(7, 5))
            sns.heatmap(tab, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{system}: discovered vs classical subtype")
            plt.tight_layout()
            plt.savefig(fig_dir / f"contingency_heatmap_{system}.png", dpi=300)
            plt.close()

        assign = assign.reset_index().rename(columns={"index": "PATIENT_ID"})
        assign.to_csv(out_dir / "classical_subtype_assignments.csv", index=False)

        overlap_df_all = pd.concat(overlap_frames, axis=0, ignore_index=True)
        overlap_df_all.to_csv(out_dir / "signature_overlap_summary.csv", index=False)

        sankey_df = pd.DataFrame(sankey_rows)
        sankey_df.to_csv(out_dir / "sankey_table.csv", index=False)

        enrich_df = pd.DataFrame(enrich_rows)
        if not enrich_df.empty:
            enrich_df["fdr"] = bh_fdr(enrich_df["p_value"].fillna(1.0).to_numpy())
            enrich_df = enrich_df.sort_values(["system", "fdr", "p_value"]).reset_index(drop=True)
        enrich_df.to_csv(out_dir / "enrichment_summary.csv", index=False)

        if survival_rows:
            surv_all = pd.concat(survival_rows, axis=0, ignore_index=True)
        else:
            surv_all = pd.DataFrame(columns=["system", "group", "n", "events", "median_time", "logrank_p"])
        surv_all.to_csv(out_dir / "classical_mapping_survival_overlays.csv", index=False)

        logger.info("Classical subtype mapping complete.")
        return 0

    except Exception as exc:
        logger.exception("Classical subtype mapping failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classical PDAC subtype mapping.")
    p.add_argument("--expression", required=True, help="Expression matrix parquet/csv.")
    p.add_argument("--subtypes", required=True, help="Discovered subtype assignment CSV.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clinical", default=None, help="Optional clinical matrix for survival overlays.")
    p.add_argument("--endpoint", default="OS", choices=["OS", "PFS"])
    p.add_argument("--min-genes-per-signature", type=int, default=3)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
