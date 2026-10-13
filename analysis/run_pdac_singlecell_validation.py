#!/usr/bin/env python3
"""
Prompt 7: Single-cell validation for bulk-derived PDAC signatures.

Pipeline:
- load scRNA-seq data
- QC / normalization / clustering / UMAP
- marker-based cell annotation
- bulk subtype-signature scoring in single-cell space
- pseudo-bulk concordance (optional)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pdac_sc_validation")
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


def require_scanpy():
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc
    return sc


def _read_annotations(path: Path) -> pd.DataFrame:
    if path.suffix.lower().endswith(".gz"):
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        idx_col = "Unnamed: 0"
    else:
        idx_col = df.columns[0]
    df[idx_col] = df[idx_col].astype(str)
    df = df.set_index(idx_col)
    return df


def load_geo_integrated_gem(
    gem_path: Path,
    selected_genes: Optional[set],
    annotations_path: Optional[Path],
    max_cells: int,
    random_seed: int,
):
    import anndata as ad

    ann = None
    if annotations_path is not None and annotations_path.exists():
        ann = _read_annotations(annotations_path)

    opener = gzip.open if gem_path.suffix.lower().endswith(".gz") else open
    with opener(gem_path, "rt", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cell_ids = [str(x).strip().strip('"') for x in header[1:]]

        keep_cells = cell_ids
        if ann is not None:
            keep_set = set(ann.index.astype(str))
            keep_cells = [c for c in cell_ids if c in keep_set]
        if max_cells > 0 and len(keep_cells) > max_cells:
            rng = random.Random(random_seed)
            keep_cells = rng.sample(keep_cells, k=max_cells)

        keep_set = set(keep_cells)
        keep_idx = [i for i, c in enumerate(cell_ids) if c in keep_set]
        keep_cells = [cell_ids[i] for i in keep_idx]
        keep_lookup = set([g.upper() for g in selected_genes]) if selected_genes else None

        genes: List[str] = []
        rows: List[np.ndarray] = []
        for row in reader:
            if not row:
                continue
            gene = str(row[0]).strip().strip('"')
            if keep_lookup is not None and gene.upper() not in keep_lookup:
                continue
            vals = []
            for j in keep_idx:
                try:
                    vals.append(float(row[j + 1]))
                except Exception:
                    vals.append(0.0)
            arr = np.asarray(vals, dtype=np.float32)
            if np.nanmax(arr) <= 0:
                continue
            genes.append(gene)
            rows.append(arr)

    if not rows:
        raise ValueError("No genes loaded from integrated GEM under current filtering rules.")

    x = np.vstack(rows).T  # cells x genes
    adata = ad.AnnData(X=x)
    adata.obs_names = pd.Index(keep_cells, dtype=str)
    adata.var_names = pd.Index(genes, dtype=str)
    if ann is not None:
        obs = ann.reindex(adata.obs_names).copy()
        for c in obs.columns:
            adata.obs[c] = obs[c].astype(str).fillna("NA")
    return adata


def load_sc_data(
    sc,
    path: Path,
    selected_genes: Optional[set] = None,
    annotations_path: Optional[Path] = None,
    max_cells: int = 0,
    random_seed: int = 42,
):
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    elif str(path).lower().endswith(".csv.gz") or path.suffix.lower() == ".csv":
        adata = load_geo_integrated_gem(
            gem_path=path,
            selected_genes=selected_genes,
            annotations_path=annotations_path,
            max_cells=max_cells,
            random_seed=random_seed,
        )
    elif path.suffix.lower() == ".h5ad":
        adata = sc.read_h5ad(path)
    elif path.suffix.lower() in {".h5", ".h5mu"}:
        adata = sc.read(path)
    else:
        raise ValueError("Unsupported single-cell input format. Use .h5ad or 10x mtx directory.")

    return adata


def parse_celltype_markers(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None or not path.exists():
        return {}

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items()}

    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)
        df.columns = [c.strip().lower() for c in df.columns]
        if "cell_type" in df.columns and "genes" in df.columns:
            out = {}
            for cell_type, sub in df.groupby("cell_type"):
                genes = []
                for val in sub["genes"].astype(str):
                    genes.extend([g.strip() for g in val.replace(";", ",").split(",") if g.strip()])
                out[str(cell_type)] = sorted(set(genes))
            return out

    return {}


def parse_subtype_signatures(marker_file: Optional[Path], top_n: int, fdr_cutoff: float) -> Dict[str, List[str]]:
    if marker_file is None or not marker_file.exists():
        return {}

    df = pd.read_csv(marker_file)
    if "subtype" not in df.columns:
        return {}

    feature_col = "feature" if "feature" in df.columns else ("gene" if "gene" in df.columns else None)
    if feature_col is None:
        return {}

    if "fdr" in df.columns:
        df = df[df["fdr"] <= fdr_cutoff]
    if "log2fc" in df.columns:
        df = df.sort_values(["subtype", "log2fc"], ascending=[True, False])

    sig = {}
    for st, sub in df.groupby("subtype"):
        genes = sub[feature_col].astype(str).head(top_n).tolist()
        sig[f"Subtype_{st}"] = genes
    return sig


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "singlecell_validation.log")

    try:
        if plt is None:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")
        sc = require_scanpy()

        # Parse marker resources before loading large expression matrices so we can
        # restrict integrated-GEM ingestion to informative genes.
        marker_dict = parse_celltype_markers(Path(args.celltype_markers) if args.celltype_markers else None)
        subtype_sigs = parse_subtype_signatures(
            Path(args.subtype_markers) if args.subtype_markers else None,
            top_n=args.subtype_top_n,
            fdr_cutoff=args.subtype_marker_fdr,
        )
        selected_genes = set()
        for genes in marker_dict.values():
            selected_genes.update([str(g) for g in genes])
        for genes in subtype_sigs.values():
            selected_genes.update([str(g) for g in genes])

        is_integrated_gem = str(args.sc_data).lower().endswith(".csv.gz") or str(args.sc_data).lower().endswith(".csv")
        adata = load_sc_data(
            sc,
            Path(args.sc_data),
            selected_genes=selected_genes if selected_genes else None,
            annotations_path=Path(args.sc_annotations) if args.sc_annotations else None,
            max_cells=args.max_cells,
            random_seed=args.seed,
        )
        logger.info("Loaded single-cell data: %s", adata)

        # Basic QC
        adata.var_names_make_unique()
        if not is_integrated_gem:
            adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
            adata = adata[adata.obs["n_genes_by_counts"] >= args.min_genes, :].copy()
            adata = adata[adata.obs["pct_counts_mt"] <= args.max_mito, :].copy()
            adata = adata[adata.obs["total_counts"] >= args.min_counts, :].copy()
        else:
            if {"nFeature_RNA", "nCount_RNA", "percent.mt"}.issubset(set(adata.obs.columns)):
                nfeat = pd.to_numeric(adata.obs["nFeature_RNA"], errors="coerce")
                ncount = pd.to_numeric(adata.obs["nCount_RNA"], errors="coerce")
                mt = pd.to_numeric(adata.obs["percent.mt"], errors="coerce")
                keep = (nfeat >= args.min_genes) & (ncount >= args.min_counts) & (mt <= args.max_mito)
                if keep.sum() > 200:
                    adata = adata[keep.to_numpy(), :].copy()

        logger.info("After QC: n_cells=%d, n_genes=%d", adata.n_obs, adata.n_vars)

        # Standard Scanpy pipeline
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata.copy()

        hvg_n = min(args.hvg, max(50, adata.n_vars))
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg_n, flavor="seurat", subset=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs)
        sc.tl.umap(adata)
        try:
            sc.tl.leiden(adata, resolution=args.resolution, key_added="leiden", flavor="igraph", directed=False)
        except Exception:
            from sklearn.cluster import KMeans

            emb = adata.obsm["X_pca"][:, : min(args.n_pcs, adata.obsm["X_pca"].shape[1])]
            k = max(3, min(8, emb.shape[0] // 1000 + 3))
            km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
            adata.obs["leiden"] = pd.Categorical(km.fit_predict(emb).astype(str))

        # Marker-based cell annotation (optional). If provided annotations include a
        # "major" label, use it as primary cell-type assignment.
        if "major" in adata.obs.columns:
            adata.obs["cell_type_auto"] = pd.Categorical(adata.obs["major"].astype(str))
        elif marker_dict:
            for cell_type, genes in marker_dict.items():
                g = [x for x in genes if x in adata.var_names]
                if len(g) < 3:
                    continue
                score_name = f"score_{cell_type}"
                sc.tl.score_genes(adata, gene_list=g, score_name=score_name, use_raw=True)

            score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
            if score_cols:
                score_mat = adata.obs[score_cols].to_numpy(dtype=float)
                best_idx = np.argmax(score_mat, axis=1)
                best_type = [score_cols[i].replace("score_", "") for i in best_idx]
                adata.obs["cell_type_auto"] = pd.Categorical(best_type)
        else:
            adata.obs["cell_type_auto"] = pd.Categorical(np.repeat("Unassigned", adata.n_obs))

        # Bulk-derived subtype signatures
        for sig_name, genes in subtype_sigs.items():
            g = [x for x in genes if x in adata.var_names]
            if len(g) < 3:
                continue
            sc.tl.score_genes(adata, gene_list=g, score_name=f"bulk_{sig_name}", use_raw=True)

        # UMAP plots
        sc.pl.umap(adata, color=["leiden"], show=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "umap_leiden.png", dpi=300)
        plt.close()

        sc.pl.umap(adata, color=["cell_type_auto"], show=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "umap_celltype_auto.png", dpi=300)
        plt.close()

        bulk_score_cols = [c for c in adata.obs.columns if c.startswith("bulk_Subtype_")]
        for c in bulk_score_cols:
            sc.pl.umap(adata, color=[c], show=False, color_map="viridis")
            plt.tight_layout()
            plt.savefig(fig_dir / f"umap_{c}.png", dpi=300)
            plt.close()

        # Optional marker dotplot
        if marker_dict:
            marker_genes = sorted({g for genes in marker_dict.values() for g in genes if g in adata.var_names})
            if marker_genes:
                sc.pl.dotplot(adata, marker_genes[: min(60, len(marker_genes))], groupby="cell_type_auto", show=False)
                plt.tight_layout()
                plt.savefig(fig_dir / "dotplot_marker_genes_by_celltype.png", dpi=300)
                plt.close()

        # Prepare expression frame for summary tables.
        expr_mat = to_dense(adata.raw.X if adata.raw is not None else adata.X)
        expr_df = pd.DataFrame(expr_mat, index=adata.obs_names, columns=adata.raw.var_names if adata.raw is not None else adata.var_names)

        # Signature score export at single-cell level.
        score_cols = [c for c in adata.obs.columns if c.startswith("bulk_Subtype_")]
        sig_cell = adata.obs[score_cols].copy() if score_cols else pd.DataFrame(index=adata.obs_names)
        sig_cell.insert(0, "cell_id", adata.obs_names.astype(str))
        sig_cell.insert(1, "cell_type_auto", adata.obs["cell_type_auto"].astype(str).to_numpy())
        sig_cell.to_csv(out_dir / "subtype_signature_scores_singlecell.csv", index=False)

        # Key-gene by cell-type summary.
        key_genes = sorted({g for genes in subtype_sigs.values() for g in genes if g in expr_df.columns})
        if key_genes:
            key_expr = expr_df[key_genes].join(adata.obs[["cell_type_auto"]])
            key_mean = key_expr.groupby("cell_type_auto").mean(numeric_only=True).T
            key_mean = key_mean.reset_index().rename(columns={"index": "gene"})
        else:
            key_mean = pd.DataFrame(columns=["gene"])
        key_mean.to_csv(out_dir / "key_gene_expression_by_celltype.csv", index=False)

        # Pseudo-bulk expression and concordance.
        concord_rows = []
        if args.sample_id_col in adata.obs.columns:
            adata.obs[args.sample_id_col] = adata.obs[args.sample_id_col].astype(str)
            pseudobulk = expr_df.groupby(adata.obs[args.sample_id_col]).mean()
            pseudobulk.index.name = "sample_id"
            pseudobulk.to_csv(out_dir / "pseudobulk_expression.csv")
        else:
            pseudobulk = expr_df.groupby(adata.obs["cell_type_auto"].astype(str)).mean()
            pseudobulk.index.name = "cell_type_auto"
            pseudobulk.to_csv(out_dir / "pseudobulk_expression.csv")

        if args.bulk_expression and Path(args.bulk_expression).exists() and args.sample_id_col in adata.obs.columns:
            bulk = pd.read_parquet(args.bulk_expression) if str(args.bulk_expression).lower().endswith(".parquet") else pd.read_csv(args.bulk_expression, index_col=0)
            if "PATIENT_ID" in bulk.columns and (bulk.index.name is None or bulk.index.name.upper() != "PATIENT_ID"):
                bulk = bulk.set_index("PATIENT_ID")

            sample_means = pseudobulk.copy()
            common_samples = sorted(set(sample_means.index.astype(str)) & set(bulk.index.astype(str)))
            if common_samples:
                sample_means = sample_means.loc[common_samples]
                bulk = bulk.loc[common_samples]

                for sig_name, genes in subtype_sigs.items():
                    g = [x for x in genes if x in sample_means.columns and x in bulk.columns]
                    if len(g) < 3:
                        continue
                    sc_score = sample_means[g].mean(axis=1)
                    bulk_score = bulk[g].mean(axis=1)
                    rho, p = spearmanr(sc_score, bulk_score)
                    concord_rows.append(
                        {
                            "signature": sig_name,
                            "n_samples": len(common_samples),
                            "spearman_rho": float(rho),
                            "p_value": float(p),
                        }
                    )

        concord_df = pd.DataFrame(concord_rows)
        concord_df.to_csv(out_dir / "pseudobulk_bulk_concordance.csv", index=False)
        concord_df.to_csv(out_dir / "pseudobulk_concordance.csv", index=False)

        # Export processed objects and metadata
        obs_export = adata.obs.copy()
        obs_export.insert(0, "cell_id", adata.obs_names.astype(str))
        obs_export.to_csv(out_dir / "singlecell_obs_annotations.csv")
        obs_export.to_csv(out_dir / "sc_cell_annotations.csv", index=False)
        adata.var.to_csv(out_dir / "singlecell_var_annotations.csv")
        adata.write(out_dir / "singlecell_processed.h5ad")

        summary = {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_clusters": int(adata.obs["leiden"].nunique()),
            "n_auto_celltypes": int(adata.obs["cell_type_auto"].nunique()),
            "n_bulk_signatures_scored": len(bulk_score_cols),
        }
        with (out_dir / "singlecell_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Single-cell validation complete.")
        return 0

    except Exception as exc:
        logger.exception("Single-cell validation failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDAC single-cell validation workflow.")
    p.add_argument("--sc-data", required=True, help="Path to .h5ad file or 10x matrix directory.")
    p.add_argument("--sc-annotations", default=None, help="Optional annotation table for integrated-GEM input.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--celltype-markers", default=None, help="Optional JSON/CSV marker file for cell annotation.")
    p.add_argument("--subtype-markers", default=None, help="Optional DE marker table from Prompt 4.")

    p.add_argument("--bulk-expression", default=None, help="Optional bulk expression matrix for pseudo-bulk concordance.")
    p.add_argument("--sample-id-col", default="sample_id", help="obs column containing sample IDs.")

    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--min-counts", type=int, default=500)
    p.add_argument("--max-mito", type=float, default=20.0)
    p.add_argument("--hvg", type=int, default=3000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=0.8)

    p.add_argument("--subtype-top-n", type=int, default=100)
    p.add_argument("--subtype-marker-fdr", type=float, default=0.05)
    p.add_argument("--max-cells", type=int, default=12000, help="Maximum cells to load from integrated-GEM inputs.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
