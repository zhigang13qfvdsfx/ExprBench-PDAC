#!/usr/bin/env python
"""Permutation/block contribution for the frozen PDAC survival model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_benchmark_import() -> Any:
    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import run_pdac_benchmark as bench

    # Older artifacts were pickled while run_pdac_benchmark.py was executed as
    # __main__. Bind those symbols so joblib can unpickle them reproducibly.
    main_mod = sys.modules.get("__main__")
    for name in [
        "DataBundle",
        "ClinicalTransformer",
        "OmicsTransformer",
        "MutationTransformer",
        "FoldFeatures",
        "FeatureAssembler",
        "SimpleMLPSurvival",
        "GatedFusionSurvival",
    ]:
        if hasattr(bench, name):
            setattr(main_mod, name, getattr(bench, name))
    return bench


def build_logger() -> logging.Logger:
    logger = logging.getLogger("pdac_feature_contribution")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def model_artifact_name(model_name: str) -> str:
    mapping = {
        "Clinical-Cox": "clinical_cox.joblib",
        "Expr-Penalized-Cox": "expr_penalized_cox.joblib",
        "LateFusion-Cox": "latefusion_cox.joblib",
    }
    if model_name not in mapping:
        raise ValueError(f"Feature contribution currently supports frozen Cox artifacts only, got {model_name!r}.")
    return mapping[model_name]


def feature_matrix_and_blocks(payload: Dict[str, Any], bundle: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    assembler = payload["assembler"]
    feats = assembler.transform(bundle.expr, bundle.meth, bundle.mut, bundle.clin)
    block = payload.get("feature_block")

    if block == "clinical":
        names = clinical_feature_names(assembler)
        return feats.clinical, {"clinical": np.arange(feats.clinical.shape[1])}, names

    if block == "expr_scaled":
        names = [f"expr::{c}" for c in assembler.expr_tf.selected_cols]
        return feats.expr_scaled, {"expression": np.arange(feats.expr_scaled.shape[1])}, names

    if block == "fused":
        sizes = {
            "clinical": feats.clinical.shape[1],
            "expression_pca": feats.expr_pca.shape[1],
            "methylation_pca": feats.meth_pca.shape[1],
            "mutation_top": feats.mut_top.shape[1],
        }
        start = 0
        blocks: Dict[str, np.ndarray] = {}
        for key, n in sizes.items():
            blocks[key] = np.arange(start, start + n)
            start += n
        names = (
            clinical_feature_names(assembler)
            + [f"expr_pc{i + 1}" for i in range(feats.expr_pca.shape[1])]
            + [f"meth_pc{i + 1}" for i in range(feats.meth_pca.shape[1])]
            + [f"mut::{g}" for g in assembler.mut_tf.selected_cols]
        )
        return feats.fused, blocks, names

    raise ValueError(f"Unsupported feature block {block!r}.")


def clinical_feature_names(assembler: Any) -> List[str]:
    tf = assembler.clinical_tf
    names = list(getattr(tf, "numeric_cols", [])) + list(getattr(tf, "dummy_cols", []))
    return [f"clinical::{x}" for x in names]


def normalize_feature_names(names: List[str], n_features: int) -> List[str]:
    if len(names) == n_features:
        return names
    if len(names) > n_features:
        return names[:n_features]
    extra = [f"feature_{i + 1}" for i in range(len(names), n_features)]
    return names + extra


def predict_risk(payload: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    model = payload["model"]
    alpha = payload.get("alpha")
    if alpha is None:
        return np.asarray(model.predict(x), dtype=float)
    return np.asarray(model.predict(x, alpha=float(alpha)), dtype=float)


def c_index(event: Sequence[float], time: Sequence[float], risk: Sequence[float], bench: Any) -> float:
    sksurv = bench.require_sksurv()
    out = sksurv["concordance_index_censored"](
        np.asarray(event).astype(bool),
        np.asarray(time).astype(float),
        np.asarray(risk).astype(float),
    )
    return float(out[0])


def cox_coefficients(payload: Dict[str, Any], n_features: int) -> np.ndarray:
    model = payload["model"]
    alpha = payload.get("alpha")
    coef = np.asarray(getattr(model, "coef_", np.zeros(n_features)), dtype=float)
    if coef.ndim == 2:
        alphas = np.asarray(getattr(model, "alphas_", []), dtype=float)
        if alpha is not None and alphas.size == coef.shape[1]:
            idx = int(np.argmin(np.abs(alphas - float(alpha))))
        else:
            idx = 0
        coef = coef[:, idx]
    if coef.size != n_features:
        coef = np.resize(coef, n_features)
    return coef


def run(args: argparse.Namespace) -> None:
    logger = build_logger()
    bench = setup_benchmark_import()

    benchmark_dir = Path(args.benchmark_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    selection_path = benchmark_dir / "artifacts" / "model_selection.json"
    if args.model == "auto":
        with selection_path.open("r", encoding="utf-8") as f:
            model_name = json.load(f).get("best_main_model", "LateFusion-Cox")
    else:
        model_name = args.model

    logger.info("Computing contribution for %s", model_name)
    bundle = bench.load_prepared_data(input_dir, args.endpoint, logger)
    artifact = benchmark_dir / "artifacts" / model_artifact_name(model_name)
    payload = joblib.load(artifact)
    x, blocks, feature_names = feature_matrix_and_blocks(payload, bundle)
    feature_names = normalize_feature_names(feature_names, x.shape[1])

    baseline_risk = predict_risk(payload, x)
    baseline_c = c_index(bundle.event, bundle.time, baseline_risk, bench)
    rng = np.random.default_rng(args.seed)

    rows = []
    for block_name, cols in blocks.items():
        vals = []
        for _ in range(args.repeats):
            x_perm = x.copy()
            order = rng.permutation(x_perm.shape[0])
            x_perm[:, cols] = x_perm[order][:, cols]
            vals.append(c_index(bundle.event, bundle.time, predict_risk(payload, x_perm), bench))
        vals_arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "model": model_name,
                "method": "block_permutation_cindex_drop",
                "block": block_name,
                "n_features": int(len(cols)),
                "baseline_cindex": baseline_c,
                "permuted_cindex_mean": float(np.nanmean(vals_arr)),
                "permuted_cindex_std": float(np.nanstd(vals_arr, ddof=1)) if vals_arr.size > 1 else 0.0,
                "cindex_drop": float(baseline_c - np.nanmean(vals_arr)),
                "permutation_repeats": int(args.repeats),
            }
        )
    block_df = pd.DataFrame(rows).sort_values("cindex_drop", ascending=False)
    block_path = out_dir / "feature_contribution_summary.csv"
    block_df.to_csv(block_path, index=False)

    coef = cox_coefficients(payload, x.shape[1])
    mean_abs = np.nanmean(np.abs(x * coef.reshape(1, -1)), axis=0)
    feature_df = pd.DataFrame(
        {
            "model": model_name,
            "method": "cox_coefficient_mean_abs_contribution",
            "feature": feature_names,
            "coefficient": coef,
            "mean_abs_contribution": mean_abs,
        }
    ).sort_values("mean_abs_contribution", ascending=False)
    feature_df.to_csv(out_dir / "feature_contribution_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plot_df = block_df.sort_values("cindex_drop", ascending=True)
    colors = ["#8DA0CB" if x >= 0 else "#BDBDBD" for x in plot_df["cindex_drop"]]
    ax.barh(plot_df["block"], plot_df["cindex_drop"], color=colors, edgecolor="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("C-index drop after block permutation")
    ax.set_ylabel("")
    ax.set_title(f"Feature block contribution: {model_name}")
    ax.text(
        0.98,
        0.03,
        f"Baseline C-index={baseline_c:.3f}; repeats={args.repeats}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_contribution_summary.pdf")
    fig.savefig(plot_dir / "feature_contribution_summary.png", dpi=300)
    plt.close(fig)

    logger.info("Wrote %s and %s", block_path, out_dir / "feature_contribution_features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="pdac_project/results/01_tcga_prep")
    parser.add_argument("--benchmark-dir", default="pdac_project/results/02_benchmark")
    parser.add_argument("--output-dir", default="pdac_project/results/02_benchmark")
    parser.add_argument("--endpoint", default="OS", choices=["OS", "PFS"])
    parser.add_argument("--model", default="auto")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260416)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
