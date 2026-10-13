#!/usr/bin/env python
"""Round-5 statistical sensitivity analyses requested by pre-submission review.

This script addresses three reviewer-facing issues:

1. Replace in-sample Table 7 risk-effect estimates with out-of-fold (OOF)
   expression-risk sensitivity estimates.
2. Re-express external validation Cox effects as cohort-standardized per-SD
   hazard ratios instead of per-1-unit transferred-score effects.
3. Add fixed-effect, DerSimonian-Laird random-effects and leave-one-out
   meta-analysis summaries.

The OOF expression-risk sensitivity is intentionally a leakage-control
surrogate: feature selection, imputation, scaling, PCA and penalized Cox fitting
are all refit inside training folds before predicting held-out patients.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "results" / "11_round5_statistical_sensitivity"
EXT_DIR = PROJECT / "results" / "07_external_validation_rescue"
PURITY_FILE = PROJECT / "data_external" / "pancanatlas" / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
SEEDS = [42, 52, 62, 72, 82]


@dataclass(frozen=True)
class OOFConfig:
    top_var_features: int
    pca_components: int
    penalizer: float


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - values.mean()) / sd


def p_text(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def ci_text(hr: float | None, low: float | None, high: float | None) -> str:
    if any(x is None or not np.isfinite(x) for x in [hr, low, high]):
        return "NA"
    return f"{hr:.2f} ({low:.2f}-{high:.2f})"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expr = pd.read_csv(PROJECT / "results" / "01_tcga_prep" / "expression_matrix.csv", index_col=0)
    if "PATIENT_ID" in expr.columns:
        expr = expr.set_index("PATIENT_ID")
    expr.index = expr.index.astype(str)
    expr = expr.apply(pd.to_numeric, errors="coerce")

    clinical = pd.read_csv(PROJECT / "results" / "01_tcga_prep" / "clinical_matrix.csv")
    k2 = pd.read_csv(PROJECT / "results" / "03_subtypes" / "subtype_assignments_k2.csv")
    immune = pd.read_csv(PROJECT / "results" / "06_immune" / "immune_signature_scores.csv")

    purity = pd.read_csv(PURITY_FILE, sep="\t").rename(
        columns={
            "array": "SAMPLE_ID",
            "call status": "ABSOLUTE_CALL_STATUS",
            "purity": "ABSOLUTE_PURITY",
            "ploidy": "ABSOLUTE_PLOIDY",
            "Cancer DNA fraction": "CANCER_DNA_FRACTION",
        }
    )
    purity = purity[["SAMPLE_ID", "ABSOLUTE_CALL_STATUS", "ABSOLUTE_PURITY", "ABSOLUTE_PLOIDY", "CANCER_DNA_FRACTION"]]
    return expr, clinical, k2, immune, purity


def fit_predict_expression_cox(
    x_train: pd.DataFrame,
    time_train: np.ndarray,
    event_train: np.ndarray,
    x_test: pd.DataFrame,
    cfg: OOFConfig,
) -> np.ndarray:
    variance = x_train.var(axis=0, skipna=True).sort_values(ascending=False)
    selected = variance.head(min(cfg.top_var_features, len(variance))).index.tolist()
    xtr = x_train[selected].to_numpy(dtype=float)
    xte = x_test.reindex(columns=selected).to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(xtr))
    xte = scaler.transform(imputer.transform(xte))

    n_components = min(cfg.pca_components, xtr.shape[1], max(1, xtr.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=1)
    ztr = pca.fit_transform(xtr)
    zte = pca.transform(xte)

    cols = [f"PC{i+1}" for i in range(ztr.shape[1])]
    fit_df = pd.DataFrame(ztr, columns=cols)
    fit_df["OS_TIME"] = time_train.astype(float)
    fit_df["OS_EVENT"] = event_train.astype(int)

    cph = CoxPHFitter(penalizer=cfg.penalizer)
    cph.fit(fit_df, duration_col="OS_TIME", event_col="OS_EVENT", show_progress=False)
    pred_df = pd.DataFrame(zte, columns=cols)
    return cph.predict_log_partial_hazard(pred_df).to_numpy(dtype=float)


def tune_oof_config(
    x_train: pd.DataFrame,
    time_train: np.ndarray,
    event_train: np.ndarray,
    seed: int,
) -> tuple[OOFConfig, float]:
    configs = [
        OOFConfig(top_var_features=500, pca_components=10, penalizer=0.05),
        OOFConfig(top_var_features=500, pca_components=20, penalizer=0.10),
        OOFConfig(top_var_features=1000, pca_components=20, penalizer=0.10),
        OOFConfig(top_var_features=1000, pca_components=30, penalizer=0.25),
        OOFConfig(top_var_features=1500, pca_components=30, penalizer=0.25),
        OOFConfig(top_var_features=1500, pca_components=40, penalizer=0.50),
    ]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    y_strat = event_train.astype(int)
    best_cfg = configs[0]
    best_score = -np.inf
    for cfg in configs:
        scores: list[float] = []
        for tr, va in inner.split(np.arange(len(y_strat)), y_strat):
            try:
                risk = fit_predict_expression_cox(
                    x_train.iloc[tr],
                    time_train[tr],
                    event_train[tr],
                    x_train.iloc[va],
                    cfg,
                )
                score = concordance_index(time_train[va], -risk, event_train[va])
                if np.isfinite(score):
                    scores.append(float(score))
            except Exception:
                continue
        mean_score = float(np.mean(scores)) if scores else np.nan
        if np.isfinite(mean_score) and mean_score > best_score:
            best_cfg = cfg
            best_score = mean_score
    return best_cfg, best_score


def generate_oof_expression_risk(expr: pd.DataFrame, clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = clinical.set_index("PATIENT_ID")
    common = sorted(set(expr.index) & set(clin.index))
    expr = expr.loc[common]
    clin = clin.loc[common]
    time = pd.to_numeric(clin["OS_TIME"], errors="coerce")
    event = pd.to_numeric(clin["OS_EVENT"], errors="coerce")
    valid = time.notna() & event.notna() & (time > 0)
    expr = expr.loc[valid.index[valid]]
    clin = clin.loc[valid.index[valid]]
    time = time.loc[clin.index].to_numpy(dtype=float)
    event = event.loc[clin.index].to_numpy(dtype=int)

    pred = pd.DataFrame(index=clin.index)
    fold_rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seed_pred = pd.Series(index=clin.index, dtype=float)
        for fold, (tr, te) in enumerate(outer.split(np.arange(len(clin)), event), start=1):
            cfg, inner_c = tune_oof_config(expr.iloc[tr], time[tr], event[tr], seed + fold)
            risk = fit_predict_expression_cox(expr.iloc[tr], time[tr], event[tr], expr.iloc[te], cfg)
            test_ids = clin.index[te]
            seed_pred.loc[test_ids] = risk
            cidx = concordance_index(time[te], -risk, event[te])
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_test": int(len(te)),
                    "events_test": int(event[te].sum()),
                    "c_index": float(cidx),
                    "inner_c_index": inner_c,
                    "top_var_features": cfg.top_var_features,
                    "pca_components": cfg.pca_components,
                    "penalizer": cfg.penalizer,
                }
            )
        pred[f"oof_risk_seed_{seed}"] = zscore(seed_pred)
    pred["oof_expr_risk"] = pred[[f"oof_risk_seed_{s}" for s in SEEDS]].mean(axis=1)
    pred["oof_expr_risk_z"] = zscore(pred["oof_expr_risk"])
    pred = pred.reset_index().rename(columns={"index": "PATIENT_ID"})
    fold_df = pd.DataFrame(fold_rows)
    return pred, fold_df


def build_sensitivity_frame(
    clinical: pd.DataFrame,
    k2: pd.DataFrame,
    immune: pd.DataFrame,
    purity: pd.DataFrame,
    oof: pd.DataFrame,
) -> pd.DataFrame:
    df = (
        clinical.merge(oof[["PATIENT_ID", "oof_expr_risk", "oof_expr_risk_z"]], on="PATIENT_ID", how="inner")
        .merge(k2[["PATIENT_ID", "subtype"]], on="PATIENT_ID", how="inner")
        .merge(immune[["PATIENT_ID", "STROMAL_SIGNATURE"]], on="PATIENT_ID", how="left")
        .merge(purity, on="SAMPLE_ID", how="left")
    )
    for col in ["OS_TIME", "OS_EVENT", "oof_expr_risk", "oof_expr_risk_z", "ABSOLUTE_PURITY", "STROMAL_SIGNATURE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ductal_tumor_type"] = df["TUMOR_TYPE"].eq("Pancreas Adenocarcinoma, Ductal Type")
    df["ductal_icd_o_3_8500_3"] = df["ICD_O_3_HISTOLOGY"].eq("8500/3")
    df["absolute_purity_called"] = df["ABSOLUTE_CALL_STATUS"].eq("called") & df["ABSOLUTE_PURITY"].notna()
    called = df[df["absolute_purity_called"]]
    lower_tertile = float(called["ABSOLUTE_PURITY"].quantile(1 / 3))
    median = float(called["ABSOLUTE_PURITY"].median())
    df["purity_lower_tertile_or_higher"] = df["ABSOLUTE_PURITY"] >= lower_tertile
    df["purity_median_or_higher"] = df["ABSOLUTE_PURITY"] >= median
    return df


def fit_single_cox(df: pd.DataFrame, covariates: Iterable[str]) -> tuple[pd.DataFrame | None, int, int]:
    covariates = list(covariates)
    d = df[["OS_TIME", "OS_EVENT", *covariates]].copy()
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    d = d[d["OS_TIME"] > 0]
    n = int(len(d))
    events = int(d["OS_EVENT"].sum()) if n else 0
    if n < 10 or events < 5 or any(d[c].nunique() < 2 for c in covariates):
        return None, n, events
    cph = CoxPHFitter()
    cph.fit(d, duration_col="OS_TIME", event_col="OS_EVENT", show_progress=False)
    return cph.summary.reset_index().rename(columns={"covariate": "term"}), n, events


def risk_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    d = df.copy()
    d["risk_z"] = zscore(d["oof_expr_risk"])
    summary, n, events = fit_single_cox(d, ["risk_z"])
    cidx = np.nan
    scored = d[["OS_TIME", "OS_EVENT", "risk_z"]].dropna()
    if not scored.empty:
        cidx = float(concordance_index(scored["OS_TIME"], -scored["risk_z"], scored["OS_EVENT"]))
    out: dict[str, float | int | str] = {
        "n_risk": n,
        "events_risk": events,
        "oof_c_index": cidx,
        "risk_hr_per_sd": np.nan,
        "risk_ci_low": np.nan,
        "risk_ci_high": np.nan,
        "risk_p": np.nan,
        "risk_hr_95ci": "NA",
        "risk_p_text": "NA",
    }
    if summary is not None:
        row = summary.iloc[0]
        out.update(
            risk_hr_per_sd=float(row["exp(coef)"]),
            risk_ci_low=float(row["exp(coef) lower 95%"]),
            risk_ci_high=float(row["exp(coef) upper 95%"]),
            risk_p=float(row["p"]),
        )
        out["risk_hr_95ci"] = ci_text(out["risk_hr_per_sd"], out["risk_ci_low"], out["risk_ci_high"])
        out["risk_p_text"] = p_text(out["risk_p"])
    return out


def state_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    d = df[df["subtype"].isin([1, 2])].copy()
    d["state2"] = (d["subtype"] == 2).astype(int)
    summary, n, events = fit_single_cox(d, ["state2"])
    counts = d.groupby("subtype").agg(n=("PATIENT_ID", "size"), events=("OS_EVENT", "sum"))
    out: dict[str, float | int | str] = {
        "n_state": n,
        "events_state": events,
        "state1_n": int(counts.loc[1, "n"]) if 1 in counts.index else 0,
        "state1_events": int(counts.loc[1, "events"]) if 1 in counts.index else 0,
        "state2_n": int(counts.loc[2, "n"]) if 2 in counts.index else 0,
        "state2_events": int(counts.loc[2, "events"]) if 2 in counts.index else 0,
        "state2_vs_state1_hr": np.nan,
        "state_ci_low": np.nan,
        "state_ci_high": np.nan,
        "state_cox_p": np.nan,
        "state_logrank_p": np.nan,
        "state_hr_95ci": "NA",
        "state_p_text": "NA",
    }
    if summary is not None:
        row = summary.iloc[0]
        out.update(
            state2_vs_state1_hr=float(row["exp(coef)"]),
            state_ci_low=float(row["exp(coef) lower 95%"]),
            state_ci_high=float(row["exp(coef) upper 95%"]),
            state_cox_p=float(row["p"]),
        )
    g1 = d[d["state2"] == 0]
    g2 = d[d["state2"] == 1]
    if len(g1) > 5 and len(g2) > 5:
        lr = logrank_test(g1["OS_TIME"], g2["OS_TIME"], event_observed_A=g1["OS_EVENT"], event_observed_B=g2["OS_EVENT"])
        out["state_logrank_p"] = float(lr.p_value)
    out["state_hr_95ci"] = ci_text(out["state2_vs_state1_hr"], out["state_ci_low"], out["state_ci_high"])
    out["state_p_text"] = p_text(out["state_cox_p"])
    return out


def cohort_summary(name: str, definition: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "cohort": name,
        "filter_definition": definition,
        "n": int(len(df)),
        "n_with_os": int(df[["OS_TIME", "OS_EVENT"]].dropna().shape[0]),
        "os_events": int(pd.to_numeric(df["OS_EVENT"], errors="coerce").fillna(0).sum()),
        "absolute_purity_available_n": int(df["ABSOLUTE_PURITY"].notna().sum()),
    }
    out.update(risk_stats(df))
    out.update(state_stats(df))
    return out


def histology_purity_oof_summary(df: pd.DataFrame) -> pd.DataFrame:
    cohorts = [
        ("Full locked TCGA-PAAD", "All OOF-scored samples.", df),
        ("ABSOLUTE purity matched", "Called ABSOLUTE purity available.", df[df["absolute_purity_called"]]),
        ("Ductal tumor type only", "TUMOR_TYPE ductal.", df[df["ductal_tumor_type"]]),
        ("ICD-O-3 8500/3 ductal only", "ICD-O-3 8500/3.", df[df["ductal_icd_o_3_8500_3"]]),
        ("ABSOLUTE purity >= lower tertile", "Lowest purity tertile excluded.", df[df["absolute_purity_called"] & df["purity_lower_tertile_or_higher"]]),
        ("ABSOLUTE purity >= median", "Lower purity half excluded.", df[df["absolute_purity_called"] & df["purity_median_or_higher"]]),
        ("Ductal + purity >= lower tertile", "Ductal and lowest purity tertile excluded.", df[df["ductal_tumor_type"] & df["absolute_purity_called"] & df["purity_lower_tertile_or_higher"]]),
        ("Ductal + purity >= median", "Ductal and lower purity half excluded.", df[df["ductal_tumor_type"] & df["absolute_purity_called"] & df["purity_median_or_higher"]]),
    ]
    return pd.DataFrame([cohort_summary(name, definition, sub) for name, definition, sub in cohorts])


def adjusted_oof_models(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["absolute_purity_called"]].copy()
    d["risk_z"] = zscore(d["oof_expr_risk"])
    d["purity_z"] = zscore(d["ABSOLUTE_PURITY"])
    d["stromal_z"] = zscore(d["STROMAL_SIGNATURE"])
    d["nonductal"] = (~d["ductal_tumor_type"]).astype(int)
    d["state2"] = (d["subtype"] == 2).astype(int)
    specs = {
        "oof_risk_plus_absolute_purity": ["risk_z", "purity_z"],
        "oof_risk_plus_purity_and_histology": ["risk_z", "purity_z", "nonductal"],
        "oof_risk_plus_purity_stromal_histology": ["risk_z", "purity_z", "stromal_z", "nonductal"],
        "state_plus_purity_and_histology": ["state2", "purity_z", "nonductal"],
    }
    rows: list[dict[str, float | int | str]] = []
    for model, covs in specs.items():
        summary, n, events = fit_single_cox(d, covs)
        if summary is None:
            rows.append({"model": model, "term": "NA", "n": n, "events": events})
            continue
        for _, row in summary.iterrows():
            rows.append(
                {
                    "model": model,
                    "term": row["term"],
                    "n": n,
                    "events": events,
                    "hr": float(row["exp(coef)"]),
                    "ci_low": float(row["exp(coef) lower 95%"]),
                    "ci_high": float(row["exp(coef) upper 95%"]),
                    "p": float(row["p"]),
                    "hr_95ci": ci_text(float(row["exp(coef)"]), float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"])),
                    "p_text": p_text(float(row["p"])),
                }
            )
    return pd.DataFrame(rows)


def external_per_sd_summary() -> pd.DataFrame:
    cohorts = ["GSE62452", "GSE28735", "GSE85916"]
    rows: list[dict[str, float | int | str]] = []
    for cohort in cohorts:
        pred = pd.read_csv(EXT_DIR / f"student_model_predictions_{cohort}.csv")
        clinical = pd.read_csv(EXT_DIR / f"{cohort}_clinical_processed.csv")
        expr = pd.read_csv(EXT_DIR / f"{cohort}_expression_processed.csv", nrows=1)
        id_like = {"PATIENT_ID", "SAMPLE_ID", "sample", "Sample", "Unnamed: 0"}
        n_genes = len([c for c in expr.columns if c not in id_like])
        df = pred.merge(clinical, on="PATIENT_ID", how="inner")
        df = df.dropna(subset=["student_risk_score", "OS_TIME", "OS_EVENT"])
        df = df[df["OS_TIME"] > 0].copy()
        df["risk_z"] = zscore(df["student_risk_score"])
        summary, n, events = fit_single_cox(df.rename(columns={"risk_z": "risk_z"}), ["risk_z"])
        if summary is None:
            continue
        row = summary.iloc[0]
        cidx = float(concordance_index(df["OS_TIME"], -df["risk_z"], df["OS_EVENT"]))
        rows.append(
            {
                "cohort": cohort,
                "n_expression": int(pd.read_csv(EXT_DIR / f"student_model_predictions_{cohort}.csv").shape[0]),
                "n_genes": int(n_genes),
                "n_survival": n,
                "events": events,
                "c_index": cidx,
                "hr_per_sd": float(row["exp(coef)"]),
                "ci_low": float(row["exp(coef) lower 95%"]),
                "ci_high": float(row["exp(coef) upper 95%"]),
                "p": float(row["p"]),
                "log_hr": float(row["coef"]),
                "se_log_hr": float(row["se(coef)"]),
                "hr_95ci": ci_text(float(row["exp(coef)"]), float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"])),
                "p_text": p_text(float(row["p"])),
            }
        )
    return pd.DataFrame(rows)


def meta_analysis(df: pd.DataFrame, label: str, model: str) -> dict[str, float | int | str]:
    y = df["log_hr"].to_numpy(dtype=float)
    se = df["se_log_hr"].to_numpy(dtype=float)
    v = se**2
    w = 1.0 / v
    fixed = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - fixed) ** 2))
    k = int(len(df))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    use_tau2 = tau2 if model == "random_effects_DL" else 0.0
    wr = 1.0 / (v + use_tau2)
    mu = float(np.sum(wr * y) / np.sum(wr))
    se_mu = float(math.sqrt(1.0 / np.sum(wr)))
    z = mu / se_mu if se_mu > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    i2 = max(0.0, (q - (k - 1)) / q) * 100.0 if q > 0 else 0.0
    return {
        "analysis": label,
        "model": model,
        "n_cohorts": k,
        "cohorts": ",".join(df["cohort"].tolist()),
        "pooled_hr_per_sd": math.exp(mu),
        "ci_low": math.exp(mu - 1.96 * se_mu),
        "ci_high": math.exp(mu + 1.96 * se_mu),
        "p": p,
        "Q": q,
        "tau2": tau2,
        "I2": i2,
        "hr_95ci": ci_text(math.exp(mu), math.exp(mu - 1.96 * se_mu), math.exp(mu + 1.96 * se_mu)),
        "p_text": p_text(p),
    }


def external_meta_tables(external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows = [
        meta_analysis(external, "All three cohorts", "fixed_effect"),
        meta_analysis(external, "All three cohorts", "random_effects_DL"),
    ]
    loo_rows: list[dict[str, float | int | str]] = []
    for omit in external["cohort"].tolist():
        sub = external[external["cohort"] != omit].copy()
        row = meta_analysis(sub, f"Omit {omit}", "fixed_effect")
        row["omitted_cohort"] = omit
        loo_rows.append(row)
        row_re = meta_analysis(sub, f"Omit {omit}", "random_effects_DL")
        row_re["omitted_cohort"] = omit
        loo_rows.append(row_re)
    return pd.DataFrame(meta_rows), pd.DataFrame(loo_rows)


def make_external_figure(external: pd.DataFrame, meta: pd.DataFrame, loo: pd.DataFrame) -> Path:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "Figure7_round5_external_per_sd_meta.png"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=220)

    plot_rows = []
    for _, row in external.iterrows():
        plot_rows.append((row["cohort"], row["hr_per_sd"], row["ci_low"], row["ci_high"], f"n={int(row['n_survival'])}; C={row['c_index']:.3f}"))
    for _, row in meta.iterrows():
        plot_rows.append((row["model"], row["pooled_hr_per_sd"], row["ci_low"], row["ci_high"], f"I2={row['I2']:.1f}%"))
    y = np.arange(len(plot_rows))[::-1]
    ax = axes[0]
    for yi, (label, hr, lo, hi, note) in zip(y, plot_rows):
        ax.errorbar(hr, yi, xerr=[[hr - lo], [hi - hr]], fmt="o", color="#2F6DB5", ecolor="#8AAFD5", capsize=3)
        ax.text(hi + 0.035, yi, note, va="center", fontsize=8)
    ax.axvline(1.0, color="#8996A3", ls="--", lw=1)
    ax.set_xlim(0.75, 1.95)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in plot_rows], fontsize=8)
    ax.set_xlabel("HR per cohort-standardized SD")
    ax.set_title("External continuous-score Cox")
    ax.grid(True, axis="x", color="#E6ECF2")

    ax = axes[1]
    x = np.arange(len(external))
    ax.bar(x, external["c_index"], color="#6FA8DC")
    ax.axhline(0.5, color="#8996A3", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(external["cohort"], rotation=25, ha="right")
    ax.set_ylim(0.45, 0.60)
    ax.set_ylabel("C-index")
    ax.set_title("Modest discrimination")
    for xi, val in zip(x, external["c_index"]):
        ax.text(xi, val + 0.006, f"{val:.3f}", ha="center", fontsize=8)

    ax = axes[2]
    loo_fixed = loo[loo["model"] == "fixed_effect"].copy()
    y2 = np.arange(len(loo_fixed))[::-1]
    for yi, (_, row) in zip(y2, loo_fixed.iterrows()):
        hr, lo, hi = row["pooled_hr_per_sd"], row["ci_low"], row["ci_high"]
        color = "#D94C3A" if row["ci_low"] <= 1 <= row["ci_high"] else "#2F6DB5"
        ax.errorbar(hr, yi, xerr=[[hr - lo], [hi - hr]], fmt="o", color=color, ecolor="#B9C7D6", capsize=3)
        ax.text(hi + 0.03, yi, f"P={row['p_text']}", va="center", fontsize=8)
    ax.axvline(1.0, color="#8996A3", ls="--", lw=1)
    ax.set_xlim(0.85, 1.65)
    ax.set_yticks(y2)
    ax.set_yticklabels(loo_fixed["analysis"], fontsize=8)
    ax.set_xlabel("Leave-one-out pooled HR")
    ax.set_title("Sensitivity to omitted cohort")
    ax.grid(True, axis="x", color="#E6ECF2")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr, clinical, k2, immune, purity = load_inputs()

    oof, fold_df = generate_oof_expression_risk(expr, clinical)
    frame = build_sensitivity_frame(clinical, k2, immune, purity, oof)
    oof_summary = histology_purity_oof_summary(frame)
    adjusted = adjusted_oof_models(frame)

    external = external_per_sd_summary()
    meta, loo = external_meta_tables(external)
    fig7 = make_external_figure(external, meta, loo)

    oof.to_csv(OUT_DIR / "oof_expression_risk_patient_level.csv", index=False)
    fold_df.to_csv(OUT_DIR / "oof_expression_risk_fold_metrics.csv", index=False)
    frame.to_csv(OUT_DIR / "round5_histology_purity_patient_level.csv", index=False)
    oof_summary.to_csv(OUT_DIR / "round5_oof_histology_purity_sensitivity.csv", index=False)
    adjusted.to_csv(OUT_DIR / "round5_oof_adjusted_cox_models.csv", index=False)
    external.to_csv(OUT_DIR / "round5_external_per_sd_summary.csv", index=False)
    meta.to_csv(OUT_DIR / "round5_external_meta_summary.csv", index=False)
    loo.to_csv(OUT_DIR / "round5_external_leave_one_out_meta.csv", index=False)

    key = {
        "oof_cindex_mean_folds": float(fold_df["c_index"].mean()),
        "oof_cindex_sd_folds": float(fold_df["c_index"].std(ddof=1)),
        "oof_full_cindex": float(oof_summary.loc[oof_summary["cohort"].eq("Full locked TCGA-PAAD"), "oof_c_index"].iloc[0]),
        "external_fixed_hr_per_sd": float(meta.loc[meta["model"].eq("fixed_effect"), "pooled_hr_per_sd"].iloc[0]),
        "external_fixed_ci_low": float(meta.loc[meta["model"].eq("fixed_effect"), "ci_low"].iloc[0]),
        "external_fixed_ci_high": float(meta.loc[meta["model"].eq("fixed_effect"), "ci_high"].iloc[0]),
        "external_leave_one_out_omit_gse62452_hr": float(
            loo[(loo["model"].eq("fixed_effect")) & (loo["omitted_cohort"].eq("GSE62452"))]["pooled_hr_per_sd"].iloc[0]
        ),
        "external_leave_one_out_omit_gse62452_ci_low": float(
            loo[(loo["model"].eq("fixed_effect")) & (loo["omitted_cohort"].eq("GSE62452"))]["ci_low"].iloc[0]
        ),
        "external_leave_one_out_omit_gse62452_ci_high": float(
            loo[(loo["model"].eq("fixed_effect")) & (loo["omitted_cohort"].eq("GSE62452"))]["ci_high"].iloc[0]
        ),
        "figure7": str(fig7),
    }
    (OUT_DIR / "round5_key_numbers.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(json.dumps(key, indent=2))
    print("\nOOF Table 7 replacement:")
    print(oof_summary[["cohort", "n", "os_events", "oof_c_index", "risk_hr_95ci", "risk_p_text", "state_hr_95ci", "state_p_text"]].to_string(index=False))
    print("\nExternal per-SD:")
    print(external[["cohort", "n_survival", "c_index", "hr_95ci", "p_text"]].to_string(index=False))
    print("\nMeta:")
    print(meta[["model", "n_cohorts", "hr_95ci", "p_text", "I2", "tau2"]].to_string(index=False))
    print("\nLeave-one-out:")
    print(loo[["analysis", "model", "hr_95ci", "p_text", "I2"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
