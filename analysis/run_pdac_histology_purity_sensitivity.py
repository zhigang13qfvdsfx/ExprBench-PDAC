#!/usr/bin/env python
"""Histology and tumor-purity sensitivity analyses for the PDAC manuscript.

The analysis keeps the already locked TCGA-PAAD risk scores and k=2 molecular
state assignments unchanged, then tests whether the prognostic signal survives
clinically motivated sample filters:

1. Ductal histology only.
2. ABSOLUTE tumor-purity matched samples.
3. Excluding the lowest ABSOLUTE-purity tertile.
4. ABSOLUTE purity above the matched-cohort median.
5. Combined ductal/high-purity sensitivity subsets.

Outputs are written as CSV/JSON files plus a compact forest plot for manuscript
and supplement use.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "results" / "10_sensitivity_histology_purity"
PURITY_SOURCE_URL = "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def pformat(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def ci_text(hr: float | None, low: float | None, high: float | None) -> str:
    if any(x is None or not np.isfinite(x) for x in [hr, low, high]):
        return "NA"
    return f"{hr:.2f} ({low:.2f}-{high:.2f})"


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - values.mean()) / sd


def fit_cox(df: pd.DataFrame, covariates: Iterable[str]) -> tuple[pd.DataFrame | None, int, int]:
    covariates = list(covariates)
    d = df[["OS_TIME", "OS_EVENT", *covariates]].copy()
    for col in ["OS_TIME", "OS_EVENT", *covariates]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    d = d[d["OS_TIME"] > 0]
    n = int(len(d))
    events = int(d["OS_EVENT"].sum()) if n else 0
    if n < 10 or events < 5:
        return None, n, events
    if any(d[c].nunique() < 2 for c in covariates):
        return None, n, events
    cph = CoxPHFitter()
    cph.fit(d, duration_col="OS_TIME", event_col="OS_EVENT")
    return cph.summary.reset_index().rename(columns={"covariate": "term"}), n, events


def risk_model_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    d = df[["OS_TIME", "OS_EVENT", "expr_risk"]].copy()
    d["expr_risk_z"] = zscore(d["expr_risk"])
    summary, n, events = fit_cox(d.rename(columns={"expr_risk_z": "risk_z"}), ["risk_z"])
    out: dict[str, float | int | str] = {
        "n_risk": n,
        "events_risk": events,
        "risk_hr_per_sd": np.nan,
        "risk_ci_low": np.nan,
        "risk_ci_high": np.nan,
        "risk_p": np.nan,
        "risk_c_index": np.nan,
        "risk_hr_95ci": "NA",
        "risk_p_text": "NA",
    }
    if summary is None:
        return out
    row = summary.iloc[0]
    out.update(
        {
            "risk_hr_per_sd": float(row["exp(coef)"]),
            "risk_ci_low": float(row["exp(coef) lower 95%"]),
            "risk_ci_high": float(row["exp(coef) upper 95%"]),
            "risk_p": float(row["p"]),
        }
    )
    scored = d[["OS_TIME", "OS_EVENT", "expr_risk_z"]].dropna()
    scored = scored[scored["OS_TIME"] > 0]
    # lifelines' concordance expects larger predicted values to mean longer
    # survival, so negate risk scores because higher risk predicts earlier death.
    out["risk_c_index"] = float(concordance_index(scored["OS_TIME"], -scored["expr_risk_z"], scored["OS_EVENT"]))
    out["risk_hr_95ci"] = ci_text(out["risk_hr_per_sd"], out["risk_ci_low"], out["risk_ci_high"])
    out["risk_p_text"] = pformat(out["risk_p"])
    return out


def state_model_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    d = df[["OS_TIME", "OS_EVENT", "subtype"]].copy()
    d = d[d["subtype"].isin([1, 2])]
    d["state2"] = (d["subtype"] == 2).astype(int)
    summary, n, events = fit_cox(d, ["state2"])
    state_counts = d.groupby("subtype").agg(n=("OS_TIME", "size"), events=("OS_EVENT", "sum"))
    state1_n = int(state_counts.loc[1, "n"]) if 1 in state_counts.index else 0
    state2_n = int(state_counts.loc[2, "n"]) if 2 in state_counts.index else 0
    state1_events = int(state_counts.loc[1, "events"]) if 1 in state_counts.index else 0
    state2_events = int(state_counts.loc[2, "events"]) if 2 in state_counts.index else 0
    out: dict[str, float | int | str] = {
        "n_state": n,
        "events_state": events,
        "state1_n": state1_n,
        "state1_events": state1_events,
        "state2_n": state2_n,
        "state2_events": state2_events,
        "state2_vs_state1_hr": np.nan,
        "state_ci_low": np.nan,
        "state_ci_high": np.nan,
        "state_cox_p": np.nan,
        "state_logrank_p": np.nan,
        "state_hr_95ci": "NA",
        "state_p_text": "NA",
        "state_logrank_p_text": "NA",
    }
    if summary is not None:
        row = summary.iloc[0]
        out.update(
            {
                "state2_vs_state1_hr": float(row["exp(coef)"]),
                "state_ci_low": float(row["exp(coef) lower 95%"]),
                "state_ci_high": float(row["exp(coef) upper 95%"]),
                "state_cox_p": float(row["p"]),
            }
        )
    g1 = d[d["state2"] == 0]
    g2 = d[d["state2"] == 1]
    if len(g1) >= 5 and len(g2) >= 5:
        lr = logrank_test(
            g1["OS_TIME"],
            g2["OS_TIME"],
            event_observed_A=g1["OS_EVENT"],
            event_observed_B=g2["OS_EVENT"],
        )
        out["state_logrank_p"] = float(lr.p_value)
    out["state_hr_95ci"] = ci_text(out["state2_vs_state1_hr"], out["state_ci_low"], out["state_ci_high"])
    out["state_p_text"] = pformat(out["state_cox_p"])
    out["state_logrank_p_text"] = pformat(out["state_logrank_p"])
    return out


def cohort_summary(name: str, definition: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "cohort": name,
        "filter_definition": definition,
        "n": int(len(df)),
        "n_with_os": int(df[["OS_TIME", "OS_EVENT"]].dropna().shape[0]),
        "os_events": int(pd.to_numeric(df["OS_EVENT"], errors="coerce").fillna(0).sum()),
        "ductal_tumor_type_n": int(df["ductal_tumor_type"].sum()),
        "nonductal_or_other_n": int((~df["ductal_tumor_type"]).sum()),
        "absolute_purity_available_n": int(df["ABSOLUTE_PURITY"].notna().sum()),
        "absolute_purity_median": float(df["ABSOLUTE_PURITY"].median()) if df["ABSOLUTE_PURITY"].notna().any() else np.nan,
    }
    out.update(risk_model_stats(df))
    out.update(state_model_stats(df))
    return out


def adjusted_models(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["risk_z"] = zscore(d["expr_risk"])
    d["purity_z"] = zscore(d["ABSOLUTE_PURITY"])
    d["stromal_z"] = zscore(d["STROMAL_SIGNATURE"])
    d["nonductal"] = (~d["ductal_tumor_type"]).astype(int)
    d["state2"] = (d["subtype"] == 2).astype(int)

    specs = {
        "risk_plus_absolute_purity": ["risk_z", "purity_z"],
        "risk_plus_purity_and_histology": ["risk_z", "purity_z", "nonductal"],
        "risk_plus_stromal_proxy": ["risk_z", "stromal_z"],
        "risk_plus_purity_stromal_histology": ["risk_z", "purity_z", "stromal_z", "nonductal"],
        "state_plus_purity_and_histology": ["state2", "purity_z", "nonductal"],
    }

    rows: list[dict[str, float | int | str]] = []
    for model, covariates in specs.items():
        summary, n, events = fit_cox(d, covariates)
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
                    "p_text": pformat(float(row["p"])),
                }
            )
    return pd.DataFrame(rows)


def make_forest_plot(summary: pd.DataFrame, out_path: Path) -> None:
    rows = summary[
        summary["cohort"].isin(
            [
                "Full locked TCGA-PAAD",
                "Ductal tumor type only",
                "ABSOLUTE purity matched",
                "ABSOLUTE purity >= lower tertile",
                "ABSOLUTE purity >= median",
                "Ductal + purity >= lower tertile",
            ]
        )
    ].copy()
    rows = rows.iloc[::-1].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=220, sharey=True)
    y = np.arange(len(rows))
    labels = [f"{r.cohort}\n(n={int(r.n)}, events={int(r.os_events)})" for r in rows.itertuples()]

    for ax, prefix, title in [
        (axes[0], "risk", "Locked expression risk score"),
        (axes[1], "state", "Primary k=2 State 2 vs State 1"),
    ]:
        hr_col = "risk_hr_per_sd" if prefix == "risk" else "state2_vs_state1_hr"
        lo_col = "risk_ci_low" if prefix == "risk" else "state_ci_low"
        hi_col = "risk_ci_high" if prefix == "risk" else "state_ci_high"
        hrs = rows[hr_col].to_numpy(dtype=float)
        lows = rows[lo_col].to_numpy(dtype=float)
        highs = rows[hi_col].to_numpy(dtype=float)
        ax.errorbar(hrs, y, xerr=[hrs - lows, highs - hrs], fmt="o", color="#2F6DB5", ecolor="#8AAFD5", capsize=3)
        ax.axvline(1, color="#8996A3", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Hazard ratio (log scale)")
        ax.grid(True, axis="x", color="#E6ECF2", lw=0.8)
        ax.set_ylim(-0.5, len(rows) - 0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    clinical = read_csv(PROJECT / "results" / "01_tcga_prep" / "clinical_matrix.csv")
    risk = read_csv(PROJECT / "results" / "02_benchmark" / "artifacts" / "full_cohort_risk_scores.csv")
    k2 = read_csv(PROJECT / "results" / "03_subtypes" / "subtype_assignments_k2.csv")
    immune = read_csv(PROJECT / "results" / "06_immune" / "immune_signature_scores.csv")
    purity = read_csv(PROJECT / "data_external" / "pancanatlas" / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt", sep="\t")

    purity = purity.rename(
        columns={
            "array": "SAMPLE_ID",
            "sample": "ABSOLUTE_SAMPLE_BARCODE",
            "call status": "ABSOLUTE_CALL_STATUS",
            "purity": "ABSOLUTE_PURITY",
            "ploidy": "ABSOLUTE_PLOIDY",
            "Cancer DNA fraction": "CANCER_DNA_FRACTION",
            "Subclonal genome fraction": "SUBCLONAL_GENOME_FRACTION",
        }
    )
    purity = purity[
        [
            "SAMPLE_ID",
            "ABSOLUTE_SAMPLE_BARCODE",
            "ABSOLUTE_CALL_STATUS",
            "ABSOLUTE_PURITY",
            "ABSOLUTE_PLOIDY",
            "CANCER_DNA_FRACTION",
            "SUBCLONAL_GENOME_FRACTION",
        ]
    ]

    df = (
        clinical.merge(risk, on="PATIENT_ID", how="inner")
        .merge(k2[["PATIENT_ID", "subtype"]], on="PATIENT_ID", how="inner")
        .merge(immune[["PATIENT_ID", "STROMAL_SIGNATURE"]], on="PATIENT_ID", how="left")
        .merge(purity, on="SAMPLE_ID", how="left")
        .rename(columns={"Expr-Penalized-Cox": "expr_risk", "LateFusion-Cox": "latefusion_risk"})
    )

    numeric_cols = [
        "OS_TIME",
        "OS_EVENT",
        "expr_risk",
        "latefusion_risk",
        "STROMAL_SIGNATURE",
        "ABSOLUTE_PURITY",
        "ABSOLUTE_PLOIDY",
        "CANCER_DNA_FRACTION",
        "SUBCLONAL_GENOME_FRACTION",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ductal_tumor_type"] = df["TUMOR_TYPE"].eq("Pancreas Adenocarcinoma, Ductal Type")
    df["ductal_icd_o_3_8500_3"] = df["ICD_O_3_HISTOLOGY"].eq("8500/3")
    df["absolute_purity_called"] = df["ABSOLUTE_CALL_STATUS"].eq("called") & df["ABSOLUTE_PURITY"].notna()

    matched = df[df["absolute_purity_called"]].copy()
    lower_tertile_threshold = float(matched["ABSOLUTE_PURITY"].quantile(1 / 3))
    median_threshold = float(matched["ABSOLUTE_PURITY"].median())
    matched["purity_lower_tertile_or_higher"] = matched["ABSOLUTE_PURITY"] >= lower_tertile_threshold
    matched["purity_median_or_higher"] = matched["ABSOLUTE_PURITY"] >= median_threshold
    df = df.merge(
        matched[["PATIENT_ID", "purity_lower_tertile_or_higher", "purity_median_or_higher"]],
        on="PATIENT_ID",
        how="left",
    )
    df["purity_lower_tertile_or_higher"] = df["purity_lower_tertile_or_higher"].astype("boolean").fillna(False).astype(bool)
    df["purity_median_or_higher"] = df["purity_median_or_higher"].astype("boolean").fillna(False).astype(bool)

    cohorts = [
        (
            "Full locked TCGA-PAAD",
            "All samples with locked risk score and k=2 state assignment.",
            df,
        ),
        (
            "ABSOLUTE purity matched",
            "Samples with called ABSOLUTE purity in the GDC PanCanAtlas file.",
            df[df["absolute_purity_called"]],
        ),
        (
            "Ductal tumor type only",
            'TUMOR_TYPE == "Pancreas Adenocarcinoma, Ductal Type".',
            df[df["ductal_tumor_type"]],
        ),
        (
            "ICD-O-3 8500/3 ductal only",
            'ICD_O_3_HISTOLOGY == "8500/3".',
            df[df["ductal_icd_o_3_8500_3"]],
        ),
        (
            "ABSOLUTE purity >= lower tertile",
            f"Called ABSOLUTE purity >= {lower_tertile_threshold:.2f}; lowest tertile excluded.",
            df[df["absolute_purity_called"] & df["purity_lower_tertile_or_higher"]],
        ),
        (
            "ABSOLUTE purity >= median",
            f"Called ABSOLUTE purity >= {median_threshold:.2f}; lower half excluded.",
            df[df["absolute_purity_called"] & df["purity_median_or_higher"]],
        ),
        (
            "Ductal + purity >= lower tertile",
            f"Ductal tumor type and ABSOLUTE purity >= {lower_tertile_threshold:.2f}.",
            df[df["ductal_tumor_type"] & df["absolute_purity_called"] & df["purity_lower_tertile_or_higher"]],
        ),
        (
            "Ductal + purity >= median",
            f"Ductal tumor type and ABSOLUTE purity >= {median_threshold:.2f}.",
            df[df["ductal_tumor_type"] & df["absolute_purity_called"] & df["purity_median_or_higher"]],
        ),
    ]

    summary = pd.DataFrame([cohort_summary(name, definition, sub) for name, definition, sub in cohorts])
    summary.to_csv(OUT_DIR / "histology_purity_sensitivity_survival.csv", index=False)

    df.to_csv(OUT_DIR / "histology_purity_patient_level.csv", index=False)
    adjusted = adjusted_models(df[df["absolute_purity_called"]].copy())
    adjusted.to_csv(OUT_DIR / "histology_purity_adjusted_cox_models.csv", index=False)

    histology_counts = (
        df.groupby(["TUMOR_TYPE", "ICD_O_3_HISTOLOGY"], dropna=False)
        .agg(
            n=("PATIENT_ID", "size"),
            os_events=("OS_EVENT", "sum"),
            absolute_purity_available_n=("ABSOLUTE_PURITY", lambda x: int(x.notna().sum())),
            absolute_purity_median=("ABSOLUTE_PURITY", "median"),
        )
        .reset_index()
        .sort_values(["n", "TUMOR_TYPE"], ascending=[False, True])
    )
    histology_counts.to_csv(OUT_DIR / "histology_counts_with_purity.csv", index=False)

    state_purity = (
        df[df["absolute_purity_called"]]
        .groupby("subtype")
        .agg(
            n=("PATIENT_ID", "size"),
            os_events=("OS_EVENT", "sum"),
            median_absolute_purity=("ABSOLUTE_PURITY", "median"),
            mean_absolute_purity=("ABSOLUTE_PURITY", "mean"),
            median_stromal_signature=("STROMAL_SIGNATURE", "median"),
        )
        .reset_index()
    )
    state_purity.to_csv(OUT_DIR / "state_purity_distribution.csv", index=False)

    make_forest_plot(summary, fig_dir / "histology_purity_sensitivity_forest.png")

    metadata = {
        "purity_source": "GDC PanCanAtlas ABSOLUTE purity/ploidy file",
        "purity_source_url": PURITY_SOURCE_URL,
        "n_locked_tcga_paad": int(len(df)),
        "n_absolute_purity_called": int(df["absolute_purity_called"].sum()),
        "n_ductal_tumor_type": int(df["ductal_tumor_type"].sum()),
        "n_icd_o_3_8500_3": int(df["ductal_icd_o_3_8500_3"].sum()),
        "absolute_purity_lower_tertile_threshold": lower_tertile_threshold,
        "absolute_purity_median_threshold": median_threshold,
        "analysis_note": (
            "Risk and subtype models were not retrained; this is a locked-score/locked-state "
            "sensitivity analysis against histology and ABSOLUTE-purity filters."
        ),
    }
    (OUT_DIR / "histology_purity_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(summary[["cohort", "n", "os_events", "risk_hr_95ci", "risk_p_text", "state_hr_95ci", "state_p_text"]].to_string(index=False))
    print(adjusted[["model", "term", "n", "events", "hr_95ci", "p_text"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
