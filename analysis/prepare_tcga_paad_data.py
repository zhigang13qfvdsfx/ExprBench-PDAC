#!/usr/bin/env python3
"""
Publication-grade TCGA-PAAD preprocessing pipeline.

This script prepares patient-aligned expression, methylation, mutation, and clinical
matrices and writes figure-ready intermediate tables for downstream modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


TCGA_PAT = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", flags=re.IGNORECASE)
NONSILENT = {
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
    "De_novo_Start_InFrame",
    "De_novo_Start_OutOfFrame",
    "Start_Codon_Del",
    "Start_Codon_Ins",
    "Start_Codon_SNP",
    "Stop_Codon_Del",
    "Stop_Codon_Ins",
}

PATHWAY_MAP = {
    "KRAS_SIGNALING": {"KRAS", "NRAS", "HRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3"},
    "PI3K_AKT": {"PIK3CA", "PIK3R1", "PTEN", "AKT1", "AKT2", "AKT3", "TSC1", "TSC2", "MTOR"},
    "TP53_PATHWAY": {"TP53", "MDM2", "CDKN2A", "ATM", "ATR", "CHEK2"},
    "TGF_BETA": {"SMAD4", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3"},
    "DNA_REPAIR": {"BRCA1", "BRCA2", "PALB2", "ATM", "ATR", "RAD51", "FANCA", "FANCD2", "MLH1", "MSH2"},
    "WNT_BETA_CATENIN": {"APC", "CTNNB1", "AXIN1", "AXIN2", "RNF43"},
    "NOTCH": {"NOTCH1", "NOTCH2", "NOTCH3", "FBXW7", "DLL1", "DLL4"},
    "CELL_CYCLE": {"CDKN2A", "CCND1", "CDK4", "CDK6", "RB1", "E2F1"},
}


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("tcga_prep")
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


def normalize_patient_id(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper().replace(".", "-")
    if not s:
        return None
    m = TCGA_PAT.search(s)
    if m:
        return m.group(1)
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return None


def write_gap(gap_report: Path, message: str) -> None:
    gap_report.parent.mkdir(parents=True, exist_ok=True)
    with gap_report.open("a", encoding="utf-8") as f:
        f.write(f"- {message}\n")


def download_cbio_study(data_dir: Path, logger: logging.Logger) -> None:
    """
    Download required cBioPortal PAAD files.

    Primary source (historical): S3 tarball.
    Fallback source: GitHub media endpoint (works for LFS-backed files).
    """
    required_files = [
        "data_clinical_patient.txt",
        "data_clinical_sample.txt",
        "data_mrna_seq_v2_rsem.txt",
        "data_methylation_hm27_hm450_merged.txt",
        "data_mutations.txt",
    ]
    extracted_flag = data_dir / ".download_complete"
    data_dir.mkdir(parents=True, exist_ok=True)

    if extracted_flag.exists() and all((data_dir / f).exists() for f in required_files):
        logger.info("TCGA-PAAD raw files already present: %s", data_dir)
        return

    # Try S3 archive first.
    s3_url = "https://cbioportal-datahub.s3.amazonaws.com/paad_tcga_pan_can_atlas_2018.tar.gz"
    tar_path = data_dir / "paad_tcga_pan_can_atlas_2018.tar.gz"
    try:
        logger.info("Attempting S3 archive download: %s", s3_url)
        with requests.get(s3_url, stream=True, timeout=240) as r:
            r.raise_for_status()
            with tar_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(data_dir)
    except Exception as s3_exc:
        logger.warning("S3 archive download failed: %s", s3_exc)

    if all((data_dir / f).exists() for f in required_files):
        extracted_flag.write_text("ok", encoding="utf-8")
        return

    # Fallback: GitHub media endpoint for LFS-backed files.
    logger.info("Using GitHub media fallback for required files.")
    base = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/paad_tcga_pan_can_atlas_2018"
    for fname in required_files:
        target = data_dir / fname
        if target.exists() and target.stat().st_size > 1024:
            continue
        url = f"{base}/{fname}"
        logger.info("Downloading %s", fname)
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    if not all((data_dir / f).exists() and (data_dir / f).stat().st_size > 1024 for f in required_files):
        raise RuntimeError("Could not download all required TCGA-PAAD files from fallback source.")
    extracted_flag.write_text("ok", encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        try:
            df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
            if df.shape[1] <= 1:
                df = pd.read_csv(path, sep=",", comment="#", low_memory=False)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python", comment="#")

    if df.empty:
        raise ValueError(f"Empty table: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_file(candidates: Sequence[str], roots: Sequence[Path]) -> Optional[Path]:
    low_candidates = [c.lower() for c in candidates]
    valid_suffix = {".txt", ".tsv", ".csv", ".maf", ".parquet", ".gz"}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            parts_lower = [x.lower() for x in p.parts]
            if any(seg in {".venv", "site-packages", "__pycache__"} for seg in parts_lower):
                continue
            if p.suffix.lower() not in valid_suffix:
                continue
            name = p.name.lower()
            if any(c in name for c in low_candidates):
                return p
    return None


def pick_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required columns; tried: {candidates}")
    return None


def parse_time(series: pd.Series, source_col: str) -> pd.Series:
    t = pd.to_numeric(series, errors="coerce")
    u = source_col.upper()
    if "DAY" in u:
        t = t / 30.4375
    elif "WEEK" in u:
        t = t * 7.0 / 30.4375
    return t


def parse_event(series: pd.Series, endpoint: str) -> pd.Series:
    raw = series.astype(str).str.upper().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype=float)

    num = pd.to_numeric(raw, errors="coerce")
    out.loc[num.notna()] = (num[num.notna()] > 0).astype(float)

    pref = pd.to_numeric(raw.str.extract(r"^([01])")[0], errors="coerce")
    mask = out.isna() & pref.notna()
    out.loc[mask] = (pref[mask] > 0).astype(float)

    if endpoint.upper() == "OS":
        pos = raw.str.contains(r"DECEASED|DEAD|DIED", regex=True, na=False)
        neg = raw.str.contains(r"LIVING|ALIVE", regex=True, na=False)
    else:
        pos = raw.str.contains(r"PROGRESS|RECUR|RELAPS|EVENT|DECEASED|DEAD|YES", regex=True, na=False)
        neg = raw.str.contains(r"DISEASE[ _-]*FREE|NO[ _-]*PROGRESS|CENSORED|NO EVENT|LIVING|ALIVE|NONE", regex=True, na=False)

    out.loc[out.isna() & pos] = 1.0
    out.loc[out.isna() & neg] = 0.0
    return out


def derive_endpoint(df: pd.DataFrame, endpoint: str) -> Tuple[pd.Series, pd.Series, str, str]:
    if endpoint.upper() == "OS":
        tc = ["OS_MONTHS", "OS_TIME", "OS_DAYS", "overall_survival_time"]
        ec = ["OS_STATUS", "OS_EVENT", "VITAL_STATUS", "overall_survival_event"]
    else:
        tc = ["PFS_MONTHS", "PFS_TIME", "PFS_DAYS", "PFI_MONTHS", "PFI_TIME", "DFS_MONTHS", "RFS_MONTHS"]
        ec = ["PFS_STATUS", "PFS_EVENT", "PFI_STATUS", "PFI_EVENT", "DFS_STATUS", "RFS_STATUS", "PROGRESSION_STATUS", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"]

    tcol = pick_col(df, tc, required=False)
    ecol = pick_col(df, ec, required=False)
    if tcol is None or ecol is None:
        raise ValueError(f"Could not derive {endpoint}: missing time/event columns")

    t = parse_time(df[tcol], tcol)
    e = parse_event(df[ecol], endpoint)
    if int((t.notna() & e.notna()).sum()) == 0:
        raise ValueError(f"Could not derive valid {endpoint} labels from {tcol} / {ecol}")
    return t, e, tcol, ecol


def clinical_merge(clin_patient: pd.DataFrame, clin_sample: Optional[pd.DataFrame]) -> pd.DataFrame:
    p = clin_patient.copy()
    p_col = pick_col(p, ["PATIENT_ID", "patient_id", "bcr_patient_barcode", "submitter_id", "case_submitter_id", "Patient_ID"])
    p["PATIENT_ID"] = p[p_col].apply(normalize_patient_id)
    p = p[p["PATIENT_ID"].notna()].copy()
    p = p.sort_values(p.notna().sum(axis=1).name if p.notna().sum(axis=1).name else p.columns[0]) if len(p) > 0 else p
    p = p.drop_duplicates("PATIENT_ID", keep="first").set_index("PATIENT_ID")

    if clin_sample is not None and not clin_sample.empty:
        s = clin_sample.copy()
        s_col = pick_col(s, ["PATIENT_ID", "patient_id", "bcr_patient_barcode", "submitter_id", "case_submitter_id", "Patient_ID"])
        s["PATIENT_ID"] = s[s_col].apply(normalize_patient_id)
        s = s[s["PATIENT_ID"].notna()].copy()
        s = s.drop_duplicates("PATIENT_ID", keep="first").set_index("PATIENT_ID")
        merged = p.join(s, how="left", rsuffix="_sample")
    else:
        merged = p

    out = merged.copy()
    # Standardized clinical fields
    age_col = pick_col(out, ["AGE", "AGE_AT_DIAGNOSIS", "age_at_initial_pathologic_diagnosis"], required=False)
    sex_col = pick_col(out, ["SEX", "GENDER"], required=False)
    stage_col = pick_col(out, ["AJCC_PATHOLOGIC_TUMOR_STAGE", "STAGE", "PATHOLOGIC_STAGE"], required=False)
    grade_col = pick_col(out, ["GRADE", "NEOPLASM_HISTOLOGIC_GRADE"], required=False)

    out["AGE_STD"] = pd.to_numeric(out[age_col], errors="coerce") if age_col else np.nan
    out["SEX_STD"] = out[sex_col].astype(str).str.title() if sex_col else np.nan
    out["STAGE_STD"] = out[stage_col].astype(str).str.upper().str.strip() if stage_col else np.nan
    out["GRADE_STD"] = out[grade_col].astype(str).str.upper().str.strip() if grade_col else np.nan

    os_t, os_e, _, _ = derive_endpoint(out, "OS")
    out["OS_TIME"] = os_t
    out["OS_EVENT"] = pd.to_numeric(os_e, errors="coerce").round().astype("Int64")

    try:
        pfs_t, pfs_e, _, _ = derive_endpoint(out, "PFS")
        out["PFS_TIME"] = pfs_t
        out["PFS_EVENT"] = pd.to_numeric(pfs_e, errors="coerce").round().astype("Int64")
    except Exception:
        out["PFS_TIME"] = np.nan
        out["PFS_EVENT"] = pd.Series([pd.NA] * out.shape[0], index=out.index, dtype="Int64")

    return out.sort_index()

def parse_sample_cols(df: pd.DataFrame, feature_col: str) -> Tuple[List[str], Dict[str, str]]:
    cols: List[str] = []
    mapping: Dict[str, str] = {}
    for c in df.columns:
        if c == feature_col:
            continue
        pid = normalize_patient_id(c)
        if pid:
            cols.append(c)
            mapping[c] = pid
    if not cols:
        raise ValueError("No TCGA sample columns found.")
    return cols, mapping


def dedup_feature_rows(df: pd.DataFrame, feature_col: str, sample_cols: Sequence[str]) -> pd.DataFrame:
    tmp = df[[feature_col] + list(sample_cols)].copy()
    tmp["_f"] = tmp[feature_col].astype(str).str.strip()
    tmp = tmp[~tmp["_f"].isin(["", "NA", "NAN", "NONE", "NULL"])].copy()

    num = tmp[list(sample_cols)].apply(pd.to_numeric, errors="coerce")
    num.insert(0, "_f", tmp["_f"].values)

    var = num.drop(columns=["_f"]).var(axis=1, skipna=True).fillna(-np.inf)
    ranked = num.assign(_var=var).sort_values(["_f", "_var"], ascending=[True, False], kind="mergesort")
    out = ranked.drop_duplicates("_f", keep="first").drop(columns=["_var"]).set_index("_f")
    return out


def process_expression(raw: pd.DataFrame) -> pd.DataFrame:
    fcol = pick_col(raw, ["Hugo_Symbol", "GENE", "Gene", "gene_name", "SYMBOL", "GENE_SYMBOL"])
    sample_cols, smap = parse_sample_cols(raw, fcol)
    feat = dedup_feature_rows(raw, fcol, sample_cols)

    mat = feat.T
    mat.index = [smap[i] for i in mat.index]
    mat = mat.groupby(level=0).median(numeric_only=True).sort_index()

    # apply log1p if values look unlogged
    if np.nanpercentile(mat.to_numpy(dtype=float), 95) > 30:
        mat = np.log1p(mat.clip(lower=0))

    return mat


def load_probe_gene_map(input_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        "probe_gene_map.csv",
        "methylation_probe_annotation.csv",
        "hm450_annotation.csv",
        "hm27_annotation.csv",
        "data_methylation_annotation.txt",
    ]
    p = find_file(candidates, [input_dir])
    if p is None:
        return None

    ann = read_table(p)
    ann.columns = [c.strip() for c in ann.columns]

    probe_col = pick_col(ann, ["probe", "Probe", "ID", "IlmnID", "Composite Element REF", "ID_REF"], required=False)
    gene_col = pick_col(ann, ["gene", "Gene", "UCSC_RefGene_Name", "Gene_Symbol", "Symbol"], required=False)
    if probe_col is None or gene_col is None:
        return None

    x = ann[[probe_col, gene_col]].copy()
    x.columns = ["probe", "gene"]
    x["probe"] = x["probe"].astype(str).str.strip()
    x["gene"] = x["gene"].astype(str).str.split(";").str[0].str.strip()
    x = x[(x["probe"] != "") & (x["gene"] != "")]
    x = x.drop_duplicates()
    return x


def process_methylation(raw: pd.DataFrame, input_dir: Path, gap_report: Path, logger: logging.Logger) -> pd.DataFrame:
    fcol = pick_col(raw, ["ENTITY_STABLE_ID", "Composite Element REF", "ID_REF", "IlmnID", "probe_id", "Probe"])
    sample_cols, smap = parse_sample_cols(raw, fcol)
    feat = dedup_feature_rows(raw, fcol, sample_cols)

    probe_gene = load_probe_gene_map(input_dir)
    if probe_gene is None:
        write_gap(gap_report, "Methylation probe->gene annotation unavailable; retained probe-level methylation features.")
        out = feat
    else:
        m = feat.join(probe_gene.set_index("probe"), how="inner")
        if m.empty:
            write_gap(gap_report, "Probe->gene annotation found but no probes overlapped; retained probe-level methylation features.")
            out = feat
        else:
            out = m.groupby("gene").mean(numeric_only=True)
            logger.info("Aggregated methylation to gene-level using %d mapped probes.", m.shape[0])

    mat = out.T
    mat.index = [smap[i] for i in mat.index]
    mat = mat.groupby(level=0).median(numeric_only=True).sort_index()
    return mat


def process_mutation(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gcol = pick_col(raw, ["Hugo_Symbol", "GENE", "Gene", "gene_symbol", "SYMBOL"])
    scol = pick_col(raw, ["Tumor_Sample_Barcode", "SAMPLE_ID", "Sample", "sample_id"])
    ccol = pick_col(raw, ["Variant_Classification", "VARIANT_CLASSIFICATION", "Consequence"], required=False)

    m = raw.copy()
    m["PATIENT_ID"] = m[scol].apply(normalize_patient_id)
    m["GENE"] = m[gcol].astype(str).str.strip().str.upper()
    m = m[(m["PATIENT_ID"].notna()) & (m["GENE"] != "")].copy()

    if ccol is not None:
        cls = m[ccol].astype(str).str.strip()
        m = m[cls.isin(NONSILENT)].copy()

    if m.empty:
        raise ValueError("Mutation table empty after filtering.")

    gene_bin = pd.crosstab(m["PATIENT_ID"], m["GENE"])
    gene_bin = (gene_bin > 0).astype(np.int8).sort_index()

    top_genes = gene_bin.sum(axis=0).sort_values(ascending=False).head(50).index.tolist()
    driver_mat = gene_bin[top_genes].copy() if top_genes else pd.DataFrame(index=gene_bin.index)

    pathway = pd.DataFrame(index=gene_bin.index)
    for pname, genes in PATHWAY_MAP.items():
        overlap = [g for g in genes if g in gene_bin.columns]
        if not overlap:
            pathway[pname] = 0.0
        else:
            pathway[pname] = gene_bin[overlap].sum(axis=1) / max(1, len(overlap))

    n_nonsilent = m.groupby("PATIENT_ID").size().reindex(gene_bin.index).fillna(0)
    pathway["N_NONSILENT"] = n_nonsilent.astype(float)
    pathway["TMB_PER_MB"] = pathway["N_NONSILENT"] / 38.0

    driver_features = pd.concat([driver_mat, pathway], axis=1)
    return gene_bin, driver_mat, driver_features


def modality_availability(expr: pd.DataFrame, meth: pd.DataFrame, mut: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(set(expr.index) | set(meth.index) | set(mut.index) | set(clin.index))
    df = pd.DataFrame(index=ids)
    df.index.name = "PATIENT_ID"
    df["has_expression"] = df.index.isin(expr.index)
    df["has_methylation"] = df.index.isin(meth.index)
    df["has_mutation"] = df.index.isin(mut.index)
    df["has_clinical"] = df.index.isin(clin.index)
    df["n_modalities"] = df[["has_expression", "has_methylation", "has_mutation", "has_clinical"]].sum(axis=1)
    return df.reset_index()


def make_missingness(expr: pd.DataFrame, meth: pd.DataFrame, mut: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(set(expr.index) & set(meth.index) & set(mut.index) & set(clin.index))
    out = pd.DataFrame(index=ids)
    out.index.name = "PATIENT_ID"

    out["expr_missing_frac"] = expr.reindex(ids).isna().mean(axis=1)
    out["meth_missing_frac"] = meth.reindex(ids).isna().mean(axis=1)
    out["mut_missing_frac"] = mut.reindex(ids).isna().mean(axis=1)

    clin_cols = ["AGE_STD", "SEX_STD", "STAGE_STD", "GRADE_STD", "OS_TIME", "OS_EVENT", "PFS_TIME", "PFS_EVENT"]
    present = [c for c in clin_cols if c in clin.columns]
    out["clin_missing_frac"] = clin.reindex(ids)[present].isna().mean(axis=1) if present else np.nan
    out["overall_missing_frac"] = out[[c for c in out.columns if c.endswith("_missing_frac")]].mean(axis=1)
    return out.reset_index()


def variance_summary(expr: pd.DataFrame, meth: pd.DataFrame, mut: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("expression", expr), ("methylation", meth), ("mutation", mut)]:
        v = pd.to_numeric(df.var(axis=0, skipna=True), errors="coerce").dropna()
        rows.append(
            {
                "modality": name,
                "n_features": int(df.shape[1]),
                "variance_mean": float(v.mean()) if len(v) else np.nan,
                "variance_median": float(v.median()) if len(v) else np.nan,
                "variance_q25": float(v.quantile(0.25)) if len(v) else np.nan,
                "variance_q75": float(v.quantile(0.75)) if len(v) else np.nan,
                "zero_variance_features": int((v <= 0).sum()) if len(v) else int(df.shape[1]),
            }
        )
    return pd.DataFrame(rows)


def write_matrix(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_parquet(out_dir / f"{stem}.parquet")
    df.to_csv(out_dir / f"{stem}.csv")


def figure_ready_exports(out_dir: Path, avail: pd.DataFrame, miss: pd.DataFrame, varsum: pd.DataFrame, clin: pd.DataFrame, mut: pd.DataFrame, sample_log: pd.DataFrame) -> None:
    fr = out_dir / "figure_ready"
    fr.mkdir(parents=True, exist_ok=True)

    # Figure2A / S1B
    sample_log.to_csv(fr / "Figure2A_flow_counts.csv", index=False)
    sample_log.to_csv(fr / "Supplementary_Figure_S1B_filtering_cascade.csv", index=False)

    # Figure2B
    avail.to_csv(fr / "Figure2B_upset_table.csv", index=False)

    # Figure2C
    c = pd.DataFrame(index=clin.index)
    c["age"] = pd.to_numeric(clin.get("AGE_STD", np.nan), errors="coerce")
    c["sex"] = clin.get("SEX_STD", np.nan)
    c["stage"] = clin.get("STAGE_STD", np.nan)
    c["grade"] = clin.get("GRADE_STD", np.nan)
    c["os_event"] = pd.to_numeric(clin.get("OS_EVENT", np.nan), errors="coerce")
    c["pfs_event"] = pd.to_numeric(clin.get("PFS_EVENT", np.nan), errors="coerce")
    c.reset_index().to_csv(fr / "Figure2C_clinical_summary.csv", index=False)

    # Figure2D / S1A
    miss.to_csv(fr / "Figure2D_missingness_heatmap.csv", index=False)
    miss.to_csv(fr / "Supplementary_Figure_S1A_qc_thresholds.csv", index=False)

    # Figure2E / S1C
    varsum.to_csv(fr / "Figure2E_variance_distribution.csv", index=False)
    varsum.to_csv(fr / "Supplementary_Figure_S1C_feature_filtering.csv", index=False)

    # Figure2I
    top_mut = mut.sum(axis=0).sort_values(ascending=False).head(30).rename("count").reset_index()
    top_mut.columns = ["gene", "count"]
    top_mut.to_csv(fr / "Figure2I_top_mutation_frequencies.csv", index=False)

    # S1D
    evt = pd.DataFrame(
        {
            "PATIENT_ID": clin.index,
            "OS_TIME": pd.to_numeric(clin.get("OS_TIME", np.nan), errors="coerce"),
            "OS_EVENT": pd.to_numeric(clin.get("OS_EVENT", np.nan), errors="coerce"),
            "PFS_TIME": pd.to_numeric(clin.get("PFS_TIME", np.nan), errors="coerce"),
            "PFS_EVENT": pd.to_numeric(clin.get("PFS_EVENT", np.nan), errors="coerce"),
        }
    )
    evt.to_csv(fr / "Supplementary_Figure_S1D_event_distribution.csv", index=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCGA-PAAD preprocessing pipeline")
    p.add_argument("--input-dir", default="./pdac_project/data_raw/tcga_paad")
    p.add_argument("--output-dir", default="./pdac_project/results/01_tcga_prep")
    p.add_argument("--gap-report", default="./pdac_project/results/_run_logs/gap_report.md")
    p.add_argument("--auto-download", action="store_true")

    p.add_argument("--clinical-patient", default=None)
    p.add_argument("--clinical-sample", default=None)
    p.add_argument("--expression", default=None)
    p.add_argument("--methylation", default=None)
    p.add_argument("--mutation", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_inputs(args: argparse.Namespace, roots: Sequence[Path], logger: logging.Logger) -> Dict[str, Path]:
    found: Dict[str, Optional[Path]] = {
        "clinical_patient": Path(args.clinical_patient) if args.clinical_patient else None,
        "clinical_sample": Path(args.clinical_sample) if args.clinical_sample else None,
        "expression": Path(args.expression) if args.expression else None,
        "methylation": Path(args.methylation) if args.methylation else None,
        "mutation": Path(args.mutation) if args.mutation else None,
    }

    if found["clinical_patient"] is None:
        found["clinical_patient"] = find_file(["data_clinical_patient", "clinical_patient"], roots)
    if found["clinical_sample"] is None:
        found["clinical_sample"] = find_file(["data_clinical_sample", "clinical_sample"], roots)
    if found["expression"] is None:
        found["expression"] = find_file(["data_mrna_seq", "mrna_seq", "rna_seq", "expression"], roots)
    if found["methylation"] is None:
        found["methylation"] = find_file(["data_methylation", "methylation"], roots)
    if found["mutation"] is None:
        found["mutation"] = find_file(["data_mutations", "mutations", "maf"], roots)

    missing = [k for k, v in found.items() if (k != "clinical_sample" and v is None)]
    if missing:
        raise FileNotFoundError(f"Missing required raw files: {missing}")

    out = {k: v for k, v in found.items() if v is not None}
    for k, v in out.items():
        logger.info("Using %s: %s", k, v)
    return out


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gap_report = Path(args.gap_report)
    logger = setup_logger(out_dir / "prep_run.log")

    sample_log_rows: List[Dict[str, object]] = []

    try:
        input_dir = Path(args.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)

        if args.auto_download:
            try:
                download_cbio_study(input_dir, logger)
            except Exception as exc:
                write_gap(gap_report, f"Automatic TCGA-PAAD download failed: {exc}")
                logger.warning("Auto-download failed; continuing with locally available files.")

        roots = [input_dir, input_dir.parent]
        files = resolve_inputs(args, roots, logger)

        clin_patient = read_table(files["clinical_patient"])
        clin_sample = read_table(files["clinical_sample"]) if "clinical_sample" in files else None
        expr_raw = read_table(files["expression"])
        meth_raw = read_table(files["methylation"])
        mut_raw = read_table(files["mutation"])

        clin = clinical_merge(clin_patient, clin_sample)
        sample_log_rows.append({"step": "clinical_merge", "n_patients": int(clin.shape[0]), "n_features": int(clin.shape[1])})

        expr = process_expression(expr_raw)
        sample_log_rows.append({"step": "expression_process", "n_patients": int(expr.shape[0]), "n_features": int(expr.shape[1])})

        meth = process_methylation(meth_raw, input_dir, gap_report, logger)
        sample_log_rows.append({"step": "methylation_process", "n_patients": int(meth.shape[0]), "n_features": int(meth.shape[1])})

        mut, driver_mat, mut_features = process_mutation(mut_raw)
        sample_log_rows.append({"step": "mutation_process", "n_patients": int(mut.shape[0]), "n_features": int(mut.shape[1])})

        avail = modality_availability(expr, meth, mut, clin)
        avail.to_csv(out_dir / "modality_availability.csv", index=False)

        common = sorted(set(expr.index) & set(meth.index) & set(mut.index) & set(clin.index))
        if len(common) < 20:
            raise RuntimeError(f"Insufficient integrated patients after intersection: {len(common)}")

        sample_log_rows.append({"step": "final_intersection", "n_patients": int(len(common)), "n_features": np.nan})

        expr_a = expr.reindex(common)
        meth_a = meth.reindex(common)
        mut_a = mut.reindex(common).fillna(0).astype(np.int8)
        clin_a = clin.reindex(common)
        mut_features_a = mut_features.reindex(common).fillna(0)

        # Core required outputs
        write_matrix(expr_a, out_dir, "expression_matrix")
        write_matrix(meth_a, out_dir, "methylation_matrix")
        write_matrix(mut_a, out_dir, "mutation_matrix")
        write_matrix(mut_features_a, out_dir, "mutation_driver_features")
        write_matrix(clin_a, out_dir, "clinical_matrix")

        miss = make_missingness(expr_a, meth_a, mut_a, clin_a)
        miss.to_csv(out_dir / "missingness_matrix.csv", index=False)

        varsum = variance_summary(expr_a, meth_a, mut_a)
        varsum.to_csv(out_dir / "variance_summary.csv", index=False)

        sample_log = pd.DataFrame(sample_log_rows)
        sample_log.to_csv(out_dir / "sample_filter_log.csv", index=False)

        figure_ready_exports(out_dir, avail, miss, varsum, clin_a, mut_a, sample_log)

        prep_summary = {
            "n_patients_final": int(len(common)),
            "expression_shape": [int(expr_a.shape[0]), int(expr_a.shape[1])],
            "methylation_shape": [int(meth_a.shape[0]), int(meth_a.shape[1])],
            "mutation_shape": [int(mut_a.shape[0]), int(mut_a.shape[1])],
            "clinical_shape": [int(clin_a.shape[0]), int(clin_a.shape[1])],
            "os_event_non_null": int(pd.to_numeric(clin_a["OS_EVENT"], errors="coerce").notna().sum()) if "OS_EVENT" in clin_a.columns else 0,
            "pfs_event_non_null": int(pd.to_numeric(clin_a["PFS_EVENT"], errors="coerce").notna().sum()) if "PFS_EVENT" in clin_a.columns else 0,
            "used_inputs": {k: str(v) for k, v in files.items()},
        }
        (out_dir / "prep_summary.json").write_text(json.dumps(prep_summary, indent=2), encoding="utf-8")

        logger.info("Preprocessing completed successfully: %s", prep_summary)
        return 0

    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        write_gap(gap_report, f"Step 01 preprocessing failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
