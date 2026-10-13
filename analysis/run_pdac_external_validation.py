#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def log_setup(p: Path) -> logging.Logger:
    lg = logging.getLogger("pdac_ext"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fm = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fm); lg.addHandler(sh)
    fh = logging.FileHandler(p, mode="w", encoding="utf-8"); fh.setFormatter(fm); lg.addHandler(fh)
    return lg


def gap_add(g: Path, s: str) -> None:
    g.parent.mkdir(parents=True, exist_ok=True)
    with g.open("a", encoding="utf-8") as f:
        f.write(s.rstrip() + "\n")


def read_df(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, index_col=0)
    if "PATIENT_ID" in df.columns and (df.index.name is None or df.index.name.upper() != "PATIENT_ID"):
        df = df.set_index("PATIENT_ID")
    df.index = df.index.astype(str)
    return df


def normalize_gene_name(g: str) -> str:
    s = str(g).strip().strip('"').strip("'")
    s = re.sub(r"\.\d+$", "", s)
    s = re.sub(r"\s+", "", s)
    return s.upper()


def normalize_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [normalize_gene_name(c) for c in df.columns]
    out = df.copy()
    out.columns = cols
    # Keep first occurrence if duplicated after normalization.
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
    return out


def load_local_cohort(cohort: str, root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    d = root / cohort
    if not d.exists():
        raise FileNotFoundError(f"{d} not found")
    expr_candidates = list(d.glob("expression_processed.*")) + list(d.glob("*expression*.csv")) + list(d.glob("*expression*.parquet"))
    clin_candidates = list(d.glob("clinical_processed.*")) + list(d.glob("*clinical*.csv")) + list(d.glob("*clinical*.parquet"))
    if expr_candidates:
        x = read_df(expr_candidates[0]).apply(pd.to_numeric, errors="coerce")
        probe_like = pd.Index(x.columns.astype(str)).str.match(r"^[0-9]+(_[A-Z])?_AT$", case=False).mean() > 0.5
        gz = d / f"{cohort}_series_matrix.txt.gz"
        if probe_like and gz.exists():
            try:
                x, c = parse_geo_series(gz)
                x.to_csv(d / "expression_processed.csv")
                c.to_csv(d / "clinical_processed.csv")
                return x, c, f"re-parsed probe-level local series matrix from {gz}"
            except Exception:
                pass
        if clin_candidates:
            c = read_df(clin_candidates[0]).copy()
        else:
            c = pd.DataFrame(index=x.index)
        if "OS_TIME" not in c.columns:
            c["OS_TIME"] = np.nan
        if "OS_EVENT" not in c.columns:
            c["OS_EVENT"] = np.nan
        c = c.loc[x.index.intersection(c.index)] if len(c.index.intersection(x.index)) else c.reindex(x.index)
        c = c.reindex(x.index)
        c.index.name = "PATIENT_ID"
        return x, c, f"loaded local processed files from {d}"

    gz = d / f"{cohort}_series_matrix.txt.gz"
    if gz.exists():
        try:
            x, c = parse_geo_series(gz)
            x.to_csv(d / "expression_processed.csv")
            c.to_csv(d / "clinical_processed.csv")
            return x, c, f"parsed local series matrix from {gz}"
        except Exception:
            pass
    if not expr_candidates:
        raise FileNotFoundError(f"{cohort} local expression file missing")
    x = read_df(expr_candidates[0]).apply(pd.to_numeric, errors="coerce")
    if clin_candidates:
        c = read_df(clin_candidates[0]).copy()
    else:
        c = pd.DataFrame(index=x.index)
    if "OS_TIME" not in c.columns:
        c["OS_TIME"] = np.nan
    if "OS_EVENT" not in c.columns:
        c["OS_EVENT"] = np.nan
    c = c.loc[x.index.intersection(c.index)] if len(c.index.intersection(x.index)) else c.reindex(x.index)
    c = c.reindex(x.index)
    c.index.name = "PATIENT_ID"
    return x, c, f"loaded local files from {d}"


def dl(urls: List[str], out: Path) -> Tuple[bool, str]:
    import requests
    out.parent.mkdir(parents=True, exist_ok=True)
    err = []
    for u in urls:
        try:
            r = requests.get(u, timeout=120)
            if r.status_code == 200 and len(r.content) > 200:
                out.write_bytes(r.content); return True, u
            err.append(f"{u}:{r.status_code}")
        except Exception as e:
            err.append(f"{u}:{e}")
    return False, "; ".join(err)


def gpl_prefix(gpl: str) -> str:
    d = re.sub(r"[^0-9]", "", gpl)
    if len(d) <= 3:
        return "GPLnnn"
    return f"GPL{d[:-3]}nnn"


def load_gpl_mapping(gpl: str, cache_dir: Path) -> Dict[str, str]:
    annot_gz = cache_dir / f"{gpl}.annot.gz"
    soft_gz = cache_dir / f"{gpl}_family.soft.gz"
    if not annot_gz.exists() and not soft_gz.exists():
        pref = gpl_prefix(gpl)
        ok, _ = dl([f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{pref}/{gpl}/annot/{gpl}.annot.gz"], annot_gz)
        if not ok:
            ok, _ = dl([f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{pref}/{gpl}/soft/{gpl}_family.soft.gz"], soft_gz)
            if not ok:
                return {}
    def read_platform_table(gz_path: Path) -> pd.DataFrame:
        lines = []
        in_table = False
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                stripped = ln.rstrip("\n")
                low = stripped.lower()
                if low.startswith("!platform_table_begin"):
                    in_table = True
                    continue
                if low.startswith("!platform_table_end"):
                    break
                if in_table and stripped and not stripped.startswith(("#", "^", "!")) and "\t" in stripped:
                    lines.append(stripped)
        if not lines:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", low_memory=False)

    try:
        if soft_gz.exists() and not annot_gz.exists():
            df = read_platform_table(soft_gz)
        else:
            gz = annot_gz
            df = pd.read_csv(gz, sep="\t", comment="#", low_memory=False)
            if df.shape[1] < 2 or str(df.columns[0]).startswith("^"):
                df = read_platform_table(gz)
    except Exception:
        try:
            df = read_platform_table(soft_gz if soft_gz.exists() and not annot_gz.exists() else annot_gz)
        except Exception:
            return {}

    cols = {c.lower().strip(): c for c in df.columns}
    id_col = next((cols[k] for k in ["id", "id_ref", "probeid", "probe_id"] if k in cols), None)
    sym_col = next((cols[k] for k in ["gene symbol", "gene_symbol", "gene_symbol_assignment", "symbol", "gene"] if k in cols), None)
    if id_col is None or sym_col is None:
        return {}
    mp = {}
    for _, r in df[[id_col, sym_col]].dropna().iterrows():
        pid = str(r[id_col]).strip()
        sym = re.split(r"\s*///\s*|\s*//\s*|;", str(r[sym_col]).strip())[0].strip()
        if pid and sym and sym.upper() not in {"NA", "N/A", "---", "."}:
            mp[pid] = sym
    return mp


def parse_geo_os_time_months(blob: str) -> float:
    patterns = [
        (r"survival\s*months?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", 1.0),
        (r"overall\s*survival\s*months?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", 1.0),
        (r"os[\._\s-]*months?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", 1.0),
        (r"os[\._\s-]*years?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", 12.0),
        (r"survival\s*time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", 1.0),
        (r"(?:os|overall survival|survival)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", 1.0),
    ]
    for pat, mult in patterns:
        m = re.search(pat, blob)
        if m:
            return float(m.group(1)) * mult
    return np.nan


def parse_geo_os_event(blob: str) -> float:
    for pat in [
        r"survival\s*status\s*[:=]\s*([01])\b",
        r"os[\._\s-]*(?:event|status)\s*[:=]\s*([01])\b",
        r"death\s*[:=]\s*([01])\b",
        r"deceased\s*[:=]\s*([01])\b",
        r"event\s*[:=]\s*([01])\b",
    ]:
        m = re.search(pat, blob)
        if m:
            return float(m.group(1))
    if re.search(r"\b(deceased|dead|death)\b", blob):
        return 1.0
    if re.search(r"\b(alive|living|censored|censor)\b", blob):
        return 0.0
    return np.nan


def parse_geo_series(gz: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    txt = gzip.decompress(gz.read_bytes()).decode("utf-8", errors="ignore")
    lines = txt.splitlines(); sid = []; tb = []; on = False; chars: Dict[str, List[str]] = {}; gpls: List[str] = []
    for ln in lines:
        if ln.startswith("!Sample_geo_accession"):
            sid = [x.strip().strip('"') for x in ln.split("\t")[1:]]
            chars = {s: [] for s in sid}
        if ln.startswith("!Series_platform_id"):
            gpls = [x.strip().strip('"') for x in ln.split("\t")[1:] if x.strip().strip('"')]
        if ln.startswith("!Sample_characteristics_ch1"):
            vals = [x.strip().strip('"') for x in ln.split("\t")[1:]]
            for s, v in zip(sid, vals):
                chars.setdefault(s, []).append(v)
        if ln.strip() == "!series_matrix_table_begin": on = True; continue
        if ln.strip() == "!series_matrix_table_end": on = False; continue
        if on: tb.append(ln)
    if not tb: raise ValueError("series matrix table missing")
    m = pd.read_csv(io.StringIO("\n".join(tb)), sep="\t", low_memory=False)
    m.columns = [str(c).strip().strip('"') for c in m.columns]
    sc = [c for c in m.columns if c in sid] or m.columns[1:].tolist()
    gc = next((c for c in ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol"] if c in m.columns), m.columns[0])
    x = m[[gc] + sc].copy()
    if gc.lower() in {"id_ref", "id"}:
        mp = {}
        for gpl in gpls:
            mp.update(load_gpl_mapping(gpl, gz.parent))
        if mp:
            x[gc] = x[gc].astype(str).map(mp).fillna("")
    x[gc] = x[gc].astype(str).str.split("///").str[0].str.split(";").str[0].str.strip()
    x = x[(x[gc] != "") & (x[gc].str.upper() != "NA")].groupby(gc, as_index=True)[sc].mean(numeric_only=True).T
    x.index.name = "PATIENT_ID"
    c = pd.DataFrame(index=x.index); c["OS_TIME"] = np.nan; c["OS_EVENT"] = np.nan
    for s in c.index:
        blob = " | ".join(chars.get(s, [])).lower()
        c.loc[s, "OS_TIME"] = parse_geo_os_time_months(blob)
        c.loc[s, "OS_EVENT"] = parse_geo_os_event(blob)

    blobs = pd.Series({s: " | ".join(chars.get(s, [])).lower() for s in x.index})
    non_tumor = blobs.str.contains(r"adjacent|non[- ]?tumou?r|normal|benign", regex=True, na=False)
    tumor = blobs.str.contains(r"tumou?r|cancer|pdac|adenocarcinoma", regex=True, na=False) & ~non_tumor
    if tumor.any() and non_tumor.any():
        keep = tumor[tumor].index
        x = x.loc[keep]
        c = c.loc[keep]
    return x, c


def parse_cbio(path: Path) -> pd.DataFrame:
    ls = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    h = 0
    for i, ln in enumerate(ls[:40]):
        if ln.startswith("Hugo_Symbol") or ln.startswith("PATIENT_ID"): h = i; break
    return pd.read_csv(io.StringIO("\n".join(ls[h:])), sep="\t", low_memory=False)


def load_icgc(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = root / "ICGC_PACA_AU"; d.mkdir(parents=True, exist_ok=True)
    ef = d / "expr.txt"; cf = d / "clin.txt"
    bases = [
        "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/paad_au_icgc",
        "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/paca_au_icgc",
        "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/paad_qcmg_uq_2016",
        "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/paad_icgc",
    ]
    ok = False; e = ""
    for b in bases:
        ok, e = dl(
            [
                f"{b}/data_mrna_seq_fpkm_capture_Zscores.txt",
                f"{b}/data_mrna_seq_v2_rsem.txt",
                f"{b}/data_mrna_seq_fpkm.txt",
                f"{b}/data_mrna_seq_v2_rsem_zscores.txt",
            ],
            ef,
        )
        if ok:
            ok2, e2 = dl([f"{b}/data_clinical_patient.txt"], cf)
            if ok2:
                ok = True
                break
            e = f"expr ok at {b} but clinical failed: {e2}"
    if not ok:
        raise RuntimeError(f"ICGC expression/clinical download failed: {e}")
    et = parse_cbio(ef)
    gc = et.columns[0]; sc = et.columns[2:] if len(et.columns) > 2 else et.columns[1:]
    x = et[[gc] + list(sc)].groupby(gc, as_index=True)[list(sc)].mean(numeric_only=True).T
    x.index.name = "PATIENT_ID"; x.index = x.index.astype(str)
    ct = parse_cbio(cf); ct = ct.rename(columns={ct.columns[0]: "PATIENT_ID"}).set_index("PATIENT_ID")
    c = pd.DataFrame(index=ct.index.astype(str))
    u = {k.upper(): k for k in ct.columns}
    if "OS_MONTHS" in u: c["OS_TIME"] = pd.to_numeric(ct[u["OS_MONTHS"]], errors="coerce")
    else: c["OS_TIME"] = np.nan
    if "OS_STATUS" in u:
        s = ct[u["OS_STATUS"]].astype(str).str.lower()
        c["OS_EVENT"] = np.where(s.str.contains("deceased|dead|1"), 1, np.where(s.str.contains("alive|0"), 0, np.nan))
    else: c["OS_EVENT"] = np.nan
    return x, c


def subtype_centroids(tcga_x: pd.DataFrame, tcga_s: pd.Series) -> pd.DataFrame:
    z = (tcga_x - tcga_x.mean()) / tcga_x.std(ddof=0).replace(0, np.nan); z = z.fillna(0.0)
    c = z.groupby(tcga_s).mean(numeric_only=True); c.index = [f"Subtype_{int(i)}" for i in c.index]; return c


def _select_shared_genes(
    shared: List[str],
    tcga_matrix: Optional[pd.DataFrame],
    max_genes: int,
) -> List[str]:
    if max_genes <= 0 or len(shared) <= max_genes or tcga_matrix is None:
        return shared
    v = tcga_matrix[shared].var(axis=0, numeric_only=True).sort_values(ascending=False)
    keep = v.head(max_genes).index.astype(str).tolist()
    return keep


def subtype_transfer(
    x: pd.DataFrame,
    c: pd.DataFrame,
    min_g: int,
    tcga_matrix: Optional[pd.DataFrame] = None,
    max_genes: int = 3000,
) -> pd.DataFrame:
    z = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan); z = z.fillna(0.0)
    g = sorted(set(z.columns) & set(c.columns))
    if len(g) < min_g: raise ValueError(f"shared genes too few for subtype transfer: {len(g)}")
    g = _select_shared_genes(g, tcga_matrix, max_genes=max_genes)
    rows = []
    for sid in z.index:
        s = z.loc[sid, g].to_numpy(dtype=float); rs = {}
        for cc in c.index:
            t = c.loc[cc, g].to_numpy(dtype=float); r = np.corrcoef(s, t)[0, 1]; rs[str(cc)] = float(r) if np.isfinite(r) else np.nan
        best = pd.Series(rs).idxmax(); rows.append({"PATIENT_ID": sid, "assigned_subtype": best, **rs})
    return pd.DataFrame(rows)


def student_predict(
    tcga_x: pd.DataFrame,
    tcga_r: pd.Series,
    ext_x: pd.DataFrame,
    min_g: int,
    seed: int,
    max_genes: int = 3000,
    cv_folds: int = 3,
) -> pd.Series:
    g = sorted(set(tcga_x.columns) & set(ext_x.columns))
    if len(g) < min_g: raise ValueError(f"shared genes too few for student model: {len(g)}")
    g = _select_shared_genes(g, tcga_x, max_genes=max_genes)
    xtr = tcga_x[g].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(tcga_r, errors="coerce"); ok = y.notna(); xtr = xtr.loc[ok]; y = y.loc[ok]
    sc = StandardScaler(); xtr_s = sc.fit_transform(xtr.to_numpy(dtype=float)); xte_s = sc.transform(ext_x[g].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float))
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50, cv=max(2, cv_folds), max_iter=4000, random_state=seed); en.fit(xtr_s, y.to_numpy(dtype=float))
    return pd.Series(en.predict(xte_s), index=ext_x.index, name="student_risk_score")


def eval_surv(pred: pd.DataFrame, clin: pd.DataFrame, cutoff: float, tr_time: np.ndarray, tr_event: np.ndarray) -> Tuple[Dict[str, float], pd.DataFrame]:
    out = {"n_survival": 0}; cal = pd.DataFrame(columns=["time_months", "group", "n", "observed_event_rate", "predicted_score_mean"])
    if not {"OS_TIME", "OS_EVENT"}.issubset(clin.columns): return out, cal
    d = pred.join(clin[["OS_TIME", "OS_EVENT"]], how="inner")
    d["OS_TIME"] = pd.to_numeric(d["OS_TIME"], errors="coerce"); d["OS_EVENT"] = pd.to_numeric(d["OS_EVENT"], errors="coerce")
    d = d.dropna(subset=["student_risk_score", "OS_TIME", "OS_EVENT"]); d = d[(d["OS_TIME"] > 0) & (d["OS_EVENT"].isin([0, 1]))]
    out["n_survival"] = int(d.shape[0]); 
    if d.shape[0] < 25: return out, cal
    d["risk_group_locked"] = np.where(d["student_risk_score"] >= cutoff, "High", "Low")
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index
    from lifelines import CoxPHFitter
    h = d[d["risk_group_locked"] == "High"]; l = d[d["risk_group_locked"] == "Low"]
    if h.shape[0] >= 5 and l.shape[0] >= 5:
        out["logrank_p"] = float(logrank_test(h["OS_TIME"], l["OS_TIME"], event_observed_A=h["OS_EVENT"], event_observed_B=l["OS_EVENT"]).p_value)
    out["c_index"] = float(concordance_index(d["OS_TIME"], -d["student_risk_score"], d["OS_EVENT"]))
    try:
        cph = CoxPHFitter(); cph.fit(d[["OS_TIME", "OS_EVENT", "student_risk_score"]], duration_col="OS_TIME", event_col="OS_EVENT")
        ci = cph.confidence_intervals_.loc["student_risk_score"]; out["hr"] = float(np.exp(cph.params_["student_risk_score"]))
        out["hr_ci_low"] = float(np.exp(ci.iloc[0])); out["hr_ci_high"] = float(np.exp(ci.iloc[1])); out["hr_p"] = float(cph.summary.loc["student_risk_score", "p"])
    except Exception: pass
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
        ytr = Surv.from_arrays(event=tr_event.astype(bool), time=tr_time.astype(float))
        yte = Surv.from_arrays(event=d["OS_EVENT"].to_numpy(dtype=int).astype(bool), time=d["OS_TIME"].to_numpy(dtype=float))
        ts = np.array([12.0, 24.0, 36.0]); ts = ts[ts < float(d["OS_TIME"].max())]
        if ts.size:
            au, _ = cumulative_dynamic_auc(ytr, yte, d["student_risk_score"].to_numpy(dtype=float), ts)
            for t, a in zip(ts, au): out[f"auc_{int(t)}m"] = float(a)
    except Exception: pass
    q = pd.qcut(d["student_risk_score"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    ns = (d["student_risk_score"] - d["student_risk_score"].min()) / (d["student_risk_score"].max() - d["student_risk_score"].min() + 1e-8)
    rr = []
    for t in [12.0, 24.0, 36.0]:
        for g in sorted(q.dropna().unique()):
            m = q == g; s = d[m]
            rr.append({"time_months": t, "group": str(g), "n": int(s.shape[0]), "observed_event_rate": float(((s["OS_EVENT"] == 1) & (s["OS_TIME"] <= t)).mean()), "predicted_score_mean": float(ns[m].mean())})
    return out, pd.DataFrame(rr)


def km_plot(pred: pd.DataFrame, clin: pd.DataFrame, cutoff: float, out: Path, title: str) -> None:
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except Exception:
        return
    d = pred.join(clin[["OS_TIME", "OS_EVENT"]], how="inner")
    d["OS_TIME"] = pd.to_numeric(d["OS_TIME"], errors="coerce"); d["OS_EVENT"] = pd.to_numeric(d["OS_EVENT"], errors="coerce")
    d = d.dropna(subset=["student_risk_score", "OS_TIME", "OS_EVENT"]); d = d[(d["OS_TIME"] > 0) & (d["OS_EVENT"].isin([0, 1]))]
    if d.shape[0] < 20: return
    d["g"] = np.where(d["student_risk_score"] >= cutoff, "High", "Low"); h = d[d["g"] == "High"]; l = d[d["g"] == "Low"]
    if h.shape[0] < 3 or l.shape[0] < 3: return
    p = float(logrank_test(h["OS_TIME"], l["OS_TIME"], event_observed_A=h["OS_EVENT"], event_observed_B=l["OS_EVENT"]).p_value)
    kmf = KaplanMeierFitter(); plt.figure(figsize=(7.2, 5.4))
    kmf.fit(h["OS_TIME"], event_observed=h["OS_EVENT"], label=f"High (n={h.shape[0]})"); kmf.plot(ci_show=False, color="#D6604D")
    kmf.fit(l["OS_TIME"], event_observed=l["OS_EVENT"], label=f"Low (n={l.shape[0]})"); kmf.plot(ci_show=False, color="#4393C3")
    plt.title(title); plt.xlabel("Months"); plt.ylabel("Survival probability"); plt.text(0.05, 0.08, f"log-rank P={p:.3g}", transform=plt.gca().transAxes)
    plt.tight_layout(); plt.savefig(out, dpi=320); plt.close()


def harmonize(tcga_x: pd.DataFrame, ext: Dict[str, pd.DataFrame], out: Path, seed: int, max_genes: int = 2000, skip_umap: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mats = {"TCGA": tcga_x}; mats.update({k: v for k, v in ext.items() if v is not None and not v.empty})
    if len(mats) < 2:
        pd.DataFrame({"note": ["<2 cohorts"]}).to_csv(out / "harmonization_status.csv", index=False); return
    g = None
    for v in mats.values(): g = set(v.columns.astype(str)) if g is None else (g & set(v.columns.astype(str)))
    g = sorted(g or [])
    if len(g) < 200:
        pd.DataFrame({"note": [f"common genes={len(g)}"]}).to_csv(out / "harmonization_status.csv", index=False); return
    if max_genes > 0 and len(g) > max_genes:
        top = tcga_x[g].var(axis=0, numeric_only=True).sort_values(ascending=False).head(max_genes).index.astype(str).tolist()
        g = top
    xs = []; ms = []
    for c, v in mats.items():
        z = (v[g] - v[g].mean()) / v[g].std(ddof=0).replace(0, np.nan); z = z.fillna(0.0); xs.append(z)
        ms.extend([{"PATIENT_ID": i, "cohort": c} for i in z.index])
    xb = pd.concat(xs, axis=0); meta = pd.DataFrame(ms).set_index("PATIENT_ID").loc[xb.index]
    xa = xb.copy()
    for c in meta["cohort"].unique():
        idx = meta.index[meta["cohort"] == c]; xa.loc[idx] = xa.loc[idx] - xa.loc[idx].mean(axis=0)
    xb.to_csv(out / "common_gene_merged_matrix_before.csv"); xa.to_csv(out / "common_gene_merged_matrix_after.csv")
    for tag, x in [("before", xb), ("after", xa)]:
        p = PCA(n_components=2, random_state=seed).fit_transform(x.to_numpy(dtype=float))
        pdp = pd.DataFrame({"PATIENT_ID": x.index, "PC1": p[:, 0], "PC2": p[:, 1], "cohort": meta.loc[x.index, "cohort"].to_numpy()})
        pdp.to_csv(out / f"pca_{tag}.csv", index=False)
        if skip_umap:
            u = p
        else:
            try:
                import umap
                u = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(x.to_numpy(dtype=float))
            except Exception:
                u = p
        um = pd.DataFrame({"PATIENT_ID": x.index, "UMAP1": u[:, 0], "UMAP2": u[:, 1], "cohort": meta.loc[x.index, "cohort"].to_numpy()})
        um.to_csv(out / f"umap_{tag}.csv", index=False)
        for name, df, x1, x2 in [(f"pca_{tag}", pdp, "PC1", "PC2"), (f"umap_{tag}", um, "UMAP1", "UMAP2")]:
            plt.figure(figsize=(7.4, 5.8)); sns.scatterplot(data=df, x=x1, y=x2, hue="cohort", s=26, linewidth=0)
            plt.tight_layout(); plt.savefig(out / f"{name}.pdf"); plt.savefig(out / f"{name}.png", dpi=320); plt.close()


def run(a: argparse.Namespace) -> int:
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    lg = log_setup(out / "external_validation.log"); gap = Path(a.gap_report)
    try:
        if plt is None or sns is None: raise ImportError("matplotlib/seaborn missing")
        tcga_x = read_df(Path(a.tcga_expression)).apply(pd.to_numeric, errors="coerce")
        tcga_x = normalize_gene_columns(tcga_x)
        tcga_c = read_df(Path(a.tcga_clinical)); tcga_r = read_df(Path(a.tcga_risk)); tcga_s = pd.read_csv(a.tcga_subtypes).set_index("PATIENT_ID")["subtype"].astype(int)
        if a.risk_column and a.risk_column in tcga_r.columns:
            rr = a.risk_column
        elif a.risk_column and a.risk_column.lower() == "auto_best":
            if "Expr-Penalized-Cox" in tcga_r.columns:
                rr = "Expr-Penalized-Cox"
            elif "LateFusion-Cox" in tcga_r.columns:
                rr = "LateFusion-Cox"
            else:
                rr = tcga_r.columns[0]
        else:
            rr = "LateFusion-Cox" if "LateFusion-Cox" in tcga_r.columns else tcga_r.columns[0]
        lg.info("Using risk column: %s", rr)
        tcga_rv = pd.to_numeric(tcga_r[rr], errors="coerce")
        cm = sorted(set(tcga_x.index) & set(tcga_c.index) & set(tcga_rv.index) & set(tcga_s.index))
        tcga_x = tcga_x.loc[cm]; tcga_c = tcga_c.loc[cm]; tcga_rv = tcga_rv.loc[cm]; tcga_s = tcga_s.loc[cm]
        tr_t = pd.to_numeric(tcga_c["OS_TIME"], errors="coerce").fillna(0).to_numpy(dtype=float); tr_e = pd.to_numeric(tcga_c["OS_EVENT"], errors="coerce").fillna(0).to_numpy(dtype=int)
        cuts = json.loads(Path(a.locked_cutoffs).read_text(encoding="utf-8"))
        cutoff = float(cuts.get("LateFusion-Cox", {}).get("median", float(tcga_rv.median(skipna=True))))
        cent = subtype_centroids(tcga_x, tcga_s)
        ext_root = Path(a.external_data_dir); ext_root.mkdir(parents=True, exist_ok=True)
        rows = []; forest = []; ext_expr = {}
        cohorts = [c.strip() for c in str(a.cohorts).split(",") if c.strip()]
        for coh in cohorts:
            lg.info("External cohort: %s", coh)
            try:
                try:
                    x, c, src = load_local_cohort(coh, ext_root)
                except Exception:
                    if coh.startswith("GSE"):
                        pref = coh[:-3] + "nnn"; gz = ext_root / coh / f"{coh}_series_matrix.txt.gz"
                        ok, src = dl(
                            [
                                f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pref}/{coh}/matrix/{coh}_series_matrix.txt.gz",
                                f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={coh}&format=file&file={coh}_series_matrix.txt.gz",
                            ],
                            gz,
                        )
                        if not ok:
                            raise RuntimeError(f"download failed: {src}")
                        x, c = parse_geo_series(gz)
                    else:
                        x, c = load_icgc(ext_root)
                        src = "downloaded ICGC from cBio datahub"
                (ext_root / coh).mkdir(parents=True, exist_ok=True)
                x = normalize_gene_columns(x.apply(pd.to_numeric, errors="coerce"))
                x.to_csv(ext_root / coh / "expression_processed.csv")
                c.to_csv(ext_root / coh / "clinical_processed.csv")
                x.to_csv(out / f"{coh}_expression_processed.csv")
                c.to_csv(out / f"{coh}_clinical_processed.csv")
                st = subtype_transfer(
                    x,
                    cent,
                    a.min_shared_genes_subtype,
                    tcga_matrix=tcga_x,
                    max_genes=a.max_shared_genes_subtype,
                )
                st.to_csv(out / f"subtype_transfer_{coh}.csv", index=False)
                pr = student_predict(
                    tcga_x,
                    tcga_rv,
                    x,
                    a.min_shared_genes_student,
                    a.seed,
                    max_genes=a.max_shared_genes_student,
                    cv_folds=a.student_cv_folds,
                )
                pdf = pd.DataFrame({"PATIENT_ID": pr.index, "student_risk_score": pr.to_numpy(dtype=float)}).set_index("PATIENT_ID")
                pdf["risk_group_locked"] = np.where(pdf["student_risk_score"] >= cutoff, "High", "Low")
                pdf.reset_index().to_csv(out / f"student_model_predictions_{coh}.csv", index=False)
                met, cal = eval_surv(pdf, c, cutoff, tr_t, tr_e)
                if not cal.empty: cal.insert(0, "cohort", coh); cal.to_csv(out / f"calibration_external_{coh}.csv", index=False)
                else: pd.DataFrame(columns=["cohort", "time_months", "group", "n", "observed_event_rate", "predicted_score_mean"]).to_csv(out / f"calibration_external_{coh}.csv", index=False)
                km_plot(pdf, c, cutoff, fig / f"km_{coh}.png", f"{coh} (locked cutoff)")
                r = {
                    "cohort": coh,
                    "status": "ok",
                    "note": src,
                    "n_expression": int(x.shape[0]),
                    "n_genes": int(x.shape[1]),
                    "locked_cutoff_used": True,
                    "locked_cutoff_value": cutoff,
                }
                r.update(met); rows.append(r); ext_expr[coh] = x
                if np.isfinite(r.get("hr", np.nan)):
                    forest.append({"cohort": coh, "hr": r.get("hr"), "hr_ci_low": r.get("hr_ci_low"), "hr_ci_high": r.get("hr_ci_high"), "p_value": r.get("hr_p", np.nan)})
                if coh == "GSE71729": gap_add(gap, "- GSE71729: primary-tumor-only selection unavailable from downloadable metadata; used all retained samples.")
                if int(r.get("n_survival", 0)) == 0: gap_add(gap, f"- {coh}: survival fields unavailable; skipped KM/HR/C-index/AUC.")
            except Exception as e:
                gap_add(gap, f"- {coh} failed: {e}")
                rows.append({"cohort": coh, "status": "failed", "note": str(e), "n_expression": 0, "n_genes": 0, "n_survival": 0, "locked_cutoff_used": True, "locked_cutoff_value": cutoff})
                pd.DataFrame(columns=["PATIENT_ID", "student_risk_score", "risk_group_locked"]).to_csv(out / f"student_model_predictions_{coh}.csv", index=False)
                pd.DataFrame(columns=["PATIENT_ID", "assigned_subtype"]).to_csv(out / f"subtype_transfer_{coh}.csv", index=False)
                pd.DataFrame(columns=["cohort", "time_months", "group", "n", "observed_event_rate", "predicted_score_mean"]).to_csv(out / f"calibration_external_{coh}.csv", index=False)
                ext_expr[coh] = pd.DataFrame()
        es = pd.DataFrame(rows); es.to_csv(out / "external_summary.csv", index=False)
        fp = pd.DataFrame(forest); fp.to_csv(out / "forest_plot_table.csv", index=False)
        if not fp.empty:
            fr = fp.copy(); fr["log_hr"] = np.log(fr["hr"]); fr["se"] = (np.log(fr["hr_ci_high"]) - np.log(fr["hr_ci_low"])) / (2 * 1.96)
            fr = fr[(fr["se"] > 0) & np.isfinite(fr["se"]) & np.isfinite(fr["log_hr"])]
            if not fr.empty:
                w = 1.0 / (fr["se"] ** 2); pl = float(np.sum(w * fr["log_hr"]) / np.sum(w)); se = float(np.sqrt(1.0 / np.sum(w)))
                meta = pd.DataFrame([{"model": "fixed_effect", "n_cohorts": int(fr.shape[0]), "pooled_hr": float(np.exp(pl)), "ci_low": float(np.exp(pl - 1.96 * se)), "ci_high": float(np.exp(pl + 1.96 * se))}])
            else:
                meta = pd.DataFrame(columns=["model", "n_cohorts", "pooled_hr", "ci_low", "ci_high"])
        else:
            meta = pd.DataFrame(columns=["model", "n_cohorts", "pooled_hr", "ci_low", "ci_high"])
        meta.to_csv(out / "meta_summary.csv", index=False)
        harmonize(tcga_x, ext_expr, out / "harmonization_visuals", a.seed, max_genes=a.harmonize_max_genes, skip_umap=a.skip_umap)
        lg.info("External validation complete."); return 0
    except Exception as e:
        lg.exception("External validation failed: %s", e); gap_add(gap, f"- External validation hard failure: {e}"); return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tcga-expression", required=True); p.add_argument("--tcga-clinical", required=True)
    p.add_argument("--tcga-risk", required=True); p.add_argument("--tcga-subtypes", required=True)
    p.add_argument("--locked-cutoffs", required=True); p.add_argument("--external-data-dir", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--gap-report", required=True)
    p.add_argument("--risk-column", default="auto_best", help="Risk score column in tcga-risk table or auto_best.")
    p.add_argument("--cohorts", default="GSE62452,GSE71729,GSE85916,ICGC_PACA_AU", help="Comma-separated cohort IDs.")
    p.add_argument("--min-shared-genes-student", type=int, default=500); p.add_argument("--min-shared-genes-subtype", type=int, default=120)
    p.add_argument("--max-shared-genes-student", type=int, default=3000)
    p.add_argument("--max-shared-genes-subtype", type=int, default=3000)
    p.add_argument("--student-cv-folds", type=int, default=3)
    p.add_argument("--harmonize-max-genes", type=int, default=2000)
    p.add_argument("--skip-umap", action="store_true", help="Skip UMAP in harmonization visuals and reuse PCA coordinates.")
    p.add_argument("--seed", type=int, default=42); return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
