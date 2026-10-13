#!/usr/bin/env python3
"""
Prompt 2: PDAC benchmark suite (nested cross-validation).

Models:
1) Clinical Cox
2) Expression penalized Cox
3) Expression DeepSurv
4) Late-fusion multi-omics Cox
5) Late-fusion multi-omics DeepSurv
6) Gated multimodal survival network

Outputs:
- fold-level metrics (C-index, IBS, time-dependent AUC)
- summary tables
- plots
- frozen best main model + best simple model
- integrated representation for subtype discovery
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:
    joblib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pdac_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def read_matrix(base_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = base_dir / f"{stem}.parquet"
    csv_path = base_dir / f"{stem}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        raise FileNotFoundError(f"Missing required file: {parquet_path} or {csv_path}")

    if df.empty:
        raise ValueError(f"Loaded empty matrix for {stem}")

    if "PATIENT_ID" in df.columns and (df.index.name is None or df.index.name.upper() != "PATIENT_ID"):
        df = df.set_index("PATIENT_ID")

    df.index = df.index.astype(str)
    return df


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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
    adj = np.minimum.accumulate(adj[order[::-1]])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def parse_eval_horizons_months(value: str) -> np.ndarray:
    vals = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return np.asarray(vals, dtype=float)


def choose_eval_times(
    train_time: np.ndarray,
    train_event: np.ndarray,
    n_times: int = 20,
    requested_times: Optional[np.ndarray] = None,
) -> np.ndarray:
    if requested_times is not None and requested_times.size:
        finite = requested_times[np.isfinite(requested_times)]
        if finite.size:
            max_train = np.nanmax(train_time)
            return np.unique(finite[(finite > 0) & (finite < max_train)]).astype(float)

    evt_times = np.asarray(train_time)[np.asarray(train_event).astype(bool)]
    evt_times = evt_times[np.isfinite(evt_times)]
    evt_times = np.sort(np.unique(evt_times))
    if evt_times.size < 5:
        return np.array([], dtype=float)

    q = np.linspace(0.1, 0.9, num=min(n_times, evt_times.size))
    times = np.quantile(evt_times, q)
    times = np.unique(times)
    return times.astype(float)


def safe_standardize_risk(x: pd.Series) -> pd.Series:
    std = x.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std


def dump_artifact(obj: Any, path: Path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
        return
    with path.open("wb") as f:
        pickle.dump(obj, f)


# -----------------------------
# Data loading / feature builders
# -----------------------------

@dataclass
class DataBundle:
    patient_ids: List[str]
    expr: pd.DataFrame
    meth: pd.DataFrame
    mut: pd.DataFrame
    clin: pd.DataFrame
    time: np.ndarray
    event: np.ndarray


def load_prepared_data(input_dir: Path, endpoint: str, logger: logging.Logger) -> DataBundle:
    expr = to_numeric_df(read_matrix(input_dir, "expression_matrix"))
    meth = to_numeric_df(read_matrix(input_dir, "methylation_matrix"))
    mut = to_numeric_df(read_matrix(input_dir, "mutation_matrix"))
    clin = read_matrix(input_dir, "clinical_matrix")

    endpoint = endpoint.upper().strip()
    time_col = f"{endpoint}_TIME"
    event_col = f"{endpoint}_EVENT"
    if time_col not in clin.columns or event_col not in clin.columns:
        raise ValueError(f"Clinical matrix must contain {time_col} and {event_col} columns.")

    common = sorted(set(expr.index) & set(meth.index) & set(mut.index) & set(clin.index))
    if not common:
        raise ValueError("No shared patients across expression, methylation, mutation, and clinical matrices.")

    expr = expr.reindex(common)
    meth = meth.reindex(common)
    mut = mut.reindex(common).fillna(0)
    clin = clin.reindex(common)

    time = pd.to_numeric(clin[time_col], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(clin[event_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(time) & np.isfinite(event) & (time > 0)
    valid &= np.isin(event.astype(int), [0, 1])

    n_drop = int((~valid).sum())
    if n_drop > 0:
        logger.warning("Dropping %d patients with invalid survival labels.", n_drop)

    keep_ids = np.asarray(common)[valid].tolist()
    expr = expr.loc[keep_ids]
    meth = meth.loc[keep_ids]
    mut = mut.loc[keep_ids]
    clin = clin.loc[keep_ids]
    time = time[valid]
    event = event[valid].astype(int)

    logger.info("Prepared dataset: n=%d, expr=%s, meth=%s, mut=%s", len(keep_ids), expr.shape, meth.shape, mut.shape)
    return DataBundle(
        patient_ids=keep_ids,
        expr=expr,
        meth=meth,
        mut=mut,
        clin=clin,
        time=time,
        event=event,
    )


class ClinicalTransformer:
    def __init__(self) -> None:
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.dummy_cols: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    @staticmethod
    def _drop_survival_columns(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = []
        for col in df.columns:
            c = col.upper()
            if c.endswith("_TIME") or c.endswith("_EVENT"):
                drop_cols.append(col)
            elif c.startswith(("OS_", "PFS_", "PFI_", "DFS_", "DSS_", "RFS_")):
                drop_cols.append(col)
            elif c in {"OS_MONTHS", "OS_STATUS", "PFS_MONTHS", "PFS_STATUS", "PFI_MONTHS", "PFI_STATUS"}:
                drop_cols.append(col)
            elif c in {"DFS_MONTHS", "DFS_STATUS", "DSS_MONTHS", "DSS_STATUS", "RFS_MONTHS", "RFS_STATUS"}:
                drop_cols.append(col)
            elif c in {"DAYS_LAST_FOLLOWUP", "DAYS_TO_LAST_FOLLOWUP", "DAYS_TO_DEATH"}:
                drop_cols.append(col)
            elif any(token in c for token in ["SURVIVAL", "FOLLOWUP", "FOLLOW_UP", "DEATH", "VITAL_STATUS"]):
                drop_cols.append(col)
            elif any(token in c for token in ["NEW_TUMOR_EVENT", "RECURRENCE", "PROGRESSION", "NEOPLASM_CANCER_STATUS"]):
                drop_cols.append(col)
        return df.drop(columns=drop_cols, errors="ignore")

    def fit(self, clin_df: pd.DataFrame) -> "ClinicalTransformer":
        base = self._drop_survival_columns(clin_df).copy()

        num = base.select_dtypes(include=[np.number]).copy()
        self.numeric_cols = num.columns.tolist()

        cat_candidates = [c for c in base.columns if c not in self.numeric_cols]
        self.categorical_cols = []
        for col in cat_candidates:
            n_unique = base[col].nunique(dropna=True)
            if 1 < n_unique <= 20:
                self.categorical_cols.append(col)

        cat_frame = pd.DataFrame(index=base.index)
        if self.categorical_cols:
            cat_frame = pd.get_dummies(
                base[self.categorical_cols].astype(str).fillna("NA"),
                drop_first=False,
            )

        design = pd.concat([num, cat_frame], axis=1)
        design = design.loc[:, design.nunique(dropna=True) > 1]
        self.dummy_cols = design.columns.tolist()

        arr = design.to_numpy(dtype=float) if design.shape[1] else np.zeros((design.shape[0], 1), dtype=float)
        self.imputer.fit(arr)
        self.scaler.fit(self.imputer.transform(arr))
        return self

    def transform(self, clin_df: pd.DataFrame) -> np.ndarray:
        base = self._drop_survival_columns(clin_df).copy()
        num = base.reindex(columns=self.numeric_cols)

        cat_frame = pd.DataFrame(index=base.index)
        if self.categorical_cols:
            cat_frame = pd.get_dummies(
                base[self.categorical_cols].astype(str).fillna("NA"),
                drop_first=False,
            )

        design = pd.concat([num, cat_frame], axis=1)
        design = design.reindex(columns=self.dummy_cols, fill_value=0.0)

        arr = design.to_numpy(dtype=float) if design.shape[1] else np.zeros((design.shape[0], 1), dtype=float)
        arr = self.imputer.transform(arr)
        arr = self.scaler.transform(arr)
        return arr.astype(np.float32)


class OmicsTransformer:
    def __init__(self, top_var_features: int, pca_components: int, seed: int) -> None:
        self.top_var_features = top_var_features
        self.pca_components = pca_components
        self.seed = seed
        self.selected_cols: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None

    def fit(self, df: pd.DataFrame) -> "OmicsTransformer":
        var = df.var(axis=0, skipna=True).sort_values(ascending=False)
        n_keep = min(self.top_var_features, max(1, df.shape[1]))
        self.selected_cols = var.head(n_keep).index.tolist()

        arr = df[self.selected_cols].to_numpy(dtype=float)
        arr = self.imputer.fit_transform(arr)
        arr = self.scaler.fit_transform(arr)

        n_comp = min(self.pca_components, arr.shape[1], max(1, arr.shape[0] - 1))
        if n_comp >= 1:
            self.pca = PCA(n_components=n_comp, random_state=self.seed)
            self.pca.fit(arr)
        else:
            self.pca = None
        return self

    def transform_scaled(self, df: pd.DataFrame) -> np.ndarray:
        arr = df.reindex(columns=self.selected_cols).to_numpy(dtype=float)
        arr = self.imputer.transform(arr)
        arr = self.scaler.transform(arr)
        return arr.astype(np.float32)

    def transform_pca(self, df: pd.DataFrame) -> np.ndarray:
        arr = self.transform_scaled(df)
        if self.pca is None:
            return arr
        return self.pca.transform(arr).astype(np.float32)


class MutationTransformer:
    def __init__(self, top_genes: int) -> None:
        self.top_genes = top_genes
        self.selected_cols: List[str] = []

    def fit(self, mut_df: pd.DataFrame) -> "MutationTransformer":
        freq = mut_df.sum(axis=0).sort_values(ascending=False)
        n_keep = min(self.top_genes, max(1, mut_df.shape[1]))
        self.selected_cols = freq.head(n_keep).index.tolist()
        return self

    def transform(self, mut_df: pd.DataFrame) -> np.ndarray:
        out = mut_df.reindex(columns=self.selected_cols, fill_value=0.0)
        return out.to_numpy(dtype=np.float32)


@dataclass
class FoldFeatures:
    clinical: np.ndarray
    expr_scaled: np.ndarray
    expr_pca: np.ndarray
    meth_pca: np.ndarray
    mut_top: np.ndarray
    fused: np.ndarray


class FeatureAssembler:
    def __init__(
        self,
        expr_top_var: int,
        expr_pca_dim: int,
        meth_top_var: int,
        meth_pca_dim: int,
        mut_top_genes: int,
        seed: int,
    ) -> None:
        self.clinical_tf = ClinicalTransformer()
        self.expr_tf = OmicsTransformer(expr_top_var, expr_pca_dim, seed)
        self.meth_tf = OmicsTransformer(meth_top_var, meth_pca_dim, seed)
        self.mut_tf = MutationTransformer(mut_top_genes)

    def fit(
        self,
        expr_train: pd.DataFrame,
        meth_train: pd.DataFrame,
        mut_train: pd.DataFrame,
        clin_train: pd.DataFrame,
    ) -> "FeatureAssembler":
        self.clinical_tf.fit(clin_train)
        self.expr_tf.fit(expr_train)
        self.meth_tf.fit(meth_train)
        self.mut_tf.fit(mut_train)
        return self

    def transform(self, expr: pd.DataFrame, meth: pd.DataFrame, mut: pd.DataFrame, clin: pd.DataFrame) -> FoldFeatures:
        clinical = self.clinical_tf.transform(clin)
        expr_scaled = self.expr_tf.transform_scaled(expr)
        expr_pca = self.expr_tf.transform_pca(expr)
        meth_pca = self.meth_tf.transform_pca(meth)
        mut_top = self.mut_tf.transform(mut)
        fused = np.concatenate([clinical, expr_pca, meth_pca, mut_top], axis=1).astype(np.float32)

        return FoldFeatures(
            clinical=clinical,
            expr_scaled=expr_scaled,
            expr_pca=expr_pca,
            meth_pca=meth_pca,
            mut_top=mut_top,
            fused=fused,
        )

# -----------------------------
# Survival modeling helpers
# -----------------------------

def require_sksurv() -> Dict[str, Any]:
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
        from sksurv.util import Surv
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "scikit-survival is required. Install with: pip install scikit-survival"
        ) from exc

    return {
        "CoxPHSurvivalAnalysis": CoxPHSurvivalAnalysis,
        "CoxnetSurvivalAnalysis": CoxnetSurvivalAnalysis,
        "concordance_index_censored": concordance_index_censored,
        "cumulative_dynamic_auc": cumulative_dynamic_auc,
        "integrated_brier_score": integrated_brier_score,
        "Surv": Surv,
    }


def require_torch() -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required. Install with: pip install torch") from exc

    return {"torch": torch, "nn": nn, "optim": optim}


def make_surv_struct(event: np.ndarray, time: np.ndarray, sksurv_ref: Dict[str, Any]) -> np.ndarray:
    return sksurv_ref["Surv"].from_arrays(event=event.astype(bool), time=time.astype(float))


def cindex_score(event: np.ndarray, time: np.ndarray, risk: np.ndarray, sksurv_ref: Dict[str, Any]) -> float:
    cidx = sksurv_ref["concordance_index_censored"](event.astype(bool), time.astype(float), risk.astype(float))[0]
    return float(cidx)


def fit_breslow_baseline(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    exp_risk = np.exp(np.clip(risk, -30, 30))
    event_times = np.sort(np.unique(time[event == 1]))

    cum_haz_values = []
    cum = 0.0
    for t in event_times:
        d_t = float(np.sum((time == t) & (event == 1)))
        at_risk = exp_risk[time >= t]
        denom = float(np.sum(at_risk))
        if denom <= 0:
            continue
        cum += d_t / denom
        cum_haz_values.append(cum)

    if len(cum_haz_values) == 0:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
    return event_times[: len(cum_haz_values)].astype(float), np.asarray(cum_haz_values, dtype=float)


def survival_from_breslow(risk: np.ndarray, baseline_t: np.ndarray, baseline_h: np.ndarray, eval_t: np.ndarray) -> np.ndarray:
    risk = np.asarray(risk, dtype=float)
    eval_t = np.asarray(eval_t, dtype=float)

    if eval_t.size == 0:
        return np.zeros((risk.shape[0], 0), dtype=float)

    idx = np.searchsorted(baseline_t, eval_t, side="right") - 1
    idx = np.clip(idx, 0, len(baseline_h) - 1)
    h0 = baseline_h[idx]

    exp_risk = np.exp(np.clip(risk, -30, 30))[:, None]
    surv = np.exp(-h0[None, :] * exp_risk)
    return np.clip(surv, 1e-8, 1.0)


def evaluate_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    event_test: np.ndarray,
    time_test: np.ndarray,
    risk_test: np.ndarray,
    surv_test: np.ndarray,
    eval_times: np.ndarray,
    sksurv_ref: Dict[str, Any],
) -> Tuple[float, float, float, np.ndarray]:
    cidx = cindex_score(event_test, time_test, risk_test, sksurv_ref)

    if eval_times.size == 0 or surv_test.shape[1] == 0:
        return cidx, np.nan, np.nan, np.array([], dtype=float)

    ibs = np.nan
    mean_auc = np.nan
    auc_by_time = np.array([], dtype=float)

    try:
        ibs = float(sksurv_ref["integrated_brier_score"](y_train, y_test, surv_test, eval_times))
    except Exception:
        ibs = np.nan

    try:
        risk_time = 1.0 - surv_test
        auc_by_time, mean_auc = sksurv_ref["cumulative_dynamic_auc"](y_train, y_test, risk_time, eval_times)
        mean_auc = float(mean_auc)
        auc_by_time = np.asarray(auc_by_time, dtype=float)
    except Exception:
        mean_auc = np.nan
        auc_by_time = np.full(eval_times.shape[0], np.nan, dtype=float)

    return cidx, ibs, mean_auc, auc_by_time


def fit_coxph(
    X_train: np.ndarray,
    event_train: np.ndarray,
    time_train: np.ndarray,
    alpha: float,
    sksurv_ref: Dict[str, Any],
) -> Any:
    model = sksurv_ref["CoxPHSurvivalAnalysis"](alpha=alpha)
    y_train = make_surv_struct(event_train, time_train, sksurv_ref)
    model.fit(X_train, y_train)
    return model


def fit_coxnet(
    X_train: np.ndarray,
    event_train: np.ndarray,
    time_train: np.ndarray,
    l1_ratio: float,
    alpha_min_ratio: float,
    sksurv_ref: Dict[str, Any],
) -> Any:
    model = sksurv_ref["CoxnetSurvivalAnalysis"](
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        fit_baseline_model=True,
        n_alphas=80,
        max_iter=300000,
    )
    y_train = make_surv_struct(event_train, time_train, sksurv_ref)
    model.fit(X_train, y_train)
    return model


def best_alpha_for_validation(
    model: Any,
    X_val: np.ndarray,
    event_val: np.ndarray,
    time_val: np.ndarray,
    sksurv_ref: Dict[str, Any],
) -> Tuple[float, float]:
    best_alpha = float(model.alphas_[0])
    best_score = -np.inf

    for alpha in model.alphas_:
        try:
            risk = model.predict(X_val, alpha=float(alpha))
            score = cindex_score(event_val, time_val, risk, sksurv_ref)
        except Exception:
            continue

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_alpha = float(alpha)

    if not np.isfinite(best_score):
        best_score = np.nan
    return best_alpha, best_score


class DeepSurvNet:
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float, torch_ref: Dict[str, Any]) -> None:
        nn = torch_ref["nn"]

        layers: List[Any] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self.model = nn.Module()
        self.model.encoder = self.encoder
        self.model.head = self.head

        def _forward(x: Any) -> Any:
            z = self.model.encoder(x)
            return self.model.head(z).squeeze(-1)

        def _encode(x: Any) -> Any:
            return self.model.encoder(x)

        self.model.forward = _forward  # type: ignore[attr-defined]
        self.model.encode = _encode  # type: ignore[attr-defined]


def cox_loss_torch(risk: Any, durations: Any, events: Any, torch_ref: Dict[str, Any]) -> Any:
    torch = torch_ref["torch"]
    order = torch.argsort(durations, descending=True)
    risk_ord = risk[order]
    evt_ord = events[order]
    log_cumsum = torch.logcumsumexp(risk_ord, dim=0)
    return -(((risk_ord - log_cumsum) * evt_ord).sum() / (evt_ord.sum() + 1e-8))


def train_deepsurv(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_val: np.ndarray,
    time_val: np.ndarray,
    event_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    torch_ref: Dict[str, Any],
    sksurv_ref: Dict[str, Any],
) -> Tuple[Any, float]:
    torch = torch_ref["torch"]
    optim = torch_ref["optim"]

    torch.manual_seed(seed)
    model_obj = DeepSurvNet(
        in_dim=X_train.shape[1],
        hidden_dims=params["hidden_dims"],
        dropout=float(params["dropout"]),
        torch_ref=torch_ref,
    )
    model = model_obj.model

    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))
    epochs = int(params.get("epochs", 200))
    patience = int(params.get("patience", 25))

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ttr = torch.tensor(time_train, dtype=torch.float32)
    etr = torch.tensor(event_train, dtype=torch.float32)

    Xva = torch.tensor(X_val, dtype=torch.float32)

    best_state = None
    best_score = -np.inf
    wait = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        risk = model(Xtr)
        loss = cox_loss_torch(risk, ttr, etr, torch_ref)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            rv = model(Xva).cpu().numpy().astype(float)

        score = cindex_score(event_val, time_val, rv, sksurv_ref)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, float(best_score if np.isfinite(best_score) else np.nan)


class GatedMultiModal:
    def __init__(
        self,
        input_dims: Sequence[int],
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
        torch_ref: Dict[str, Any],
    ) -> None:
        nn = torch_ref["nn"]
        n_modal = len(input_dims)

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.ReLU(),
                )
                for d in input_dims
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(n_modal * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_modal),
        )
        self.out = nn.Linear(latent_dim, 1)

        self.model = nn.Module()
        self.model.encoders = self.encoders
        self.model.gate = self.gate
        self.model.out = self.out

        def _forward(mod_list: List[Any]) -> Tuple[Any, Any]:
            zs = [enc(x) for enc, x in zip(self.model.encoders, mod_list)]
            concat = torch_ref["torch"].cat(zs, dim=1)
            w = torch_ref["torch"].softmax(self.model.gate(concat), dim=1)
            fused = sum(z * w[:, i : i + 1] for i, z in enumerate(zs))
            risk = self.model.out(fused).squeeze(-1)
            return risk, fused

        self.model.forward = _forward  # type: ignore[attr-defined]


def train_gated_model(
    mods_train: List[np.ndarray],
    time_train: np.ndarray,
    event_train: np.ndarray,
    mods_val: List[np.ndarray],
    time_val: np.ndarray,
    event_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    torch_ref: Dict[str, Any],
    sksurv_ref: Dict[str, Any],
) -> Tuple[Any, float]:
    torch = torch_ref["torch"]
    optim = torch_ref["optim"]

    torch.manual_seed(seed)

    model_obj = GatedMultiModal(
        input_dims=[m.shape[1] for m in mods_train],
        latent_dim=int(params["latent_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        torch_ref=torch_ref,
    )
    model = model_obj.model

    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))
    epochs = int(params.get("epochs", 220))
    patience = int(params.get("patience", 30))

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = [torch.tensor(x, dtype=torch.float32) for x in mods_train]
    ttr = torch.tensor(time_train, dtype=torch.float32)
    etr = torch.tensor(event_train, dtype=torch.float32)
    Xva = [torch.tensor(x, dtype=torch.float32) for x in mods_val]

    best_state = None
    best_score = -np.inf
    wait = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        risk, _ = model(Xtr)
        loss = cox_loss_torch(risk, ttr, etr, torch_ref)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            rv, _ = model(Xva)
            rv = rv.cpu().numpy().astype(float)

        score = cindex_score(event_val, time_val, rv, sksurv_ref)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, float(best_score if np.isfinite(best_score) else np.nan)


def torch_predict_risk(model: Any, X: np.ndarray, torch_ref: Dict[str, Any]) -> np.ndarray:
    torch = torch_ref["torch"]
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        risk = model(xt).cpu().numpy().astype(float)
    return risk


def torch_predict_risk_and_embedding(model: Any, X: np.ndarray, torch_ref: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    torch = torch_ref["torch"]
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        z = model.encode(xt).cpu().numpy().astype(float)
        risk = model(xt).cpu().numpy().astype(float)
    return risk, z


def torch_gated_predict(model: Any, mods: List[np.ndarray], torch_ref: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    torch = torch_ref["torch"]
    model.eval()
    with torch.no_grad():
        mt = [torch.tensor(x, dtype=torch.float32) for x in mods]
        risk, fused = model(mt)
        return risk.cpu().numpy().astype(float), fused.cpu().numpy().astype(float)

# -----------------------------
# Inner-loop tuning
# -----------------------------

def tune_clinical_cox(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    inner_cv: StratifiedKFold,
    sksurv_ref: Dict[str, Any],
) -> Dict[str, Any]:
    grid = [{"alpha": 0.0}, {"alpha": 0.01}, {"alpha": 0.05}]
    best = {"params": grid[0], "score": -np.inf}

    for params in grid:
        scores = []
        for tr, va in inner_cv.split(X, event):
            try:
                model = fit_coxph(X[tr], event[tr], time[tr], params["alpha"], sksurv_ref)
                risk = model.predict(X[va]).astype(float)
                scores.append(cindex_score(event[va], time[va], risk, sksurv_ref))
            except Exception:
                continue

        if scores:
            s = float(np.nanmean(scores))
            if np.isfinite(s) and s > best["score"]:
                best = {"params": params, "score": s}

    return best


def tune_coxnet(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    inner_cv: StratifiedKFold,
    sksurv_ref: Dict[str, Any],
) -> Dict[str, Any]:
    grid = [
        {"l1_ratio": 0.1, "alpha_min_ratio": 0.05},
        {"l1_ratio": 0.5, "alpha_min_ratio": 0.05},
        {"l1_ratio": 0.9, "alpha_min_ratio": 0.05},
    ]

    best = {"params": grid[0], "score": -np.inf, "alpha": None}

    for params in grid:
        fold_scores = []
        fold_alphas = []
        for tr, va in inner_cv.split(X, event):
            try:
                model = fit_coxnet(
                    X[tr],
                    event[tr],
                    time[tr],
                    params["l1_ratio"],
                    params["alpha_min_ratio"],
                    sksurv_ref,
                )
                alpha, score = best_alpha_for_validation(model, X[va], event[va], time[va], sksurv_ref)
                fold_scores.append(score)
                fold_alphas.append(alpha)
            except Exception:
                continue

        if fold_scores:
            s = float(np.nanmean(fold_scores))
            alpha_med = float(np.nanmedian(fold_alphas)) if fold_alphas else None
            if np.isfinite(s) and s > best["score"]:
                best = {"params": params, "score": s, "alpha": alpha_med}

    return best


def tune_deepsurv(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    inner_cv: StratifiedKFold,
    seed: int,
    torch_ref: Dict[str, Any],
    sksurv_ref: Dict[str, Any],
    fast: bool,
) -> Dict[str, Any]:
    epochs = 120 if fast else 220
    grid = [
        {"hidden_dims": [64, 32], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "epochs": epochs, "patience": 20},
        {"hidden_dims": [128, 64], "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4, "epochs": epochs, "patience": 20},
    ]

    best = {"params": grid[0], "score": -np.inf}
    for params in grid:
        scores = []
        for i, (tr, va) in enumerate(inner_cv.split(X, event), start=1):
            try:
                model, _ = train_deepsurv(
                    X[tr],
                    time[tr],
                    event[tr],
                    X[va],
                    time[va],
                    event[va],
                    params,
                    seed + i,
                    torch_ref,
                    sksurv_ref,
                )
                risk = torch_predict_risk(model, X[va], torch_ref)
                score = cindex_score(event[va], time[va], risk, sksurv_ref)
                scores.append(score)
            except Exception:
                continue

        if scores:
            s = float(np.nanmean(scores))
            if np.isfinite(s) and s > best["score"]:
                best = {"params": params, "score": s}

    return best


def tune_gated(
    feats: FoldFeatures,
    time: np.ndarray,
    event: np.ndarray,
    inner_cv: StratifiedKFold,
    seed: int,
    torch_ref: Dict[str, Any],
    sksurv_ref: Dict[str, Any],
    fast: bool,
) -> Dict[str, Any]:
    epochs = 140 if fast else 260
    grid = [
        {"latent_dim": 24, "hidden_dim": 64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "epochs": epochs, "patience": 25},
        {"latent_dim": 32, "hidden_dim": 96, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4, "epochs": epochs, "patience": 25},
    ]

    mods_full = [feats.clinical, feats.expr_pca, feats.meth_pca, feats.mut_top]
    best = {"params": grid[0], "score": -np.inf}

    for params in grid:
        scores = []
        for i, (tr, va) in enumerate(inner_cv.split(feats.fused, event), start=1):
            try:
                mods_tr = [m[tr] for m in mods_full]
                mods_va = [m[va] for m in mods_full]
                model, _ = train_gated_model(
                    mods_tr,
                    time[tr],
                    event[tr],
                    mods_va,
                    time[va],
                    event[va],
                    params,
                    seed + i,
                    torch_ref,
                    sksurv_ref,
                )
                risk, _ = torch_gated_predict(model, mods_va, torch_ref)
                score = cindex_score(event[va], time[va], risk, sksurv_ref)
                scores.append(score)
            except Exception:
                continue

        if scores:
            s = float(np.nanmean(scores))
            if np.isfinite(s) and s > best["score"]:
                best = {"params": params, "score": s}

    return best

# -----------------------------
# Outer CV benchmark
# -----------------------------

def stratified_holdout(event: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    event = np.asarray(event, dtype=int)
    idx = np.arange(event.shape[0])

    val_idx = []
    train_idx = []
    for cls in [0, 1]:
        cls_idx = idx[event == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction))) if len(cls_idx) > 1 else 0
        if n_val >= len(cls_idx):
            n_val = max(1, len(cls_idx) - 1)
        if n_val > 0:
            val_idx.extend(cls_idx[:n_val].tolist())
            train_idx.extend(cls_idx[n_val:].tolist())
        else:
            train_idx.extend(cls_idx.tolist())

    train_idx = np.asarray(sorted(train_idx), dtype=int)
    val_idx = np.asarray(sorted(val_idx), dtype=int)

    if len(train_idx) == 0 or len(val_idx) == 0:
        split = max(1, int(0.8 * len(idx)))
        train_idx = idx[:split]
        val_idx = idx[split:]

    return train_idx, val_idx


def nearest_alpha(model: Any, target_alpha: Optional[float]) -> float:
    alphas = np.asarray(model.alphas_, dtype=float)
    if alphas.size == 0:
        raise ValueError("Coxnet returned empty alpha path.")
    if target_alpha is None or not np.isfinite(target_alpha):
        return float(alphas[len(alphas) // 2])
    i = int(np.argmin(np.abs(alphas - float(target_alpha))))
    return float(alphas[i])


def sksurv_survival_matrix(model: Any, X: np.ndarray, eval_times: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    if eval_times.size == 0:
        return np.zeros((X.shape[0], 0), dtype=float)

    if alpha is None:
        surv_funcs = model.predict_survival_function(X)
    else:
        surv_funcs = model.predict_survival_function(X, alpha=float(alpha))

    mat = np.row_stack([fn(eval_times) for fn in surv_funcs]).astype(float)
    return np.clip(mat, 1e-8, 1.0)


def run_nested_cv(
    bundle: DataBundle,
    args: argparse.Namespace,
    logger: logging.Logger,
    sksurv_ref: Dict[str, Any],
    torch_ref: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: List[Dict[str, Any]] = []
    auc_rows: List[Dict[str, Any]] = []
    param_rows: List[Dict[str, Any]] = []
    requested_eval_times = parse_eval_horizons_months(args.auc_horizons_months)

    outer_cv = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(np.arange(len(bundle.patient_ids)), bundle.event), start=1):
        logger.info("Outer fold %d/%d", fold_id, args.outer_folds)

        train_ids = np.asarray(bundle.patient_ids)[train_idx]
        test_ids = np.asarray(bundle.patient_ids)[test_idx]

        expr_train = bundle.expr.loc[train_ids]
        expr_test = bundle.expr.loc[test_ids]
        meth_train = bundle.meth.loc[train_ids]
        meth_test = bundle.meth.loc[test_ids]
        mut_train = bundle.mut.loc[train_ids]
        mut_test = bundle.mut.loc[test_ids]
        clin_train = bundle.clin.loc[train_ids]
        clin_test = bundle.clin.loc[test_ids]

        time_train = bundle.time[train_idx]
        event_train = bundle.event[train_idx]
        time_test = bundle.time[test_idx]
        event_test = bundle.event[test_idx]

        assembler = FeatureAssembler(
            expr_top_var=args.expr_top_var,
            expr_pca_dim=args.expr_pca_dim,
            meth_top_var=args.meth_top_var,
            meth_pca_dim=args.meth_pca_dim,
            mut_top_genes=args.mut_top_genes,
            seed=args.seed + fold_id,
        )
        assembler.fit(expr_train, meth_train, mut_train, clin_train)

        f_train = assembler.transform(expr_train, meth_train, mut_train, clin_train)
        f_test = assembler.transform(expr_test, meth_test, mut_test, clin_test)

        y_train = make_surv_struct(event_train, time_train, sksurv_ref)
        y_test = make_surv_struct(event_test, time_test, sksurv_ref)
        eval_times = choose_eval_times(
            time_train,
            event_train,
            n_times=args.auc_time_points,
            requested_times=requested_eval_times,
        )

        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + 1000 + fold_id,
        )

        best_clin = tune_clinical_cox(f_train.clinical, time_train, event_train, inner_cv, sksurv_ref)
        best_expr_cox = tune_coxnet(f_train.expr_scaled, time_train, event_train, inner_cv, sksurv_ref)
        best_expr_ds = tune_deepsurv(
            f_train.expr_pca,
            time_train,
            event_train,
            inner_cv,
            args.seed + 2000 + fold_id,
            torch_ref,
            sksurv_ref,
            args.fast_mode,
        )
        best_lf_cox = tune_coxnet(f_train.fused, time_train, event_train, inner_cv, sksurv_ref)
        best_lf_ds = tune_deepsurv(
            f_train.fused,
            time_train,
            event_train,
            inner_cv,
            args.seed + 3000 + fold_id,
            torch_ref,
            sksurv_ref,
            args.fast_mode,
        )
        best_gated = tune_gated(
            f_train,
            time_train,
            event_train,
            inner_cv,
            args.seed + 4000 + fold_id,
            torch_ref,
            sksurv_ref,
            args.fast_mode,
        )

        tuned = {
            "Clinical-Cox": best_clin,
            "Expr-Penalized-Cox": best_expr_cox,
            "Expr-DeepSurv": best_expr_ds,
            "LateFusion-Cox": best_lf_cox,
            "LateFusion-DeepSurv": best_lf_ds,
            "Gated-MultiModal": best_gated,
        }

        for model_name, item in tuned.items():
            param_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "inner_cindex": item.get("score", np.nan),
                    "params_json": json.dumps(item.get("params", {}), sort_keys=True),
                    "selected_alpha": item.get("alpha", np.nan),
                }
            )

        # 1) Clinical Cox
        try:
            clin_alpha = float(best_clin["params"].get("alpha", 0.0))
            model = fit_coxph(f_train.clinical, event_train, time_train, clin_alpha, sksurv_ref)
            risk_test = model.predict(f_test.clinical).astype(float)
            surv_test = sksurv_survival_matrix(model, f_test.clinical, eval_times)
            cidx, ibs, mauc, auc_t = evaluate_metrics(
                y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
            )
            results.append({"fold": fold_id, "model": "Clinical-Cox", "cindex": cidx, "ibs": ibs, "mean_auc": mauc})
            for t, a in zip(eval_times, auc_t):
                auc_rows.append({"fold": fold_id, "model": "Clinical-Cox", "time": t, "auc": a})
        except Exception as exc:
            logger.exception("Clinical-Cox failed on fold %d: %s", fold_id, exc)

        # 2) Expression penalized Cox
        try:
            model = fit_coxnet(
                f_train.expr_scaled,
                event_train,
                time_train,
                best_expr_cox["params"]["l1_ratio"],
                best_expr_cox["params"]["alpha_min_ratio"],
                sksurv_ref,
            )
            alpha = nearest_alpha(model, best_expr_cox.get("alpha", None))
            risk_test = model.predict(f_test.expr_scaled, alpha=alpha).astype(float)
            surv_test = sksurv_survival_matrix(model, f_test.expr_scaled, eval_times, alpha=alpha)
            cidx, ibs, mauc, auc_t = evaluate_metrics(
                y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
            )
            results.append({"fold": fold_id, "model": "Expr-Penalized-Cox", "cindex": cidx, "ibs": ibs, "mean_auc": mauc})
            for t, a in zip(eval_times, auc_t):
                auc_rows.append({"fold": fold_id, "model": "Expr-Penalized-Cox", "time": t, "auc": a})
        except Exception as exc:
            logger.exception("Expr-Penalized-Cox failed on fold %d: %s", fold_id, exc)

        # 3) Expression DeepSurv
        try:
            tr_sub, va_sub = stratified_holdout(event_train, val_fraction=0.2, seed=args.seed + 5000 + fold_id)
            model, _ = train_deepsurv(
                f_train.expr_pca[tr_sub],
                time_train[tr_sub],
                event_train[tr_sub],
                f_train.expr_pca[va_sub],
                time_train[va_sub],
                event_train[va_sub],
                best_expr_ds["params"],
                args.seed + 5100 + fold_id,
                torch_ref,
                sksurv_ref,
            )
            risk_train = torch_predict_risk(model, f_train.expr_pca, torch_ref)
            risk_test = torch_predict_risk(model, f_test.expr_pca, torch_ref)
            bt, bh = fit_breslow_baseline(time_train, event_train, risk_train)
            surv_test = survival_from_breslow(risk_test, bt, bh, eval_times)
            cidx, ibs, mauc, auc_t = evaluate_metrics(
                y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
            )
            results.append({"fold": fold_id, "model": "Expr-DeepSurv", "cindex": cidx, "ibs": ibs, "mean_auc": mauc})
            for t, a in zip(eval_times, auc_t):
                auc_rows.append({"fold": fold_id, "model": "Expr-DeepSurv", "time": t, "auc": a})
        except Exception as exc:
            logger.exception("Expr-DeepSurv failed on fold %d: %s", fold_id, exc)

        # 4) Late-fusion Cox
        try:
            model = fit_coxnet(
                f_train.fused,
                event_train,
                time_train,
                best_lf_cox["params"]["l1_ratio"],
                best_lf_cox["params"]["alpha_min_ratio"],
                sksurv_ref,
            )
            alpha = nearest_alpha(model, best_lf_cox.get("alpha", None))
            risk_test = model.predict(f_test.fused, alpha=alpha).astype(float)
            surv_test = sksurv_survival_matrix(model, f_test.fused, eval_times, alpha=alpha)
            cidx, ibs, mauc, auc_t = evaluate_metrics(
                y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
            )
            results.append({"fold": fold_id, "model": "LateFusion-Cox", "cindex": cidx, "ibs": ibs, "mean_auc": mauc})
            for t, a in zip(eval_times, auc_t):
                auc_rows.append({"fold": fold_id, "model": "LateFusion-Cox", "time": t, "auc": a})
        except Exception as exc:
            logger.exception("LateFusion-Cox failed on fold %d: %s", fold_id, exc)

        # 5) Late-fusion DeepSurv
        try:
            tr_sub, va_sub = stratified_holdout(event_train, val_fraction=0.2, seed=args.seed + 5200 + fold_id)
            model, _ = train_deepsurv(
                f_train.fused[tr_sub],
                time_train[tr_sub],
                event_train[tr_sub],
                f_train.fused[va_sub],
                time_train[va_sub],
                event_train[va_sub],
                best_lf_ds["params"],
                args.seed + 5300 + fold_id,
                torch_ref,
                sksurv_ref,
            )
            risk_train = torch_predict_risk(model, f_train.fused, torch_ref)
            risk_test, _ = torch_predict_risk_and_embedding(model, f_test.fused, torch_ref)
            bt, bh = fit_breslow_baseline(time_train, event_train, risk_train)
            surv_test = survival_from_breslow(risk_test, bt, bh, eval_times)
            cidx, ibs, mauc, auc_t = evaluate_metrics(
                y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
            )
            results.append({"fold": fold_id, "model": "LateFusion-DeepSurv", "cindex": cidx, "ibs": ibs, "mean_auc": mauc})
            for t, a in zip(eval_times, auc_t):
                auc_rows.append({"fold": fold_id, "model": "LateFusion-DeepSurv", "time": t, "auc": a})
        except Exception as exc:
            logger.exception("LateFusion-DeepSurv failed on fold %d: %s", fold_id, exc)

        # 6) Gated multimodal
        try:
            tr_sub, va_sub = stratified_holdout(event_train, val_fraction=0.2, seed=args.seed + 5400 + fold_id)
            mods_train = [f_train.clinical, f_train.expr_pca, f_train.meth_pca, f_train.mut_top]
            mods_test = [f_test.clinical, f_test.expr_pca, f_test.meth_pca, f_test.mut_top]

            model, _ = train_gated_model(
                [m[tr_sub] for m in mods_train],
                time_train[tr_sub],
                event_train[tr_sub],
                [m[va_sub] for m in mods_train],
                time_train[va_sub],
                event_train[va_sub],
                best_gated["params"],
                args.seed + 5500 + fold_id,
                torch_ref,
                sksurv_ref,
            )

            risk_train, _ = torch_gated_predict(model, mods_train, torch_ref)
            risk_test, _ = torch_gated_predict(model, mods_test, torch_ref)
            bt, bh = fit_breslow_baseline(time_train, event_train, risk_train)
            surv_test = survival_from_breslow(risk_test, bt, bh, eval_times)
            cidx, ibs, mauc, auc_t = evaluate_metrics(
                y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
            )
            results.append({"fold": fold_id, "model": "Gated-MultiModal", "cindex": cidx, "ibs": ibs, "mean_auc": mauc})
            for t, a in zip(eval_times, auc_t):
                auc_rows.append({"fold": fold_id, "model": "Gated-MultiModal", "time": t, "auc": a})
        except Exception as exc:
            logger.exception("Gated-MultiModal failed on fold %d: %s", fold_id, exc)

    fold_df = pd.DataFrame(results)
    auc_df = pd.DataFrame(auc_rows)
    params_df = pd.DataFrame(param_rows)
    return fold_df, auc_df, params_df

# -----------------------------
# Summaries, plots, freezing
# -----------------------------

def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    summary = (
        fold_df.groupby("model", as_index=False)
        .agg(
            cindex_mean=("cindex", "mean"),
            cindex_std=("cindex", "std"),
            ibs_mean=("ibs", "mean"),
            ibs_std=("ibs", "std"),
            mean_auc_mean=("mean_auc", "mean"),
            mean_auc_std=("mean_auc", "std"),
            n_folds=("fold", "count"),
        )
        .sort_values("cindex_mean", ascending=False)
        .reset_index(drop=True)
    )

    for col in ["cindex_mean", "mean_auc_mean", "ibs_mean"]:
        if col not in summary:
            summary[col] = np.nan

    # Composite ranking: higher c-index and AUC, lower IBS.
    c_z = safe_standardize_risk(summary["cindex_mean"])
    a_z = safe_standardize_risk(summary["mean_auc_mean"])
    i_z = safe_standardize_risk(summary["ibs_mean"])
    summary["composite_score"] = c_z + a_z - i_z
    summary = summary.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return summary


def select_best_models(summary_df: pd.DataFrame) -> Tuple[str, str]:
    if summary_df.empty:
        raise ValueError("Cannot select best models from empty summary table.")

    integrated_candidates = ["LateFusion-Cox", "LateFusion-DeepSurv", "Gated-MultiModal"]
    simple_candidates = ["Expr-Penalized-Cox", "Clinical-Cox", "Expr-DeepSurv"]

    integ = summary_df[summary_df["model"].isin(integrated_candidates)].copy()
    simple = summary_df[summary_df["model"].isin(simple_candidates)].copy()

    if integ.empty:
        best_main = summary_df.iloc[0]["model"]
    else:
        best_main = integ.sort_values("composite_score", ascending=False).iloc[0]["model"]

    if simple.empty:
        best_simple = summary_df.iloc[min(1, len(summary_df) - 1)]["model"]
    else:
        best_simple = simple.sort_values("composite_score", ascending=False).iloc[0]["model"]

    return str(best_main), str(best_simple)


def consensus_params(params_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    sub = params_df[params_df["model"] == model_name].copy()
    if sub.empty:
        return {}
    mode_row = sub["params_json"].mode()
    if mode_row.empty:
        return {}
    return json.loads(mode_row.iloc[0])


def save_plots(fold_df: pd.DataFrame, auc_df: pd.DataFrame, out_plot_dir: Path) -> None:
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if not fold_df.empty:
        for metric in ["cindex", "ibs", "mean_auc"]:
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=fold_df, x="model", y=metric)
            ax.tick_params(axis="x", rotation=35)
            plt.tight_layout()
            plt.savefig(out_plot_dir / f"nestedcv_{metric}_boxplot.png", dpi=300)
            plt.close()

    if not auc_df.empty:
        auc_mean = auc_df.groupby(["model", "time"], as_index=False).agg(auc_mean=("auc", "mean"))
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=auc_mean, x="time", y="auc_mean", hue="model", marker="o")
        plt.xlabel("Time")
        plt.ylabel("Dynamic AUC")
        plt.tight_layout()
        plt.savefig(out_plot_dir / "time_dependent_auc_curves.png", dpi=300)
        plt.close()


def freeze_models_and_export_representation(
    bundle: DataBundle,
    params_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
    sksurv_ref: Dict[str, Any],
    torch_ref: Dict[str, Any],
    out_art_dir: Path,
) -> None:
    out_art_dir.mkdir(parents=True, exist_ok=True)

    best_main, best_simple = select_best_models(summary_df)
    logger.info("Best main model: %s | Best simple model: %s", best_main, best_simple)

    assembler = FeatureAssembler(
        expr_top_var=args.expr_top_var,
        expr_pca_dim=args.expr_pca_dim,
        meth_top_var=args.meth_top_var,
        meth_pca_dim=args.meth_pca_dim,
        mut_top_genes=args.mut_top_genes,
        seed=args.seed + 9999,
    )
    assembler.fit(bundle.expr, bundle.meth, bundle.mut, bundle.clin)
    feats = assembler.transform(bundle.expr, bundle.meth, bundle.mut, bundle.clin)

    risks: Dict[str, np.ndarray] = {}
    main_rep = feats.fused.copy()

    def _fit_and_predict(model_name: str) -> None:
        nonlocal main_rep
        params = consensus_params(params_df, model_name)

        if model_name == "Clinical-Cox":
            alpha = float(params.get("alpha", 0.01))
            model = fit_coxph(feats.clinical, bundle.event, bundle.time, alpha, sksurv_ref)
            risk = model.predict(feats.clinical).astype(float)
            risks[model_name] = risk
            dump_artifact({"model": model, "assembler": assembler, "feature_block": "clinical"}, out_art_dir / "clinical_cox.joblib")

        elif model_name == "Expr-Penalized-Cox":
            model = fit_coxnet(
                feats.expr_scaled,
                bundle.event,
                bundle.time,
                float(params.get("l1_ratio", 0.5)),
                float(params.get("alpha_min_ratio", 0.05)),
                sksurv_ref,
            )
            alpha = nearest_alpha(model, None)
            risk = model.predict(feats.expr_scaled, alpha=alpha).astype(float)
            risks[model_name] = risk
            dump_artifact(
                {"model": model, "alpha": alpha, "assembler": assembler, "feature_block": "expr_scaled"},
                out_art_dir / "expr_penalized_cox.joblib",
            )

        elif model_name == "Expr-DeepSurv":
            tr_sub, va_sub = stratified_holdout(bundle.event, 0.2, args.seed + 30001)
            ds_params = params if params else {
                "hidden_dims": [64, 32],
                "dropout": 0.2,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 220,
                "patience": 25,
            }
            model, _ = train_deepsurv(
                feats.expr_pca[tr_sub],
                bundle.time[tr_sub],
                bundle.event[tr_sub],
                feats.expr_pca[va_sub],
                bundle.time[va_sub],
                bundle.event[va_sub],
                ds_params,
                args.seed + 30011,
                torch_ref,
                sksurv_ref,
            )
            risk, emb = torch_predict_risk_and_embedding(model, feats.expr_pca, torch_ref)
            risks[model_name] = risk
            payload = {
                "state_dict": model.state_dict(),
                "params": ds_params,
                "assembler": assembler,
                "feature_block": "expr_pca",
            }
            torch_ref["torch"].save(payload, out_art_dir / "expr_deepsurv.pt")
            main_rep = emb

        elif model_name == "LateFusion-Cox":
            model = fit_coxnet(
                feats.fused,
                bundle.event,
                bundle.time,
                float(params.get("l1_ratio", 0.5)),
                float(params.get("alpha_min_ratio", 0.05)),
                sksurv_ref,
            )
            alpha = nearest_alpha(model, None)
            risk = model.predict(feats.fused, alpha=alpha).astype(float)
            risks[model_name] = risk
            dump_artifact(
                {"model": model, "alpha": alpha, "assembler": assembler, "feature_block": "fused"},
                out_art_dir / "latefusion_cox.joblib",
            )
            main_rep = feats.fused

        elif model_name == "LateFusion-DeepSurv":
            tr_sub, va_sub = stratified_holdout(bundle.event, 0.2, args.seed + 31001)
            ds_params = params if params else {
                "hidden_dims": [128, 64],
                "dropout": 0.3,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "epochs": 240,
                "patience": 25,
            }
            model, _ = train_deepsurv(
                feats.fused[tr_sub],
                bundle.time[tr_sub],
                bundle.event[tr_sub],
                feats.fused[va_sub],
                bundle.time[va_sub],
                bundle.event[va_sub],
                ds_params,
                args.seed + 31011,
                torch_ref,
                sksurv_ref,
            )
            risk, emb = torch_predict_risk_and_embedding(model, feats.fused, torch_ref)
            risks[model_name] = risk
            torch_ref["torch"].save(
                {"state_dict": model.state_dict(), "params": ds_params, "assembler": assembler, "feature_block": "fused"},
                out_art_dir / "latefusion_deepsurv.pt",
            )
            main_rep = emb

        elif model_name == "Gated-MultiModal":
            tr_sub, va_sub = stratified_holdout(bundle.event, 0.2, args.seed + 32001)
            gm_params = params if params else {
                "latent_dim": 32,
                "hidden_dim": 96,
                "dropout": 0.3,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "epochs": 260,
                "patience": 30,
            }
            mods = [feats.clinical, feats.expr_pca, feats.meth_pca, feats.mut_top]
            model, _ = train_gated_model(
                [m[tr_sub] for m in mods],
                bundle.time[tr_sub],
                bundle.event[tr_sub],
                [m[va_sub] for m in mods],
                bundle.time[va_sub],
                bundle.event[va_sub],
                gm_params,
                args.seed + 32011,
                torch_ref,
                sksurv_ref,
            )
            risk, emb = torch_gated_predict(model, mods, torch_ref)
            risks[model_name] = risk
            torch_ref["torch"].save(
                {"state_dict": model.state_dict(), "params": gm_params, "assembler": assembler},
                out_art_dir / "gated_multimodal.pt",
            )
            main_rep = emb

        else:
            raise ValueError(f"Unknown model for freezing: {model_name}")

    _fit_and_predict(best_main)
    if best_simple != best_main:
        _fit_and_predict(best_simple)

    risk_df = pd.DataFrame(index=bundle.patient_ids)
    for name, arr in risks.items():
        risk_df[name] = arr
    risk_df.index.name = "PATIENT_ID"
    risk_df.to_csv(out_art_dir / "full_cohort_risk_scores.csv")

    rep_df = pd.DataFrame(main_rep, index=bundle.patient_ids)
    rep_df.index.name = "PATIENT_ID"
    rep_df.columns = [f"latent_{i+1}" for i in range(rep_df.shape[1])]
    rep_df.to_parquet(out_art_dir / "best_integrated_representation.parquet")
    rep_df.to_csv(out_art_dir / "best_integrated_representation.csv")

    meta_main = {
        "best_main_model": best_main,
        "best_simple_model": best_simple,
        "n_patients": len(bundle.patient_ids),
        "endpoint": args.endpoint,
    }
    with (out_art_dir / "model_selection.json").open("w", encoding="utf-8") as f:
        json.dump(meta_main, f, indent=2)


def parse_seed_list(seed_text: str) -> List[int]:
    vals = [s.strip() for s in str(seed_text).split(",") if s.strip()]
    out = [int(v) for v in vals]
    if not out:
        raise ValueError("No valid seeds were provided.")
    return out


def run_ablation(
    bundle: DataBundle,
    args: argparse.Namespace,
    seeds: Sequence[int],
    logger: logging.Logger,
    sksurv_ref: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    requested_eval_times = parse_eval_horizons_months(args.auc_horizons_months)
    for seed in seeds:
        outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=seed + 777)
        for fold_id, (tr, te) in enumerate(outer.split(np.arange(len(bundle.patient_ids)), bundle.event), start=1):
            train_ids = np.asarray(bundle.patient_ids)[tr]
            test_ids = np.asarray(bundle.patient_ids)[te]

            assembler = FeatureAssembler(
                expr_top_var=args.expr_top_var,
                expr_pca_dim=args.expr_pca_dim,
                meth_top_var=args.meth_top_var,
                meth_pca_dim=args.meth_pca_dim,
                mut_top_genes=args.mut_top_genes,
                seed=seed + fold_id,
            )
            assembler.fit(bundle.expr.loc[train_ids], bundle.meth.loc[train_ids], bundle.mut.loc[train_ids], bundle.clin.loc[train_ids])
            f_train = assembler.transform(bundle.expr.loc[train_ids], bundle.meth.loc[train_ids], bundle.mut.loc[train_ids], bundle.clin.loc[train_ids])
            f_test = assembler.transform(bundle.expr.loc[test_ids], bundle.meth.loc[test_ids], bundle.mut.loc[test_ids], bundle.clin.loc[test_ids])

            combos = {
                "clinical_only": [f_train.clinical, f_test.clinical],
                "expression_only": [f_train.expr_pca, f_test.expr_pca],
                "methylation_only": [f_train.meth_pca, f_test.meth_pca],
                "mutation_only": [f_train.mut_top, f_test.mut_top],
                "expression_methylation": [np.concatenate([f_train.expr_pca, f_train.meth_pca], axis=1), np.concatenate([f_test.expr_pca, f_test.meth_pca], axis=1)],
                "expression_mutation": [np.concatenate([f_train.expr_pca, f_train.mut_top], axis=1), np.concatenate([f_test.expr_pca, f_test.mut_top], axis=1)],
                "methylation_mutation": [np.concatenate([f_train.meth_pca, f_train.mut_top], axis=1), np.concatenate([f_test.meth_pca, f_test.mut_top], axis=1)],
                "all_modalities": [f_train.fused, f_test.fused],
            }

            event_train = bundle.event[tr]
            time_train = bundle.time[tr]
            event_test = bundle.event[te]
            time_test = bundle.time[te]
            y_train = make_surv_struct(event_train, time_train, sksurv_ref)
            y_test = make_surv_struct(event_test, time_test, sksurv_ref)
            eval_times = choose_eval_times(time_train, event_train, n_times=15, requested_times=requested_eval_times)

            for combo_name, (xtr, xte) in combos.items():
                try:
                    model = fit_coxnet(xtr, event_train, time_train, 0.5, 0.05, sksurv_ref)
                    tr_sub, va_sub = stratified_holdout(event_train, 0.2, seed + fold_id)
                    alpha, _ = best_alpha_for_validation(model, xtr[va_sub], event_train[va_sub], time_train[va_sub], sksurv_ref)
                    alpha = nearest_alpha(model, alpha)
                    risk_test = model.predict(xte, alpha=alpha).astype(float)
                    surv_test = sksurv_survival_matrix(model, xte, eval_times, alpha=alpha)
                    cidx, ibs, auc_mean, _ = evaluate_metrics(
                        y_train, y_test, event_test, time_test, risk_test, surv_test, eval_times, sksurv_ref
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "fold": fold_id,
                            "modality_combo": combo_name,
                            "cindex": cidx,
                            "ibs": ibs,
                            "mean_auc": auc_mean,
                        }
                    )
                except Exception as exc:
                    logger.warning("Ablation failure | seed=%s fold=%s combo=%s: %s", seed, fold_id, combo_name, exc)
    return pd.DataFrame(rows)


def calibration_summary(bundle: DataBundle, risk_df: pd.DataFrame, horizons: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    event = bundle.event.astype(int)
    time = bundle.time.astype(float)
    for model in risk_df.columns:
        z = safe_standardize_risk(risk_df[model])
        prob = 1.0 / (1.0 + np.exp(-z))
        for h in horizons:
            y = ((event == 1) & (time <= h)).astype(int)
            tmp = pd.DataFrame({"prob": prob.values, "y": y})
            tmp["bin"] = pd.qcut(tmp["prob"], q=min(5, tmp.shape[0]), labels=False, duplicates="drop")
            for b, g in tmp.groupby("bin"):
                rows.append(
                    {
                        "model": model,
                        "horizon_month": float(h),
                        "bin": int(b),
                        "n": int(g.shape[0]),
                        "mean_pred": float(g["prob"].mean()),
                        "observed_rate": float(g["y"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def dca_summary(bundle: DataBundle, risk_df: pd.DataFrame, horizons: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    event = bundle.event.astype(int)
    time = bundle.time.astype(float)
    thresholds = np.arange(0.05, 0.81, 0.05)
    n = len(event)
    for model in risk_df.columns:
        z = safe_standardize_risk(risk_df[model])
        prob = (1.0 / (1.0 + np.exp(-z))).to_numpy(dtype=float)
        for h in horizons:
            y = ((event == 1) & (time <= h)).astype(int)
            for pt in thresholds:
                pred_pos = prob >= pt
                tp = float(np.sum((pred_pos == 1) & (y == 1)))
                fp = float(np.sum((pred_pos == 1) & (y == 0)))
                nb = tp / n - fp / n * (pt / (1.0 - pt))
                rows.append(
                    {
                        "model": model,
                        "horizon_month": float(h),
                        "threshold": float(pt),
                        "net_benefit": float(nb),
                    }
                )
    return pd.DataFrame(rows)


def locked_cutoff_table(risk_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in risk_df.columns:
        s = pd.to_numeric(risk_df[c], errors="coerce").dropna()
        out[c] = {
            "median": float(s.median()),
            "q33": float(s.quantile(0.33)),
            "q67": float(s.quantile(0.67)),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
        }
    return out


def multivariable_cox_table(bundle: DataBundle, risk_df: pd.DataFrame, best_model: str) -> pd.DataFrame:
    try:
        from lifelines import CoxPHFitter
    except Exception:
        return pd.DataFrame()

    stage_map = {
        "STAGE IA": 1,
        "STAGE I": 1,
        "STAGE IB": 2,
        "STAGE IIA": 3,
        "STAGE IIB": 4,
        "STAGE III": 5,
        "STAGE IV": 6,
    }
    grade_map = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}

    risk_col = best_model if best_model in risk_df.columns else risk_df.columns[0]
    df = pd.DataFrame(index=bundle.patient_ids)
    df["T"] = bundle.time
    df["E"] = bundle.event
    df["risk_score"] = pd.to_numeric(risk_df[risk_col], errors="coerce")
    df["age"] = pd.to_numeric(bundle.clin.get("AGE_STD", np.nan), errors="coerce")
    df["stage_ord"] = bundle.clin.get("STAGE_STD", pd.Series(index=bundle.clin.index)).astype(str).str.upper().map(stage_map)
    df["grade_ord"] = bundle.clin.get("GRADE_STD", pd.Series(index=bundle.clin.index)).astype(str).str.upper().map(grade_map)
    df = df.dropna(subset=["T", "E", "risk_score"])

    # Impute optional clinical covariates by median if present.
    for col in ["age", "stage_ord", "grade_ord"]:
        if col in df and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = np.nan

    covars = ["risk_score"] + [c for c in ["age", "stage_ord", "grade_ord"] if df[c].notna().any()]
    fit_df = df[["T", "E"] + covars].copy()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(fit_df, duration_col="T", event_col="E", show_progress=False)
    s = cph.summary.reset_index().rename(columns={"index": "variable"})
    keep_cols = ["variable", "coef", "exp(coef)", "se(coef)", "z", "p", "coef lower 95%", "coef upper 95%"]
    return s[[c for c in keep_cols if c in s.columns]]


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "benchmark.log")

    try:
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")
        seed_everything(args.seed)
        sksurv_ref = require_sksurv()
        torch_ref = require_torch()
        seeds = parse_seed_list(args.seeds)

        logger.info("Loading prepared matrices from %s", args.input_dir)
        bundle = load_prepared_data(Path(args.input_dir), args.endpoint, logger)

        if np.sum(bundle.event) < args.outer_folds:
            raise ValueError("Not enough events for requested outer folds. Reduce --outer-folds.")

        fold_frames = []
        auc_frames = []
        param_frames = []
        for s in seeds:
            logger.info("Running repeated nested CV for seed=%s", s)
            args_seed = argparse.Namespace(**vars(args))
            args_seed.seed = int(s)
            fold_df_s, auc_df_s, params_df_s = run_nested_cv(bundle, args_seed, logger, sksurv_ref, torch_ref)
            if not fold_df_s.empty:
                fold_df_s["seed"] = s
                fold_frames.append(fold_df_s)
            if not auc_df_s.empty:
                auc_df_s["seed"] = s
                auc_frames.append(auc_df_s)
            if not params_df_s.empty:
                params_df_s["seed"] = s
                param_frames.append(params_df_s)

        fold_df = pd.concat(fold_frames, axis=0, ignore_index=True) if fold_frames else pd.DataFrame()
        auc_df = pd.concat(auc_frames, axis=0, ignore_index=True) if auc_frames else pd.DataFrame()
        params_df = pd.concat(param_frames, axis=0, ignore_index=True) if param_frames else pd.DataFrame()
        if fold_df.empty:
            raise RuntimeError("Benchmark produced no valid fold-level results.")

        summary_df = build_summary(fold_df)
        # Save both legacy and required naming.
        fold_df.to_csv(output_dir / "benchmark_fold_level_metrics.csv", index=False)
        fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
        auc_df.to_csv(output_dir / "benchmark_time_dependent_auc.csv", index=False)
        params_df.to_csv(output_dir / "benchmark_selected_hyperparams.csv", index=False)
        summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False)

        save_plots(fold_df, auc_df, output_dir / "plots")
        # Freeze models using first configured seed to avoid stochastic artifact drift.
        args_freeze = argparse.Namespace(**vars(args))
        args_freeze.seed = int(seeds[0])
        freeze_models_and_export_representation(
            bundle,
            params_df,
            summary_df,
            args_freeze,
            logger,
            sksurv_ref,
            torch_ref,
            output_dir / "artifacts",
        )

        # Required tables derived from nested CV.
        time_auc_summary = (
            auc_df.groupby(["model", "time"], as_index=False).agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"), n=("auc", "count"))
            if not auc_df.empty
            else pd.DataFrame()
        )
        time_auc_summary.to_csv(output_dir / "time_auc_summary.csv", index=False)

        ibs_summary = fold_df.groupby("model", as_index=False).agg(ibs_mean=("ibs", "mean"), ibs_std=("ibs", "std"), n=("ibs", "count"))
        ibs_summary.to_csv(output_dir / "ibs_summary.csv", index=False)

        # Risk, latent representation, locked cutoffs.
        risk_path = output_dir / "artifacts" / "full_cohort_risk_scores.csv"
        rep_path = output_dir / "artifacts" / "best_integrated_representation.csv"
        if risk_path.exists():
            shutil.copy2(risk_path, output_dir / "risk_scores_all_patients.csv")
            risk_df = pd.read_csv(risk_path, index_col=0)
            cut = locked_cutoff_table(risk_df)
            (output_dir / "locked_cutoffs.json").write_text(json.dumps(cut, indent=2), encoding="utf-8")

            sel_json = output_dir / "artifacts" / "model_selection.json"
            best_main = ""
            if sel_json.exists():
                best_main = json.loads(sel_json.read_text(encoding="utf-8")).get("best_main_model", "")

            mcox = multivariable_cox_table(bundle, risk_df, best_main)
            mcox.to_csv(output_dir / "multivariable_cox.csv", index=False)

            horizons = [12.0, 24.0, 36.0]
            calib = calibration_summary(bundle, risk_df, horizons)
            calib.to_csv(output_dir / "calibration_summary.csv", index=False)
            dca = dca_summary(bundle, risk_df, horizons)
            dca.to_csv(output_dir / "dca_summary.csv", index=False)
        else:
            pd.DataFrame().to_csv(output_dir / "multivariable_cox.csv", index=False)
            pd.DataFrame().to_csv(output_dir / "calibration_summary.csv", index=False)
            pd.DataFrame().to_csv(output_dir / "dca_summary.csv", index=False)
            (output_dir / "locked_cutoffs.json").write_text("{}", encoding="utf-8")

        if rep_path.exists():
            shutil.copy2(rep_path, output_dir / "latent_representation.csv")

        # Required ablation table.
        ablation_df = run_ablation(bundle, args, seeds, logger, sksurv_ref)
        ablation_summary = (
            ablation_df.groupby("modality_combo", as_index=False)
            .agg(cindex_mean=("cindex", "mean"), cindex_std=("cindex", "std"), ibs_mean=("ibs", "mean"), auc_mean=("mean_auc", "mean"), n=("fold", "count"))
            .sort_values("cindex_mean", ascending=False)
            if not ablation_df.empty
            else pd.DataFrame()
        )
        ablation_summary.to_csv(output_dir / "ablation_summary.csv", index=False)

        logger.info("Benchmark complete.")
        return 0

    except Exception as exc:
        logger.exception("Benchmark failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDAC multi-omics benchmark (nested CV).")
    parser.add_argument("--input-dir", required=True, help="Directory containing prepared matrices from Prompt 1.")
    parser.add_argument("--output-dir", required=True, help="Output directory for benchmark results.")
    parser.add_argument("--endpoint", default="OS", choices=["OS", "PFS"], help="Survival endpoint to model.")

    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--auc-time-points", type=int, default=20)
    parser.add_argument(
        "--auc-horizons-months",
        default="12,24,36",
        help="Comma-separated time-dependent AUC horizons in months. Use an empty string to fall back to quantile times.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,52,62,72,82", help="Comma-separated repeated-CV seeds.")
    parser.add_argument("--fast-mode", action="store_true", help="Use fewer epochs for faster benchmarking.")

    parser.add_argument("--expr-top-var", type=int, default=3000)
    parser.add_argument("--expr-pca-dim", type=int, default=80)
    parser.add_argument("--meth-top-var", type=int, default=4000)
    parser.add_argument("--meth-pca-dim", type=int, default=80)
    parser.add_argument("--mut-top-genes", type=int, default=120)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
