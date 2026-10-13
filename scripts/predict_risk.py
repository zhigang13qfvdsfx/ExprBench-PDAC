"""Apply the portable export of the archived expression penalized Cox model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def predict(expression: pd.DataFrame, model_path: Path) -> pd.Series:
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Sample identifiers and gene columns must be unique.")
    with np.load(model_path, allow_pickle=False) as weights:
        genes = weights["genes"].astype(str).tolist()
        missing = sorted(set(genes) - set(expression.columns))
        if missing:
            raise ValueError(f"Missing {len(missing)} required gene columns; examples: {missing[:8]}")
        values = expression.loc[:, genes].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Expression contains infinite values.")
        values = np.where(np.isnan(values), weights["imputer_median"], values)
        scaled = ((values - weights["scaler_mean"]) / weights["scaler_scale"]).astype(np.float32)
        risk = scaled @ weights["coef"] - float(weights["offset"])
    return pd.Series(risk, index=expression.index, name="Expr-Penalized-Cox")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, required=True,
                        help="CSV or CSV.gz; rows=samples, columns=gene symbols, first column=sample ID.")
    parser.add_argument("--model", type=Path, default=ROOT / "models/expr_penalized_cox_portable.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expression = pd.read_csv(args.expression, index_col=0)
    scores = predict(expression, args.model)
    scores.index.name = "PATIENT_ID"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_frame().to_csv(args.output)
    print(f"Wrote {len(scores)} relative-risk scores to {args.output}")


if __name__ == "__main__":
    main()
