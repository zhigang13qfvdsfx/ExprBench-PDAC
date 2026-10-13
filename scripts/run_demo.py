"""Verify the portable expression model on twelve public TCGA examples."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from predict_risk import predict
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/demo')
    args = parser.parse_args()
    expected = pd.read_csv(ROOT / 'examples/expected_risk_scores.csv', index_col=0)
    expression = pd.read_csv(ROOT / 'examples/expression.csv.gz', index_col=0)
    scores = predict(expression, ROOT / 'models/expr_penalized_cox_portable.npz').reindex(expected.index)
    assert len(scores) == 12 and np.isfinite(scores).all()
    error = float(np.max(np.abs(scores.to_numpy() - expected['Expr-Penalized-Cox'].to_numpy())))
    assert error < 1e-6, error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_frame().to_csv(args.output_dir / 'portable_expression.csv')
    result = {'verified': True, 'patients': len(scores), 'maximum_absolute_difference': error}
    (args.output_dir / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
