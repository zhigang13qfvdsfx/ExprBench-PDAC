"""Reconstruct the archived survival-weighted mean AUC from held-out folds."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def km_at(time, event, horizons):
    survival, event_times, probabilities = 1.0, [], []
    for t in np.sort(np.unique(time[event == 1])):
        survival *= 1.0 - np.sum((time == t) & (event == 1)) / np.sum(time >= t)
        event_times.append(t)
        probabilities.append(survival)
    values = []
    for t in horizons:
        before = np.searchsorted(event_times, t, side='right') - 1
        values.append(1.0 if before < 0 else probabilities[before])
    return np.asarray(values)

def audit(root, output):
    revision = root / 'revision_20261013'
    clinical = pd.read_csv(revision / 'tcga_os_for_auc_audit.csv').sort_values('PATIENT_ID')
    clinical = clinical.set_index('PATIENT_ID')
    table_folder = root / 'pdac_project/results/02_benchmark'
    if not table_folder.exists():
        table_folder = root / 'baseline_result_tables/02_benchmark'
    stored = pd.read_csv(table_folder / 'benchmark_fold_level_metrics.csv')
    by_time = pd.read_csv(table_folder / 'benchmark_time_dependent_auc.csv')
    summary = pd.read_csv(table_folder / 'benchmark_summary.csv')
    horizons = sorted(by_time.time.unique())
    assert horizons == [12.0, 24.0, 36.0]
    assert len(clinical) == 166 and clinical.OS_EVENT.sum() == 88
    reconstructed = []
    for seed in sorted(stored.seed.unique()):
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
        for fold, (_, test) in enumerate(splits.split(np.arange(len(clinical)), clinical.OS_EVENT), 1):
            test_clinical = clinical.iloc[test]
            survival = km_at(test_clinical.OS_TIME.to_numpy(), test_clinical.OS_EVENT.to_numpy(), horizons)
            weights = -np.diff(np.r_[1.0, survival]) / (1.0 - survival[-1])
            rows = stored.loc[(stored.seed == seed) & (stored.fold == fold)]
            assert len(rows) == 6
            for row in rows.itertuples():
                aucs = by_time.loc[(by_time.seed == seed) & (by_time.fold == fold) & (by_time.model == row.model)].sort_values('time').auc.to_numpy()
                assert len(aucs) == 3
                value = float(np.dot(aucs, weights))
                reconstructed.append({'seed': int(seed), 'fold': fold, 'model': row.model,
                    'n_test': len(test), 'weight_12mo': weights[0], 'weight_24mo': weights[1], 'weight_36mo': weights[2],
                    'reconstructed_mean_auc': value, 'archived_mean_auc': row.mean_auc,
                    'absolute_difference': abs(value - row.mean_auc)})
    reconstructed = pd.DataFrame(reconstructed)
    assert len(reconstructed) == 150
    assert reconstructed.absolute_difference.max() < 1e-12
    rows = []
    for record in summary.itertuples():
        values = reconstructed.loc[reconstructed.model == record.model]
        time_means = by_time.loc[by_time.model == record.model].groupby('time').auc.mean()
        value = values.reconstructed_mean_auc.mean()
        assert abs(value - record.mean_auc_mean) < 1e-12
        rows.append({'model': record.model, 'n_folds': len(values), 'survival_weighted_mean_auc': value,
            'archived_mean_auc': record.mean_auc_mean, 'arithmetic_mean_of_horizon_auc': time_means.mean(),
            'auc_12mo': time_means.loc[12.0], 'auc_24mo': time_means.loc[24.0], 'auc_36mo': time_means.loc[36.0]})
    output.mkdir(parents=True, exist_ok=True)
    reconstructed.to_csv(output / 'mean_auc_fold_reconstruction.csv', index=False, float_format='%.17g')
    pd.DataFrame(rows).to_csv(output / 'mean_auc_definition_summary.csv', index=False, float_format='%.17g')
    verification = {'verified': True, 'patients': len(clinical), 'events': int(clinical.OS_EVENT.sum()),
        'model_folds': len(reconstructed), 'maximum_absolute_difference': float(reconstructed.absolute_difference.max()),
        'definition': 'Within each held-out fold, AUC at 12/24/36 months is weighted by successive decrements of its Kaplan-Meier survival curve, with survival initialized at 1 and weights normalized by 1-S(36). Mean AUC is the arithmetic mean of these 25 fold-specific weighted summaries. Each horizon column separately averages that horizon over folds.',
        'primary_implementation': 'https://github.com/sebp/scikit-survival/blob/master/sksurv/metrics.py',
        'methods_documentation': 'https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html'}
    (output / 'mean_auc_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    audit(args.root, args.output_dir or args.root / 'outputs/reporting_audit')
