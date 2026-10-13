from pathlib import Path
import json, sys, argparse
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RESULTS = ROOT / 'pdac_project/results'
if not RESULTS.exists():
    RESULTS = ROOT / 'baseline_result_tables'
parser = argparse.ArgumentParser(description='Regenerate current Supplementary Figure S3.')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reporting_figures')
OUT = parser.parse_args().output_dir
OUT.mkdir(parents=True, exist_ok=True)
audit = json.loads((ROOT / 'revision_20261012/expanded_external/audit_results.json').read_text(encoding='utf-8'))
stats = pd.read_csv(RESULTS / '06_immune/immune_feature_statistics.csv')
train = pd.read_csv(RESULTS / '08_drug/drug_model_training_stats.csv')
sc = pd.read_csv(RESULTS / '09_singlecell/sc_cell_annotations.csv.gz')
BLUE, ROSE = '#326B8E', '#8C5C65'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
names = {'RISK_SCORE': 'Expression risk score', 'TIDE_DYSFUNCTION_PROXY': 'T-cell dysfunction proxy',
    'TIDE_EXCLUSION_PROXY': 'T-cell exclusion proxy', 'T_CELL_INFLAMED': 'T-cell inflamed signature',
    'CYTOTOXIC': 'Cytotoxic signature', 'CYTOLYTIC_SCORE': 'Cytolytic score', 'STROMAL_SIGNATURE': 'Stromal signature',
    'IFNG_SIGNATURE': 'Interferon gamma signature', 'HLA__NLRC5': 'NLRC5',
    'CHECKPOINT__PDCD1': 'PDCD1 (PD-1)', 'CHECKPOINT__HAVCR2': 'HAVCR2 (TIM-3)',
    'CHECKPOINT__CD274': 'CD274 (PD-L1)', 'CHECKPOINT__CTLA4': 'CTLA4', 'CHECKPOINT__TIGIT': 'TIGIT'}
fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
ax = axs[0, 0]
extra = stats.sort_values('fdr').head(14).iloc[::-1]
labels = [names.get(feature, feature.replace('CHECKPOINT__', '').replace('HLA__', '').replace('_', ' ')) for feature in extra.feature]
values = -np.log10(np.clip(extra.fdr, 1e-300, 1))
ax.hlines(np.arange(len(extra)), 0, values, color='#CED6DD')
ax.scatter(values, np.arange(len(extra)), s=40, color=BLUE)
ax.set_yticks(np.arange(len(extra)), labels, fontsize=9)
ax.set_xlabel('-log10(FDR)')
ax.set_title('A  Additional expression associations', loc='left', fontweight='bold')
ax = axs[0, 1]
ax.scatter(train.train_n, train.cv_mse, s=38, color=BLUE, edgecolors='white', linewidth=.4)
ax.set_xlabel('Number of training cell lines')
ax.set_ylabel('Training mean squared error')
ax.set_title('B  Drug model training diagnostics', loc='left', fontweight='bold')
ax = axs[1, 0]
offsets = {'GSE62452': (6, 6), 'GSE85916': (6, -12), 'GSE28735': (6, 9), 'GSE71729': (8, 6), 'GSE57495': (6, 7), 'E-MTAB-6134': (-8, 7)}
for row in audit['external']:
    overlap = row['cohort'] == 'GSE28735'
    ax.scatter(row['n_survival'], row['c_index'], s=55, facecolors='none' if overlap else BLUE, edgecolors=ROSE if overlap else BLUE)
    ax.annotate(row['cohort'] + (' (overlap)' if overlap else ''), (row['n_survival'], row['c_index']),
        xytext=offsets[row['cohort']], textcoords='offset points', ha='right' if row['cohort'] == 'E-MTAB-6134' else 'left', fontsize=8.5)
ax.axhline(.5, color='gray', linestyle='--')
ax.set_xlim(20, 320)
ax.set_ylim(.485, .66)
ax.set_xlabel('Survival-eligible tumor patients')
ax.set_ylabel('Harrell C-index')
ax.set_title('C  Expanded external performance', loc='left', fontweight='bold')
ax = axs[1, 1]
qc = sc.sample(min(len(sc), 1800), random_state=7)
for state, color, label in [('PDAC', BLUE, 'PDAC'), ('AdjNorm', '#C98262', 'Adjacent non-tumor')]:
    data = qc[qc.DiseaseState == state]
    ax.scatter(data.nCount_RNA, data.nFeature_RNA, s=8, alpha=.4, c=color, label=label, linewidths=0)
ax.set_xscale('log')
ax.set_xlabel('Total RNA counts per cell')
ax.set_ylabel('Detected genes per cell')
ax.legend(frameon=False, fontsize=9)
ax.set_title('D  Single-cell quality control', loc='left', fontweight='bold')
fig.tight_layout(pad=2.2)
fig.savefig(OUT / 'Supplementary_Figure_S3_Revised_20261013.png', dpi=300, facecolor='white')
plt.close(fig)
(OUT / 'S3_label_map.json').write_text(json.dumps(dict(zip(extra.feature, labels)), indent=2), encoding='utf-8')
print(json.dumps({'figure': str(OUT / 'Supplementary_Figure_S3_Revised_20261013.png'), 'labels': labels}))
