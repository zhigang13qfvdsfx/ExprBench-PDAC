# ExprBench-PDAC

Lightweight source-code and verification package for CIX-26-0164, corresponding to the 13 October 2026 manuscript revision (v1.1.1).

## What is included

- `analysis/`: original discovery, survival benchmarking, molecular states, immune, drug, single-cell and sensitivity analysis entrypoints.
- `models/`: portable numerical export of the primary expression penalized Cox model; frozen array adapters and their output hashes are under `revision_20261012/frozen_predictions/`.
- Small public sample inputs, archived external risk/survival tables, internal fold metrics and selected downstream summary tables.
- Code to verify predictions and estimates and regenerate Figure 7 and Supplementary Figure S3.

Large expression/methylation matrices, full single-cell inputs, serialized joblib checkpoints, intermediate outputs and image exports are stored separately. This is a code and small-result repository, not a full raw-data mirror. [Data sources](data_sources.md) identify the public inputs.

## Quick verification

Run from the repository root with Python 3.12 or a compatible environment:

```bash
python -m pip install -r requirements.txt
python scripts/verify_package.py
python scripts/run_demo.py
python scripts/verify_expanded_external.py
python scripts/audit_mean_auc.py --root .
```

The demo checks the portable expression model on 12 public TCGA examples. External verification checks frozen model/output hashes and reconstructs cohort HRs, C-indices, fixed/random-effects pooling and all ten leave-one-cohort-out estimates from the supplied risk/survival tables. Expression-to-score reconstruction for the external adapters requires the separately stored selected-expression inputs; use `--verify-predictions` once those inputs are restored. The mean-AUC audit checks all 150 model/fold summaries.

```bash
python scripts/make_expanded_figures.py
python scripts/make_reporting_s3.py
```

Generated files go to ignored `outputs/`. The compact single-cell QC table retains all 9,948 rows, their source order and the three columns used by S3D. `results/summary_tables.json` combines 18 downstream summaries with unchanged CSV cell values and source hashes.

## External assessment

| Cohort | Eligible patients | Deaths |
|---|---:|---:|
| GSE62452 | 65 | 49 |
| GSE85916 | 79 | 57 |
| GSE71729 | 123 | 83 |
| GSE57495 | 63 | 42 |
| E-MTAB-6134 | 288 | 181 |
| Total | 618 | 412 |

All five independent cohorts are retained. Fixed-effect HR/SD is 1.16 and random-effects HR/SD is 1.22, with I²=54.4%. E-MTAB-6134 is approximately null; omission of GSE62452 gives random-effects P=0.0526. GSE28735 is an overlapping cancer-death sensitivity and is excluded from independent pooling; GSE21501 is excluded for unresolved independence. The dated analysis plan and archived results are retained unchanged.

External assessment concerns continuous risk; the two-state partition still requires external validation. Drug diagnostics are training error. Single-cell projection concerns the sampled immune compartment.

## Full source reanalysis

Full reanalysis requires retrieving and preparing the large inputs listed in [data_sources.md](data_sources.md) and installing optional dependencies in `requirements-full.txt`. Consult each `analysis/` entrypoint's `--help`. The source retrieval, expanded-cohort preparation and adapter-fitting scripts remain in `scripts/`; they require the documented source and TCGA preparation files and are separate from the download-free checks above.

Original code is released under the [MIT License](LICENSE). Upstream datasets retain their terms. See [CITATION.cff](CITATION.cff) for the authors and [NOTICE.md](NOTICE.md) for attribution.

中文：此目录只保留代码、轻量模型、示例及关键核对表。大型矩阵、旧图、重复结果与完整压缩包已移到目录外的备份。全流程重跑需另行获取数据；目录自带的小文件可核对示例预测、五队列统计和 Mean AUC。
