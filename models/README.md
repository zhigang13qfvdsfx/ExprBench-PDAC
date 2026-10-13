# Portable expression model

`expr_penalized_cox_portable.npz` preserves the primary archived expression penalized Cox model's genes, imputation medians, scaling parameters, coefficients and offset. `scripts/predict_risk.py` applies the original float32 scaling convention. The 12-example demo compares its predictions with archived scores.

Direct input requires the prepared TCGA RSEM-derived expression scale and all selected genes. Output is relative risk. Platform-specific frozen array adapters are stored under `revision_20261012/frozen_predictions/`; their matrices are supplied separately for full prediction reconstruction.

The large serialized joblib checkpoints are stored outside this lightweight repository. The portable export is unchanged and does not fit a new model. This package does not provide an externally validated two-state classifier or calibrated absolute-risk assay.
