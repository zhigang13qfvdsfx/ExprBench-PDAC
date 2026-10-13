# Public data sources and full reanalysis inputs

| Resource | Source | Use |
|---|---|---|
| TCGA-PAAD | https://github.com/cBioPortal/datahub/tree/master/public/paad_tcga_pan_can_atlas_2018 | Discovery clinical, RSEM expression, methylation and mutation inputs |
| PanCanAtlas ABSOLUTE | https://gdc.cancer.gov/about-data/publications/pancanatlas | Purity and histology sensitivity |
| GSE62452 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62452 | 65 eligible independent OS cases |
| GSE85916 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85916 | 79 eligible independent OS cases |
| GSE71729 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71729 | 123 eligible primary-tumor OS cases; channel-2 annotations |
| GSE57495 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57495 | 63 eligible independent OS cases |
| E-MTAB-6134 | https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-6134 | 288 eligible independent OS cases from 309 pathology-reviewed profiles |
| GSE28735 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28735 | Overlapping cancer-death sensitivity, outside independent pooling |
| GSE21501 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE21501 | Screened, excluded for unresolved independence |
| GSE205049 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205049 | Source expression and cell annotations; the local immune projection retains 9,948 cells |
| GDSC | https://www.cancerrxgene.org/ | Drug-reference expression, response and compound annotations |

## Retrieval and preparation

`analysis/prepare_tcga_paad_data.py` contains TCGA source retrieval and preparation. The original external-validation entrypoint contains the GEO workflow. Large prepared matrices are excluded from this upload directory; retain the source scale, identifiers and original endpoint definitions when reconstructing them.

`scripts/fetch_expanded_sources.py` retrieves the GSE57495 series matrix, E-MTAB-6134 processed expression and SDRF/IDF, and extracts the required GEO platform annotations. It stops at the platform-table boundary and does not cache the large associated-sample block. `scripts/prepare_expanded_cohorts.py` also expects the separately prepared legacy GSE71729 matrix at `pdac_project/results/07_external_validation_rescue/GSE71729_expression_processed.csv`; its archived metadata remains supplied. `scripts/fit_expanded_adapter.py` requires the TCGA expression matrix and archived teacher scores. These source-regeneration scripts are optional and download large files.

E-MTAB-6134 uses deposited probe values, primary SDRF OS definitions, and mean unambiguous gene-symbol mapping. The source IDF describes RMA and within-study clinical-center ComBat. GSE28735 overlaps GSE62452. No external outcome-based tuning is part of the saved plan.

For strict byte-level external prediction verification, restore the original `revision_20261012/inference_inputs/<cohort>/selected_expression.csv.gz` inputs from the separate full package and run `python scripts/verify_expanded_external.py --verify-predictions`. The lightweight default verifies estimates from the archived scores rather than recomputing scores from missing large matrices.

Retrieve complete single-cell expression and annotation files and the GDSC reference matrices from their original repositories for full downstream reanalysis. Consult source terms and cite each original study.
