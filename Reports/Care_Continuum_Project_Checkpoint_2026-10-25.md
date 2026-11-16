# Care Continuum — Project Checkpoint
**Checkpoint date:** 2026-10-25  
**Purpose:** Single source of truth for project continuity. Continue from this state; do not restart completed work without evidence that files or decisions have changed.

## 1. Project and goal
- Project: **Care Continuum**, hospital readmission prediction using the UCI Diabetes 130-US Hospitals dataset.
- Project folder on Windows: `C:\Users\AGM.PRASAD\OneDrive\Desktop\Care Continum` (folder spelling is “Continum”).
- Target: `readmitted_30d` = 1 when original `readmitted == "<30"`, otherwise 0.
- The goal at this stage is model evaluation/improvement, not initial data cleaning.

## 2. Current dataset / cohort history
- Current pruned dataset filename: `dataset/diabetic_data_drugs_pruned(3).csv`.
- Reported shape: **100,114 encounters × 35 columns**.
- Reported unique patients: **70,439**.
- Target counts: **11,357 positive (11.34%)**, **88,757 negative (88.66%)**.
- **Death-related discharge filtering is already done.** Project history records excluding discharge disposition IDs **11, 19, and 20**; an earlier audit reportedly removed 1,652 encounters. ID 21 had no records in that check. These are historical project records and have not been independently re-counted from the currently available CSV in this checkpoint.
- **Hospice-related encounters are still retained** under the existing policy; IDs **13 and 14** were explicitly retained. Hospice handling is an open decision. Do not redo death filtering.
- Diagnosis codes have already been consolidated into grouped categorical features: `diag_1_group`, `diag_2_group`, `diag_3_group`.
- Eleven low-signal drug columns were removed after earlier analysis; two constant drug columns had also been removed. The preprocessing script defensively excludes a list of 13 drug columns.

## 3. Split and leakage controls
- Existing saved split sizes: **train 65,110**, **validation 15,049**, **test 19,955**.
- Split is by `patient_nbr`; patient groups are disjoint across train/validation/test. Existing metadata says alignment was verified.
- Multiple encounters for a patient may remain, but all encounters for that patient are assigned to a single split. Do not automatically drop all but the first encounter; that would change the cohort and study design.
- `patient_nbr` is a grouping identifier, not a model feature. `encounter_id` is also excluded from features.
- The leakage-aware Random Forest uses 5-fold `GroupKFold` by patient inside the training split. Imputation/scaling/encoding are refitted inside each CV fold.
- **The test set has not been evaluated**; test metrics must remain untouched until model and threshold selection are complete.

## 4. Current preprocessing behavior
- Numeric columns: median imputation + `StandardScaler`.
- Categorical columns: constant imputation to `Unknown` + `OneHotEncoder(handle_unknown="ignore")`.
- `max_glu_serum` and `A1Cresult` blank/`None` values are mapped to the semantic category “Not measured”.
- Raw categorical strings are not being passed directly into Logistic Regression / Random Forest; encoding is already implemented.
- Current Random Forest metadata records **189 encoded features**.

## 5. Models and metrics already obtained
### Unweighted Logistic Regression — validation, threshold 0.5
- Accuracy: 0.88597
- Balanced accuracy: 0.50870
- Precision: 0.46053
- Recall: 0.02047
- F1: 0.03919
- PR-AUC / Average Precision: 0.20286
- ROC-AUC: 0.65367
- Confusion matrix (labels 0 then 1): `[[13298, 41], [1675, 35]]`
- Validation F1-max threshold was approximately 0.139; precision 20.45%, recall 38.95%, F1 0.268. This was tuned and measured on the same validation data, so it is exploratory/optimistic.

### Class-weighted Logistic Regression — validation, threshold 0.5
- Accuracy: 0.66024
- Balanced accuracy: 0.60492
- Precision: 0.17448
- Recall: 0.53333
- F1: 0.26294
- PR-AUC / Average Precision: 0.20277
- ROC-AUC: 0.65513
- Confusion matrix: `[[9024, 4315], [798, 912]]`
- Validation F1-max threshold approximately 0.5516: precision 20.17%, recall 40.76%, F1 0.2699. Threshold selected on same validation set; exploratory only.

### Leakage-aware Random Forest — 5-fold grouped CV, threshold 0.5
- Samples: 65,110; positive rate 11.439%; unique patients 45,785.
- Accuracy: 0.88521
- Balanced accuracy: 0.50515
- Precision: 0.43810
- Recall: 0.01235
- F1: 0.02403
- PR-AUC / Average Precision: 0.21887
- ROC-AUC: 0.66868
- Confusion matrix: `[[57544, 118], [7356, 92]]`
- Metadata explicitly confirms grouped 5-fold CV, preprocessing refit inside folds, patient grouping by `patient_nbr`, and no test metrics computed.

### Leakage-aware Random Forest — validation, threshold 0.5
- Samples: 15,049; positive rate 11.363%.
- Accuracy: 0.88630
- Balanced accuracy: 0.50761
- Precision: 0.49180
- Recall: 0.01754
- F1: 0.03388
- PR-AUC / Average Precision: 0.20490
- ROC-AUC: 0.65315
- Confusion matrix: `[[13308, 31], [1680, 30]]`
- Validation F1-max threshold sweep: threshold about 0.2557; precision 19.59%, recall 44.62%, F1 0.2723. Threshold selected and evaluated on same validation set; exploratory only.

### Current comparison at validation-selected F1 thresholds
- Unweighted LR: threshold ~0.1391, precision 20.45%, recall 38.95%, F1 0.2682.
- Weighted LR: threshold ~0.5516, precision 20.17%, recall 40.76%, F1 0.2699.
- Random Forest: threshold ~0.2557, precision 19.59%, recall 44.62%, F1 0.2723.
- These are not unbiased final performance estimates because threshold selection and measurement used the same validation predictions. The F1 differences are small; do not declare Random Forest decisively better based on this comparison.

## 6. Key remaining decisions / next steps
1. **Define intended prediction time** (e.g., at admission vs at discharge). `discharge_disposition_id` is currently included as a categorical feature according to Random Forest metadata. If predicting at admission, this end-of-encounter feature is unavailable and should be excluded. If predicting at discharge, state that explicitly and assess suitability.
2. **Decide hospice cohort policy** for IDs 13 and 14 after checking the exact official mapping and target-population implications. Death filtering has already been done and should not be repeated.
3. Consider, but do not assume benefit from, a medication engineering experiment: `medication_changes_count` counting retained medication columns whose value is `Up` or `Down`. Verify exact retained columns, preserve original features in the first experiment, and compare with identical patient-grouped CV.
4. `has_specialty` may be tested as an optional feature, but “Unknown” categories are already imputed and one-hot encoded. This is not an unhandled-string problem.
5. LightGBM has been proposed as a future model but **has not yet been trained**.
6. Keep test metrics untouched until model choice and threshold policy are locked.

## 7. Current interpretation
- PR-AUC is around 0.203–0.205 on validation for the three models; grouped CV Random Forest PR-AUC is 0.2189.
- Default threshold 0.5 gives extremely low recall for unweighted LR and Random Forest. Class weighting changes operating behavior substantially.
- Accuracy around 88.6% is not informative by itself due to class imbalance; prioritize PR-AUC, recall, precision, F1, calibration/threshold behavior, and clinical use context.
- Do not train all models at once or regenerate all preprocessing by default. Continue with incremental experiments on the existing pipeline.

## 8. Important source/script context
- Existing preprocessing script: `prepare_model_data(1).py` (previously uploaded as `/mnt/data/prepare_model_data(1).py` in the working environment). It creates patient-level splits, fold-independent artifacts, and fit/transform preprocessing correctly for the original split pipeline.
- New leakage-aware Random Forest script: `train_random_forest_leakage_aware.py` (provided previously; syntax checked but not executed inside the user's Windows environment by the assistant). It validates split alignment and patient disjointness, runs grouped CV with preprocessing inside each fold, fits on the training split, and evaluates validation only.
- Current Random Forest output files previously supplied: `grouped_cv_metrics.json`, `grouped_cv_predictions.csv`, `run_metadata.json`, `validation_classification_report.csv`, `validation_confusion_matrix.csv`, `validation_metrics(2).json`, and `validation_predictions(2).csv`.

## Working rule for future assistance
Before suggesting any cleaning or modeling step, first classify it as **already completed**, **partially completed**, or **not done** using this checkpoint. Clearly separate remembered project decisions from facts independently verified in files. Do not ask the user to paste the entire chat again when this checkpoint and available project context are sufficient.
