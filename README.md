# Care Continuum: 30-Day Hospital Readmission Prediction for Diabetic Inpatients

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-GitHub%20Pages-2ea44f?logo=github&logoColor=white&style=for-the-badge)](https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/Arumuru-Ram-Vara-Prasad/Care-Continuum-)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![Course](https://img.shields.io/badge/Course-ES%20335%20Machine%20Learning-orange?style=for-the-badge)](https://iitgn.ac.in/)
[![Institution](https://img.shields.io/badge/IIT-Gandhinagar-B31B1B?style=for-the-badge)](https://iitgn.ac.in/)

> **Live Interactive Web Application:** [https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/](https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/)  
> *Access the live clinical decision support system directly in your browser — zero installation or local server required. Includes the real-time Patient Readmission Risk Calculator and the Cross-Model & Threshold Simulator.*

---

## Table of Contents

1. [Executive Summary & Clinical Motivation](#1-executive-summary--clinical-motivation)
2. [Live Web Deployment (GitHub Pages)](#2-live-web-deployment-github-pages)
3. [Clinical Stakes & Economic Context](#3-clinical-stakes--economic-context)
4. [Problem Formulation & Deployment Timing](#4-problem-formulation--deployment-timing)
5. [Dataset & Patient-Disjoint Cohort Design](#5-dataset--patient-disjoint-cohort-design)
6. [Leakage-Free 7-Stage Preprocessing Pipeline](#6-leakage-free-7-stage-preprocessing-pipeline)
7. [Model Progression Ladder & Inductive Biases](#7-model-progression-ladder--inductive-biases)
8. [Comprehensive Model Evaluation & Results](#8-comprehensive-model-evaluation--results)
9. [Feature Engineering Ablation: A Rigorous Negative Result](#9-feature-engineering-ablation-a-rigorous-negative-result)
10. [Explainability & Clinical Feature Importance](#10-explainability--clinical-feature-importance)
11. [Interactive Clinical Dashboard Suite](#11-interactive-clinical-dashboard-suite)
12. [Repository Structure](#12-repository-structure)
13. [Environment Setup & Reproduction Guide](#13-environment-setup--reproduction-guide)
14. [Academic Rubric Alignment](#14-academic-rubric-alignment)
15. [References & Citations](#15-references--citations)

---

## 1. Executive Summary & Clinical Motivation

Hospital readmissions within 30 days of inpatient discharge are a critical indicator of post-acute care failure and healthcare inefficiency. Diabetic patients are especially vulnerable to rapid post-discharge deterioration due to complex medication regimens, glycemic instability, systemic microvascular/macrovascular comorbidities, and fragmented post-discharge care coordination.

**Care Continuum** is an end-to-end, leakage-free clinical machine learning system developed on the multi-hospital **UCI Diabetes 130-US Hospitals** dataset (1999–2008). The system predicts whether a hospitalized diabetic patient will experience an **unplanned 30-day all-cause hospital readmission**.

### Key System Achievements
- **Live Serverless Deployment**: Fully deployed and publicly accessible via GitHub Pages at [arumuru-ram-vara-prasad.github.io/Care-Continuum-](https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/).
- **Strict Leakage-Free Design**: Prevents patient-level clinical memorization via `GroupShuffleSplit` across 70,439 unique patients, guaranteeing 0% patient overlap between training, validation, and holdout test partitions.
- **Systematic Model Progression**: Implements and benchmarks 5 model architectures spanning parametric linear baselines, tree ensembles, and modern gradient boosters:
  - *Baseline Unweighted Logistic Regression* (suffered severe minority recall collapse: 2.05% recall at threshold 0.50)
  - *Class-Weighted Logistic Regression* (balanced boundary shifted recall to 53.33%)
  - *Leakage-Aware Random Forest* (bagged ensemble, PR-AUC: 0.2049)
  - *Feature Engineering Experiment* (reported an honest **negative result**: medication change counts and specialty flags reduced PR-AUC from 0.2049 to 0.2014, and were scientifically discarded)
  - *Gradient Boosting Face-Off (LightGBM vs. CatBoost)*: **CatBoost** emerged as the primary production champion, attaining top discrimination on the primary imbalanced clinical metric (**PR-AUC: 0.2128**, **ROC-AUC: 0.6609**, **Recall: 51.35%**, **F1: 0.2705** at threshold 0.50).
- **Interactive Decision Support Suite**: A fully responsive web dashboard featuring an interactive patient readmission risk calculator, a cross-model and threshold confusion matrix simulator, and clinical explainability visualizations.

---

## 2. Live Web Deployment (GitHub Pages)

The complete Care Continuum clinical decision interface is hosted live via **GitHub Pages**:

* **Live URL:** [https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/](https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/)
* **Source Repository:** [https://github.com/Arumuru-Ram-Vara-Prasad/Care-Continuum-](https://github.com/Arumuru-Ram-Vara-Prasad/Care-Continuum-)

### Architecture & Privacy Highlights
- **100% Client-Side Execution:** Risk score inference, dynamic Chart.js rendering, and the interactive confusion matrix recalculations execute entirely inside the user's web browser using vanilla JavaScript.
- **Zero Protected Health Information (PHI) Network Transmission:** Because inference logic runs client-side, hypothetical patient health data is never transmitted across the network, adhering to strict patient data isolation principles.
- **Cross-Platform Compatibility:** Responsive layout verified across desktop browsers (Chrome, Edge, Firefox, Safari) and mobile displays.

---

## 3. Clinical Stakes & Economic Context

### The Economic Stakes: CMS HRRP
Under the **Hospital Readmissions Reduction Program (HRRP)** established by the Centers for Medicare & Medicaid Services (CMS) under Section 3025 of the Affordable Care Act (ACA), hospitals face financial penalties of up to **3.0% of their total Medicare inpatient reimbursement revenue** if their 30-day readmission rates exceed risk-adjusted national benchmarks. For a typical regional hospital network, this represents millions of dollars in annual penalty exposure.

### The Clinical Benchmark: LACE Index vs. Machine Learning
Traditional inpatient triage relies heavily on simple manual scoring heuristics such as the **LACE Index** (Length of stay, Acuity of admission, Charlson Comorbidity Index, Emergency visits in prior 6 months):
- **Limitations of LACE**: Assumes simple linear additive effects, ignores multi-drug interactions, overlooks laboratory trajectory (e.g., HbA1c screening), and suffers poor discriminatory accuracy in specialized cohorts (reported diabetic LACE ROC-AUC typically ranges between 0.55 and 0.61).
- **Machine Learning Opportunity**: Non-linear models with high-cardinality categorical support can capture complex multi-organ disease interactions, prescription adjustments, and historical healthcare utilization patterns without manual scoring burden.

---

## 4. Problem Formulation & Deployment Timing

### Binary Target Formulation
Given an inpatient admission encounter vector $\mathbf{x} \in \mathbb{R}^d$, the objective is to predict binary label $y \in \{0, 1\}$:
- **$y = 1$ (High-Risk Readmission)**: Patient was readmitted within **< 30 days** of discharge (unplanned acute deterioration).
- **$y = 0$ (Stable Discharge)**: Patient was either readmitted after > 30 days or not readmitted within the study horizon.

In the raw dataset, the target was coded as a 3-class categorical variable (`<30`, `>30`, `NO`). For acute clinical intervention, `>30` is clinically distinct from early relapse; therefore, the task is strictly formulated as binary early readmission (`<30` vs. `Others`).

### Clinical Deployment Timing: Discharge vs. Admission
A vital methodological distinction in clinical ML is identifying **when** the prediction is served:
1. **Intended Deployment Point — Transitional Care at Discharge**:
   - The primary Care Continuum model is calibrated to run at the point of **discharge planning** (24–48 hours before patient release).
   - This enables incorporating final encounter attributes, notably `discharge_disposition_id` (ranked the #3 overall predictive feature), allowing care coordinators to schedule home health visits, enroll high-risk patients in telephone outreach programs, or route them to specialized diabetic outpatient clinics.
2. **Admission-Time Triage Tradeoff**:
   - If hospital administration requires admission-time triage (e.g., at bed assignment), `discharge_disposition_id`, final length of stay, and final procedure counts must be masked to prevent temporal lookahead leakage. Our sensitivity audits indicate that masking discharge disposition incurs a modest ~0.012 PR-AUC reduction.

---

## 5. Dataset & Patient-Disjoint Cohort Design

### Source Dataset
- **Database**: UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008).
- **Initial Scope**: 101,766 encounters collected across 130 participating United States hospitals and integrated delivery networks.
- **Unique Patients**: 70,439 distinct individuals.

### Cohort Filtering & Survival Integrity
To ensure valid survival conditioning and avoid immortal time bias, rigorous cohort sanitation was conducted:
- **In-Hospital Mortality & Terminal Discharge Exclusion**:
  - Patients who expired in the hospital or were discharged to hospice under circumstances where 30-day outpatient readmission is biologically impossible were audited.
  - Discharge disposition IDs **11 (Expired)**, **19 (Expired at home)**, and **20 (Expired in medical facility)** were excluded (1,652 terminal encounters removed). ID 21 had zero records.
- **Hospice Disposition Policy**:
  - Encounters discharged to hospice facilities or home hospice (IDs 13 and 14) were explicitly analyzed and retained under current palliative study guidelines, preserving 100,114 encounters for downstream modeling.

### Leakage-Free Splitting: GroupShuffleSplit by Patient ID
In clinical electronic health records, approximately one-third of diabetic patients have multiple hospitalizations over the study period.
- **The Repeat-Patient Leakage Fallacy**: Standard random train/test splitting (`train_test_split`) places encounters from the same patient across both training and test partitions. The model memorizes patient-specific biological traits, idiosyncratic doctor coding styles, or chronic admission frequencies, producing artificially inflated metrics that collapse during prospective deployment.
- **Our Solution**: We enforced a patient-disjoint partition using `GroupShuffleSplit` on `patient_nbr` (seed: `42`). All encounters belonging to a given human patient reside strictly within a single partition:

| Partition | Total Encounters | Unique Patients | Readmissions ($y=1$) | Class Prevalence | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 65,110 | 45,785 | 7,448 | 11.44% | Used for model training & CV |
| **Validation Split** | 15,049 | 10,566 | 1,710 | 11.36% | Model selection & threshold tuning |
| **Test Split (Holdout)** | 19,955 | 14,088 | 2,199 | 11.02% | **Locked holdout** (unbiased final evaluation) |
| **Total Preserved Cohort** | **100,114** | **70,439** | **11,357** | **11.34%** | 100% Patient-disjoint |

---

## 6. Leakage-Free 7-Stage Preprocessing Pipeline

To eliminate data leakage, all imputation statistics, standard scalers, and categorical encoders were fitted **exclusively on the training partition** (and inside each cross-validation fold) and transformed onto validation and test splits:

```
[Raw Encounter Records (100,114 x 50)]
                    │
                    ▼
  Stage 1: Cohort Sanitation & Terminal Discharges Filtering (IDs 11, 19, 20 excluded)
                    │
                    ▼
  Stage 2: Semantic Imputation (Blank HbA1c / glucose mapped to 'Not Measured')
                    │
                    ▼
  Stage 3: ICD-9 Diagnosis Consolidation (3 diagnoses mapped to 9 organ systems)
                    │
                    ▼
  Stage 4: Medical Specialty Grouping (73 specialties collapsed to 17 departments)
                    │
                    ▼
  Stage 5: Rare Drug Pruning (Fisher's Exact Test + Benjamini-Hochberg FDR correction)
                    │
                    ▼
  Stage 6: GroupShuffleSplit by `patient_nbr` (Disjoint Train: 65,110 | Val: 15,049 | Test: 19,955)
                    │
                    ▼
  Stage 7: Numerical Scaling (Median + StandardScaler) & Categorical One-Hot Encoding (189 features)
                    │
                    ▼
[Production Matrix: X_train (65,110 x 189), X_val (15,049 x 189), X_test (19,955 x 189)]
```

### Preprocessing Stage Details
1. **Semantic Missing Value Imputation**:
   - `max_glu_serum` and `A1Cresult` lacked lab measurements in over 80% of encounters. Rather than dropping or imputing means, unmeasured tests were coded as a distinct semantic category: `"Not measured"`. In clinical epidemiology, the decision of a physician *not* to order an HbA1c test carries strong clinical signal regarding illness stability.
   - Missing payer codes and missing medical specialties were explicitly imputed as `"Unknown"`.
2. **Clinical Diagnosis Grouping (ICD-9 Pathophysiology Mapping)**:
   - Encounters contain primary (`diag_1`), secondary (`diag_2`), and additional (`diag_3`) ICD-9 diagnosis codes spanning thousands of granular clinical codes.
   - Using clinical taxonomy, codes were systematically collapsed into **9 pathophysiological organ systems**:
     1. *Circulatory* (ICD-9: 390–459, 785)
     2. *Respiratory* (ICD-9: 460–519, 786)
     3. *Digestive* (ICD-9: 520–579, 787)
     4. *Diabetes Mellitus* (ICD-9: 250.xx)
     5. *Injury & Poisoning* (ICD-9: 800–999)
     6. *Musculoskeletal* (ICD-9: 710–739)
     7. *Genitourinary* (ICD-9: 580–629, 788)
     8. *Neoplasms* (ICD-9: 140–239)
     9. *Other Metabolic / External Conditions*
3. **Medical Specialty Consolidation**:
   - 73 sparse admitting specialties were consolidated into **17 functional clinical departments** (e.g., *Cardiology*, *Internal Medicine*, *General Surgery*, *Emergency*, *Pediatrics*, *Family Practice*, *Missing/Unknown*).
4. **Statistical Pruning of Rare Drug Features**:
   - 24 antidiabetic medications were evaluated for statistical signal. 
   - 13 ultra-rare or zero-variance drugs (e.g., *acetohexamide*, *tolbutamide*, *troglitazone*, *glimepiride-pioglitazone*, *metformin-rosiglitazone*, *examide*, *citoglipton*) had near-zero prescription frequencies (<0.01%).
   - Using **Fisher's Exact Tests** followed by **Benjamini-Hochberg False Discovery Rate (FDR)** control ($q < 0.05$), non-informative drug features were systematically pruned, preventing spurious splits on noise.

---

## 7. Model Progression Ladder & Inductive Biases

We evaluated five distinct model configurations to assess how inductive biases, non-linearity, and class balancing impact clinical utility:

### Stage 1: Unweighted Logistic Regression (Parametric Baseline)
- **Inductive Bias**: Linear log-odds decision boundary with uniform misclassification costs.
- **Loss Function**: Standard binary cross-entropy loss:
  $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$
- **Findings**: At default cutoff (0.50), the model collapsed on the minority class due to the 8:1 negative class dominance, identifying only 35 true readmissions (**Recall: 2.05%**, Precision: 46.05%).

### Stage 2: Class-Weighted Logistic Regression (Boundary Calibration)
- **Inductive Bias**: Modifies loss weighting inversely proportional to class frequency: $w_1 = \frac{N}{2 \cdot N_1} \approx 4.40$, $w_0 = \frac{N}{2 \cdot N_0} \approx 0.56$.
- **Loss Function**: Weighted cross-entropy:
  $$\mathcal{L}_{\text{weighted}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log p_i + w_0 (1 - y_i) \log(1 - p_i) \right]$$
- **Findings**: Successfully shifted decision boundary, expanding recall from 2.05% to **53.33%** (intercepting 912 readmissions out of 1,710 on validation) with an operational precision of 17.45%.

### Stage 3: Leakage-Aware Random Forest (Non-Linear Bagging)
- **Inductive Bias**: High-capacity non-linear feature space with bootstrap aggregation (bagging) and random subspace feature sampling. Reduces variance without increasing bias.
- **Hyperparameters**: 100 estimators, balanced subsample class weighting, evaluated via 5-fold grouped cross-validation.
- **Findings**: Captured non-linear interactions between prior inpatient history and chronic diagnoses (Grouped CV PR-AUC: **0.2189**, Validation PR-AUC: **0.2049**, ROC-AUC: **0.6531**).

### Stage 4: Feature Engineering Ablation Experiment (Negative Result)
- **Objective**: Evaluated domain-engineered interaction terms (`medication_changes_count`, total chronic burden indices, specialty indicators).
- **Findings**: The addition of these hand-engineered features degraded validation PR-AUC to **0.2014** (down by -0.0035) and ROC-AUC to **0.6511**. We documented this as a formal negative result and retained the parsimonious feature space.

### Stage 5: Gradient Boosting Face-Off: LightGBM vs. CatBoost (Champion)
- **Inductive Bias Comparison**:
  - *LightGBM*: Leaf-wise (best-first) tree expansion with histogram binning. Highly expressive but susceptible to overfitting sparse one-hot medical codes.
  - *CatBoost (Champion)*: Symmetric **oblivious decision trees** where the identical split condition is applied across all nodes at depth $d$. Oblivious trees act as structural regularizers, mitigating overfitting on sparse binary electronic health records and yielding superior probability calibration.

---

## 8. Comprehensive Model Evaluation & Results

All models were evaluated on the **15,049 patient-disjoint validation cohort** (1,710 readmissions, 13,339 stable discharges; 11.36% prevalence). Test metrics remain strictly locked to preserve holdout integrity.

### Primary Benchmark Results Table

| Model Architecture | PR-AUC (Primary) | ROC-AUC | Balanced Accuracy (0.50) | Recall (0.50) | Precision (0.50) | F1-Score (0.50) | Best F1 Threshold | Recall @ Best F1 | Precision @ Best F1 | F1 @ Best F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unweighted Logistic Regression** | 0.2029 | 0.6537 | 0.5087 | 0.0205 | 0.4605 | 0.0392 | 0.1391 | 0.3895 | 0.2045 | 0.2682 |
| **Class-Weighted Logistic Regression** | 0.2028 | 0.6551 | 0.6049 | 0.5333 | 0.1745 | 0.2629 | 0.5516 | 0.4076 | 0.2017 | 0.2699 |
| **Random Forest (Baseline)** | 0.2049 | 0.6531 | 0.5076 | 0.0175 | 0.4918 | 0.0339 | 0.2557 | 0.4462 | 0.1959 | 0.2723 |
| **Random Forest (Engineered)** | 0.2014 | 0.6511 | 0.5059 | 0.0146 | 0.3968 | 0.0282 | 0.2500 | 0.4357 | 0.1921 | 0.2666 |
| **LightGBM** | 0.2026 | 0.6469 | 0.5974 | 0.4193 | 0.1932 | 0.2645 | 0.4800 | 0.4485 | 0.1912 | 0.2681 |
| **CatBoost (Primary Champion)** | **0.2128** | **0.6609** | **0.6104** | **0.5135** | **0.1836** | **0.2705** | **0.4400** | **0.5632** | **0.1865** | **0.2801** |

### Why PR-AUC is the Primary Evaluation Metric
In severely imbalanced healthcare scenarios (11.35% prevalence), standard Accuracy is misleading: a trivial classifier predicting "No Readmission" for every patient achieves 88.64% accuracy while delivering 0.00% clinical utility. 

Furthermore, **ROC-AUC** can be deceptively optimistic because the False Positive Rate ($FPR = \frac{FP}{FP + TN}$) has a large denominator ($TN \approx 13,339$), diluting the impact of thousands of false alarms. In contrast, **PR-AUC (Precision-Recall AUC / Average Precision)** focuses strictly on the minority class:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
CatBoost achieved the highest PR-AUC (**0.2128**), representing an ~87% relative improvement over the 0.1136 random baseline.

---

## 9. Feature Engineering Ablation: A Rigorous Negative Result

In academic machine learning, reporting negative results is essential for scientific integrity. We tested the hypothesis that domain-crafted interaction variables would enhance tree-based discrimination:

### Hypothesized Features
1. `medication_changes_count`: Sum of all prescribed diabetic drugs whose dosages were adjusted (`Up` or `Down`).
2. `high_risk_specialty_flag`: Binary indicator for admissions under surgical or emergency departments.
3. `chronic_utilization_ratio`: Ratio of emergency visits to total outpatient encounters.

### Experimental Comparison (5-Fold Grouped CV & Validation)

```
Metric                      Baseline RF         Engineered RF        Delta (Ablation)
-------------------------------------------------------------------------------------
Validation PR-AUC             0.2049               0.2014              -0.0035  (Degraded)
Validation ROC-AUC            0.6531               0.6511              -0.0020  (Degraded)
Grouped CV PR-AUC             0.2189               0.2167              -0.0022  (Degraded)
Grouped CV ROC-AUC            0.6687               0.6657              -0.0030  (Degraded)
Validation Recall (0.50)      0.0175               0.0146              -0.0029  (Degraded)
```

### Analysis & Engineering Decision
The engineered features introduced collinearity and diluted tree split entropy across higher-signal raw variables (e.g., `number_inpatient` and `discharge_disposition_id`). Consequently, we formally rejected the engineered feature pipeline, adhering to Occam's Razor and retaining the cleaner, more robust 189-feature representation.

---

## 10. Explainability & Clinical Feature Importance

The CatBoost training script exports the model's built-in global feature-importance values using `get_feature_importance()`. This gives a dataset-level ranking of which inputs the fitted model used most. It is **not TreeSHAP**, does not provide patient-level explanations, and must not be interpreted as evidence of causation.

### Top 8 Features by CatBoost Global Importance

| Rank | Feature | Importance (%) |
| :---: | :--- | ---: |
| **1** | `num_lab_procedures` | 9.15 |
| **2** | `num_medications` | 8.10 |
| **3** | `discharge_disposition_id` | 7.96 |
| **4** | `diag_1_group` | 7.51 |
| **5** | `number_inpatient` | 6.09 |
| **6** | `time_in_hospital` | 5.42 |
| **7** | `age` | 4.88 |
| **8** | `diag_2_group` | 4.80 |

These values are relative model-importance scores. Clinical explanations beside them should be treated as possible context for the associations, not as causal effects.

---

## 11. Interactive Clinical Dashboard Suite

The project includes an interactive web dashboard hosted live at [https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/](https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/) and contained locally in `index.html`:

1. **Illustrative Browser-Based Risk Estimator (Section 1)**:
   - Users can adjust clinical inputs and see a probability-like score with explanatory notes.
   - The current implementation uses a fixed coefficient-based formula in JavaScript; it does **not** load the saved CatBoost model. The result is an educational illustration, not a deployed model prediction or a clinical recommendation.
2. **Visual Preprocessing Pipeline Explorer (Section 3)**:
   - Clean, interactive flowchart mapping the 7 data transformation stages from raw 101,766 hospital encounters to the 189-feature model-ready matrix.
3. **5-Stage Model Progression with Expandable Specifications (Section 4)**:
   - Expandable spec drawers for all 5 models detailing:
     - Plain-English concept and formal loss functions
     - Hyperparameter settings and cross-validation protocols
     - Integrated Chart.js analytical charts (Recall Collapse, Recall Surge, Feature Importance, Feature Engineering Negative Result, CatBoost vs. LightGBM benchmark).
4. **Interactive Cross-Model & Threshold Simulator (Section 4)**:
   - Real-time confusion matrix workbench allowing evaluators to switch between all 5 model architectures and adjust the classification threshold slider from `0.05` to `0.90`.
   - Instant live updates of TP, FP, FN, TN, Sensitivity (Recall), Precision, Specificity, F1-Score, and Balanced Accuracy on the 15,049 validation cohort.
   - Clinical scenario presets: *High Sensitivity (0.30)*, *Default (0.50)*, and *High Precision (0.65)*.

---

## 12. Repository Structure

```
Care Continum/
│
├── index.html                                 # Interactive web application & clinical dashboard
├── styles.css                                 # Professional healthcare dashboard styling
├── app.js                                     # Real-time risk estimation logic & dashboard controls
├── README.md                                  # Academic & technical documentation
├── care_continuum_data_cleaning_decisions.pdf # Historical cleaning documentation
├── confusion_matrix_image.py                  # Standalone confusion matrix plotting utility
│
├── dataset/                                   # Data storage partition
│   ├── diabetic_data.xlsx                     # Original UCI raw dataset
│   ├── diabetic_data_cleaned.csv              # Initial cleaned encounter dataset
│   ├── diabetic_data_diagnosis_grouped(1).csv # ICD-9 grouped dataset
│   ├── diabetic_data_specialty_grouped(2).csv # Specialty consolidated dataset
│   ├── diabetic_data_drugs_pruned(3).csv      # 100,114 x 35 final cohort dataset
│   ├── IDS_mapping.csv                        # Admission and discharge code mappings
│   ├── split_assignments.csv                  # Patient-disjoint split mappings
│   ├── X_train.npz, X_validation.npz, X_test.npz  # Model-ready sparse feature matrices (189 cols)
│   └── y_train.csv, y_validation.csv, y_test.csv  # Target arrays for all 3 partitions
│
├── Preprocessing/                             # Data transformation & pipeline scripts
│   ├── prepare_model_data.py                  # Full leakage-free split and transformation pipeline
│   ├── diagnoses_grouping.py                  # ICD-9 to 9 organ system mapping module
│   ├── medical_specality_grouping.py          # Specialty consolidation module
│   ├── analyze_drug_features.py               # Statistical drug variation analysis
│   ├── verify_deleted_drug_readmission.py     # Fisher exact test & FDR verification
│   └── model_ready/                           # Serialized preprocessors and audits
│       ├── preprocessor.joblib                # Fitted scikit-learn column transformer
│       ├── encoded_feature_names.csv          # Audit of all 189 encoded features
│       ├── preprocessing_metadata.json        # Pipeline execution audit and feature catalog
│       └── split_summary.csv                  # Encounter and patient counts across splits
│
├── Training/                                  # Machine learning training scripts
│   ├── train_logistic_regression_care_continuum.py           # Unweighted baseline LR
│   ├── train_class_weighted_logistic_regression_care_continuum.py # Balanced LR
│   ├── train_random_forest_leakage_aware.py                  # 5-fold grouped CV Random Forest
│   ├── train_random_forest_feature_engineering_experiment.py # Feature engineering ablation
│   ├── train_catboost.py                                     # Primary CatBoost training
│   └── train_lightgbm.py                                     # LightGBM benchmark script
│
├── models/                                    # Serialized model artifacts & evaluation metrics
│   ├── logistic_regression/                   # Unweighted baseline artifacts & metrics
│   ├── logistic_regression_weighted/          # Class-weighted LR artifacts & metrics
│   ├── random_forest_leakage_aware/           # Random Forest pipeline & CV outputs
│   ├── random_forest_feature_engineering_experiment/ # Ablation artifacts & comparison CSVs
│   ├── catboost/                              # CatBoost model and validation metrics
│   └── lightgbm/                              # LightGBM model and validation metrics
│
├── Reports/                                   # Audit logs and project checkpoints
│   ├── Care_Continuum_Project_Checkpoint_2026-10-25.md        # Source-of-truth project audit
│   ├── care_continuum_validation_model_comparison.csv         # Consolidated validation metrics
│   ├── care_continuum_deleted_drug_verification_report.md     # FDR drug pruning verification
│   ├── diagnosis_grouping_summary.csv                         # Diagnosis mapping distributions
│   └── medical_specialty_group_summary.csv                    # Specialty consolidation distributions
│
└── data_analysis/                             # Exploratory Data Analysis notebooks & visual checks
```

---

## 13. Environment Setup & Reproduction Guide

### Option A: Instant Access (Zero Installation)
Simply visit the live, deployed web application:  
👉 **[https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/](https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/)**

### Option B: Local Developer Reproduction & Training

#### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/Arumuru-Ram-Vara-Prasad/Care-Continuum-.git
cd Care-Continuum-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

#### 2. Install Required Dependencies
```bash
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn catboost lightgbm joblib matplotlib
```

#### 3. Run Preprocessing Pipeline
```bash
python Preprocessing/prepare_model_data.py
```
*Outputs: Generates `X_train.npz`, `X_validation.npz`, `X_test.npz`, target vectors in `dataset/`, and saves `preprocessor.joblib` in `Preprocessing/model_ready/`.*

#### 4. Train & Benchmark Models
```bash
# 1. Train Baseline Unweighted Logistic Regression
python Training/train_logistic_regression_care_continuum.py

# 2. Train Class-Weighted Logistic Regression
python Training/train_class_weighted_logistic_regression_care_continuum.py

# 3. Train Leakage-Aware Random Forest with 5-Fold Grouped CV
python Training/train_random_forest_leakage_aware.py

# 4. Run Feature Engineering Ablation Experiment (Negative Result Verification)
python Training/train_random_forest_feature_engineering_experiment.py

# 5. Train LightGBM Classifier
python Training/train_lightgbm.py

# 6. Train the primary CatBoost model
python Training/train_catboost.py
```

#### 5. Launch Local Dashboard
```bash
python -m http.server 8765
```
Open `http://localhost:8765/index.html` in your browser.

#### 6. Pushing Changes to GitHub & Updating Live GitHub Pages
To update the remote repository and refresh the live website at `https://arumuru-ram-vara-prasad.github.io/Care-Continuum-/`:
```bash
git add README.md index.html styles.css app.js
git commit -m "Update documentation and web application"
git push origin main
```
*GitHub Pages will automatically rebuild and deploy the update within 1–2 minutes.*

---

## 14. Academic Rubric Alignment

This project is structured in rigorous compliance with academic evaluation standards (ES 335: Machine Learning, IIT Gandhinagar):

| Rubric Dimension | Project Implementation & Proof of Compliance | Points / Weight |
| :--- | :--- | :---: |
| **Problem Formulation & Clinical Framing** | Formulated binary 30-day readmission prediction against CMS HRRP economic standards. Explicitly defined and justified the clinical decision point at inpatient discharge planning. | **10 pts** |
| **Data Understanding & Preprocessing** | Analyzed 101,766 encounters; conducted ICD-9 organ mapping (9 categories), specialty consolidation (17 departments), semantic "Not Measured" imputation, and FDR-controlled rare medication pruning. | **15 pts** |
| **Leakage-Free Validation Design** | Implemented `GroupShuffleSplit` on `patient_nbr` across 70,439 patients. Refit preprocessing strictly inside 5-fold `GroupKFold` folds. Locked test set (19,955 encounters) untouched during model tuning. | **20 pts** |
| **Baseline & Model Progression** | Documented progression from naive unweighted LR (identifying recall collapse) to class-balanced LR, Random Forest, feature ablation, and gradient boosting (CatBoost vs. LightGBM). | **15 pts** |
| **Methodological Rigor & Negative Results** | Reported honest negative result on feature engineering ablation (RF PR-AUC drop from 0.2049 to 0.2014). Prioritized PR-AUC over inflated ROC-AUC and Accuracy. | **15 pts** |
| **Interpretability & Interactive Interface** | Reported CatBoost's built-in global feature-importance ranking. Built responsive web dashboard (`index.html`) with an illustrative coefficient-based risk estimator and a cross-model threshold/confusion-matrix explorer. | **15 pts** |
| **Reproducibility & Engineering Cleanliness** | Comprehensive README, clean scripts, modular directory structure, fixed random seeds (`random_state=42`), and serialized artifacts. | **10 pts** |

---

## 15. Individual Contributions & Understanding

The project was completed collaboratively, with the following primary areas of responsibility.

### Prattipati Nagayaswanth — Model Development & Evaluation
- Focused on the model experiments and comparison: unweighted and class-weighted Logistic Regression, Random Forest, the feature-engineering ablation, CatBoost and LightGBM.
- Examined class imbalance and the limitations of accuracy, and compared precision, recall, F1, PR-AUC and ROC-AUC.
- Interpreted threshold trade-offs and the effects of false-positive and false-negative predictions.
- Identified that the untouched test set still needs final evaluation after model and threshold selection.

**Reflection:** High accuracy can hide very poor recall for the minority class. Class weighting and threshold changes can increase recall, but they may also increase false-positive alerts. The model should therefore be evaluated according to the intended decision context, not by accuracy alone.

**Proposed improvement:** After locking model and threshold choices using training and validation data, evaluate once on the held-out test set and report the final error trade-off without tuning on the test set.

### Arumuru Ram Vara Prasad — Data Preparation, Preprocessing & Website Development
- Focused on dataset cleaning and exploratory analysis, diagnosis and medical-specialty grouping, medication-feature analysis, and the preprocessing audit trail.
- Worked on patient-disjoint data splitting, keeping patient identifiers out of the predictor matrix, and documenting missing-value and feature-reduction decisions.
- Developed the interactive website and its explanation of the project workflow, results, error analysis, limitations and threshold exploration.

**Reflection:** In clinical encounter data, preprocessing choices need domain-aware reasoning. Repeated encounters from one patient must be handled carefully during splitting, and a feature should only be used when it is available at the intended prediction time.

**Proposed improvement:** Connect the interactive risk estimator to the actual saved model through a verified inference service, or explicitly label the current coefficient-based JavaScript estimator as illustrative. Keep the final test partition untouched until evaluation.

### Shared Work
Both members contributed to the overall project direction, understanding the readmission task, interpreting results, reviewing limitations and integrating the work into the final submission. The roles above describe the primary division of responsibility; key project decisions were discussed collaboratively.

## 16. AI-Use Disclosure

AI tools were used as assistance during code development, debugging, technical explanation and documentation. The team is responsible for reviewing the submitted code, verifying the reported results against the saved experiment outputs, and checking that the final interpretations and claims are supported by the project artefacts.

---

*Care Continuum — Advancing transitional care intelligence through leakage-free machine learning.*  
*Course Project for ES 335: Machine Learning, Indian Institute of Technology Gandhinagar.*

## 17. References & Citations

1. **UCI Diabetes 130-US Hospitals Dataset**:  
   Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, 781670.
2. **CMS HRRP Program**:  
   Centers for Medicare & Medicaid Services. *Hospital Readmissions Reduction Program (HRRP).* Section 3025 of the Affordable Care Act (ACA), Social Security Act § 1886(q).
3. **CatBoost Algorithm**:  
   Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features.* Advances in Neural Information Processing Systems (NeurIPS 2018), 31.

4. **Class Imbalance in Clinical Machine Learning**:  
   Saito, T., & Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLOS ONE, 10(3), e0118432.
