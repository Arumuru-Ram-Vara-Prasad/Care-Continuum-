# Care Continuum: Deleted Drug Feature Readmission Verification Report

## Executive summary

**Conclusion: no deleted medication feature shows convincing statistical evidence of an unusual 30-day readmission association in this report.** All 11 deleted features have Benjamini–Hochberg adjusted p-values of 1.000. Several comparisons show raw rate differences, but the non-`No` groups are often very small, and none meets the report’s combined criteria of at least 30 encounters per group, an absolute difference of at least 2 percentage points, and adjusted p-value below 0.05.

This is a screening result, not proof that the features have no predictive value. It tests crude association, not the incremental value of features in a model adjusted for patient and encounter characteristics.

## Method

- Compared encounters where a drug status was `No` with encounters where the status was anything other than `No` (`Steady`, `Up`, or `Down`).
- Outcome: 30-day readmission (`readmitted_30d = 1`).
- Used two-sided Fisher exact tests and Benjamini–Hochberg adjustment across the deleted drug features.
- Used a 2 percentage-point absolute rate difference as an exploratory effect-size flag and 30 encounters per group as a minimum support screen.
- This analysis uses the uploaded CSV reports and does not modify any dataset.

## Main comparison: `No` versus all non-`No` statuses

| Deleted drug feature | Non-`No` encounters | Readmission rate for `No` | Readmission rate for non-`No` | Difference (non-`No` minus `No`) | Raw p-value | Adjusted p-value | Assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| chlorpropamide | 85 | 11.35% | 5.88% | -5.47 pp | 0.124 | 1.000 | LARGE DIFFERENCE - not statistically established |
| acetohexamide | 1 | 11.34% | 0.00% | -11.34 pp | 1.000 | 1.000 | LOW SAMPLE SIZE - rates are unstable |
| tolbutamide | 21 | 11.35% | 4.76% | -6.58 pp | 0.503 | 1.000 | LOW SAMPLE SIZE - rates are unstable |
| acarbose | 307 | 11.35% | 9.12% | -2.23 pp | 0.242 | 1.000 | LARGE DIFFERENCE - not statistically established |
| miglitol | 38 | 11.34% | 10.53% | -0.82 pp | 1.000 | 1.000 | NO STRONG ANOMALY DETECTED |
| troglitazone | 3 | 11.34% | 0.00% | -11.34 pp | 1.000 | 1.000 | LOW SAMPLE SIZE - rates are unstable |
| tolazamide | 39 | 11.35% | 7.69% | -3.65 pp | 0.618 | 1.000 | LARGE DIFFERENCE - not statistically established |
| glipizide-metformin | 13 | 11.34% | 7.69% | -3.65 pp | 1.000 | 1.000 | LOW SAMPLE SIZE - rates are unstable |
| glimepiride-pioglitazone | 1 | 11.34% | 0.00% | -11.34 pp | 1.000 | 1.000 | LOW SAMPLE SIZE - rates are unstable |
| metformin-rosiglitazone | 2 | 11.34% | 0.00% | -11.34 pp | 1.000 | 1.000 | LOW SAMPLE SIZE - rates are unstable |
| metformin-pioglitazone | 1 | 11.34% | 0.00% | -11.34 pp | 1.000 | 1.000 | LOW SAMPLE SIZE - rates are unstable |

## Key findings

1. **Chlorpropamide:** 5 early readmissions among 85 non-`No` encounters (5.88%) versus 11.35% among `No` encounters, a difference of −5.47 percentage points. The raw p-value is 0.124 and adjusted p-value is 1.000. This is not statistically established.
2. **Acarbose:** 28 early readmissions among 307 non-`No` encounters (9.12%) versus 11.35% among `No` encounters, a difference of −2.23 percentage points. Raw p-value 0.242; adjusted p-value 1.000. No strong evidence of an anomaly.
3. **Tolazamide:** 3 early readmissions among 39 non-`No` encounters (7.69%), a difference of −3.65 percentage points. The group is small; raw p-value 0.618 and adjusted p-value 1.000.
4. **Tolbutamide:** 1 early readmission among only 21 non-`No` encounters (4.76%). The observed difference is large (−6.58 percentage points), but the group is below the minimum support threshold and the raw p-value is 0.503.
5. **Very rare drugs / combination drugs:** acetohexamide (1 non-`No` encounter), troglitazone (3), glimepiride-pioglitazone (1), metformin-rosiglitazone (2), and metformin-pioglitazone (1) have too few non-`No` observations to support a reliable rate comparison.

## Status-level checks: `Steady`, `Up`, and `Down`

The detailed report contains some visually high rates in very small categories. These should **not** be interpreted as reliable anomalies:

| Drug | Status | Early readmissions / encounters | Observed readmission rate | Interpretation |
|---|---|---:|---:|---|
| chlorpropamide | down | 0/1 | 0.00% | Too few encounters for a stable estimate. |
| chlorpropamide | up | 0/6 | 0.00% | Too few encounters for a stable estimate. |
| acarbose | down | 1/3 | 33.33% | Too few encounters for a stable estimate. |
| acarbose | up | 2/10 | 20.00% | Too few encounters for a stable estimate. |
| miglitol | down | 2/5 | 40.00% | Too few encounters for a stable estimate. |
| miglitol | up | 0/2 | 0.00% | Too few encounters for a stable estimate. |
| tolazamide | up | 0/1 | 0.00% | Too few encounters for a stable estimate. |

For example, acarbose `Down` is 1/3 (33.33%), acarbose `Up` is 2/10 (20.00%), and miglitol `Down` is 2/5 (40.00%). These rates look high, but they are based on only 3, 10, and 5 encounters respectively. A single additional outcome could change each rate substantially.

## Important limitations

- **Small and imbalanced comparison groups:** many non-`No` groups contain fewer than 40 encounters, and some contain fewer than five.
- **Multiple comparisons:** checking many drugs creates a risk of false positives; the adjusted p-values are all 1.000 in this report.
- **Confounding:** prescription status is not randomly assigned. Differences may reflect patient severity, other diagnoses, or clinical decisions, rather than an effect of the medication itself.
- **Association is not model usefulness:** these tests do not establish whether a drug improves prediction after preprocessing or adds information beyond other features.
- **This test only checks the removed features:** it does not validate all medication features, nor does it assess downstream model performance.

## Recommendation for Care Continuum

Based on this verification alone, there is **no strong statistical reason to restore any of these 11 deleted features immediately**. Keep a record of them as removed by the low-variation rule. Later, when the patient-grouped training/validation pipeline is ready, compare model performance with and without the medication feature set. That experiment is the appropriate way to determine whether rare medication information contributes predictive value.

**Do not conclude that the medications are clinically unimportant.** The result only says that this small exploratory comparison did not establish an unusual readmission-rate association for the deleted features.
