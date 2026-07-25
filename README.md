# Advanced Data Mining for Data-Driven Insights and Predictive Modeling

## Project Deliverable 1: Data Collection, Cleaning, and Exploration

**Course:** 2026 Summer - Advanced Big Data and Data Mining<br>
**Course Number:** MSCS-634-M20<br>
**Residency:** July 24-26, 2026

## Group Members

1. Ashish Mahajan
2. Murali Krishna Vattikunta
3. Sreesh Sattiyamourthy
4. Sreeharsha Varma Tinnanuri

## Project Overview

This residency project develops a complete data-mining workflow for credit-card transaction analysis. Deliverable 1 establishes the data foundation through schema inspection, exact-duplicate cleaning, extreme-value diagnostics, class-imbalance analysis, exploratory feature engineering, and focused visualizations. Later deliverables will build regression, classification, clustering, and discretized pattern-mining analyses from this shared foundation.

## Dataset

The project uses only the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The original dataset contains transaction-level numerical records with:

- `Time`: seconds elapsed since the first transaction
- `V1` through `V28`: anonymized PCA-transformed numerical features
- `Amount`: transaction amount
- `Class`: `0` for legitimate and `1` for fraudulent

The dataset exceeds the project requirements of 500 records and 8-10 attributes. It supports regression, classification, clustering, and exploratory association-rule mining. Its severe class imbalance reflects an important real-world fraud-detection challenge, while anonymization limits direct interpretation of the original financial variables.

## Dataset Requirement

The raw CSV is required locally but is not committed to Git.

The notebook uses the following order:

1. Load `data/creditcard.csv` when it already exists.
2. Otherwise, load `creditcard.csv` from the repository root.
3. If neither exists, use:

```python
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)
```

The notebook locates `creditcard.csv` in that KaggleHub download, copies it to the ignored local path below, and validates the expected schema:

```text
data/creditcard.csv
```

Confirm the local copy:

```bash
ls -lh data/creditcard.csv
```

Confirm Git ignores the raw dataset:

```bash
git check-ignore -v data/creditcard.csv
```

The raw CSV, `kaggle.json`, and authentication tokens must not be committed. No substitute dataset is used.

No retail, university, demographic, synthetic-customer, or unrelated dataset is joined to these records.

## Repository Structure

```text
MSCS_634_ProjectDeliverable_1/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- Project_Deliverable_1.ipynb
|-- Project_Deliverable_2.ipynb
|-- Project_Deliverable_3.ipynb
|-- Project_Deliverable_4.ipynb
|-- data/
|   |-- README.md
|   `-- creditcard.csv                     # Local only; excluded from Git
|-- src/
|   |-- __init__.py
|   `-- project_utils.py
|-- results/
|   |-- deliverable_1_dataset_profile.csv
|   |-- deliverable_1_cleaning_summary.csv
|   |-- regression_metrics.csv
|   |-- regression_cv_metrics.csv
|   |-- regression_coefficients.csv
|   |-- classification_metrics.csv
|   |-- classification_tuning_results.csv
|   |-- clustering_metrics.csv
|   |-- cluster_profiles.csv
|   |-- association_rules.csv
|   `-- final_project_summary.csv
|-- images/
|   |-- deliverable_1/
|   |-- deliverable_2/
|   |-- deliverable_3/
|   `-- final/
|-- rubrics/
|   |-- README.md
|   |-- Residency_Day_2_Deliverable_2_Rubric.png
|   |-- Residency_Day_2_Deliverable_3_Rubric.png
|   `-- Residency_Day_3_Deliverable_4_Rubric.png
|-- report/
|   |-- MSCS_634_Final_Project_Report.docx
|   `-- report_source.md
`-- presentation/
    |-- MSCS_634_Final_Project_Presentation.pptx
    |-- presentation_script.md
    `-- video_link.txt
```

## Deliverable 1 Workflow

1. Load the approved dataset from a relative path and validate its exact schema.
2. Inspect the head, shape, columns, data types, `info()`, descriptive statistics, target values, and original class distribution.
3. Create a concise data dictionary for `Time`, `V1` through `V28`, `Amount`, and `Class`.
4. Audit missing values without creating artificial missing records.
5. Remove only exact duplicate rows and compare class counts before and after cleaning.
6. Review amount percentiles, selected PCA ranges, and the largest transactions without automatically deleting statistical extremes.
7. Quantify class counts, percentages, and the legitimate-to-fraud imbalance ratio.
8. Add `Hour`, `DayIndex`, and `LogAmount` without overwriting the original variables.
9. Save seven focused EDA visualizations and two CSV summaries.
10. Translate the EDA findings into leakage-safe requirements for later modeling.

## Cleaning Decisions

- Missing values are reported for every column. Imputation is unnecessary when the executed audit finds no missingness.
- Only exact duplicate rows are removed. Similar transactions are retained because they may represent separate legitimate purchases or meaningful fraud events.
- Statistical extremes are inspected but not automatically deleted. Large amounts may be legitimate, and fraud itself may appear anomalous; blind IQR removal could destroy target signal.
- Missing or invalid target values, unexpected classes, and negative `Time` or `Amount` values cause explicit validation errors rather than silent correction.
- PCA features remain separate because their original meanings are anonymized and distinct components should not be merged.

## Feature Engineering

- `Hour = (Time / 3600) % 24`
- `DayIndex = floor(Time / 86400)`
- `LogAmount = log1p(Amount)`

These are exploratory features. The original `Time` and `Amount` columns remain unchanged. Later model transformations and scaling will be learned within training pipelines after data splitting to prevent leakage.

## Deliverable 1 Status

- [x] Canonical notebook and repository structure prepared
- [x] Approved dataset loader and schema validation implemented
- [x] Missing-value and exact-duplicate audits implemented
- [x] Extreme-value and class-imbalance analysis implemented
- [x] Exploratory feature engineering implemented
- [x] Seven required visualizations implemented
- [x] Dataset-profile and cleaning-summary exports implemented
- [x] Deliverables 2-4 canonical notebook roadmaps created
- [x] Execute the notebook with `data/creditcard.csv`
- [x] Reconcile this README with observed cleaning and EDA results
- [x] Validate all generated Deliverable 1 artifacts
- [ ] Create the Day 1 checkpoint commit

## Observed Cleaning Summary

| Measure | Result |
|---|---:|
| Original rows | 284,807 |
| Missing cells | 0 |
| Exact duplicate rows | 1,081 |
| Rows after duplicate removal | 283,726 |
| Original legitimate rows | 284,315 |
| Original fraud rows | 492 |
| Cleaned legitimate rows | 283,253 |
| Cleaned fraud rows | 473 |
| Legitimate duplicates removed | 1,062 |
| Fraud duplicates removed | 19 |

No imputation was required. Only exact duplicates were removed; statistical extremes remained available for analysis.

## Key EDA Findings

- Fraud represented 0.17% of cleaned transactions, with approximately 598.84 legitimate transactions per fraud transaction.
- The Amount distribution was strongly right-skewed. Its 95th, 99th, and 99.9th percentiles were $365.34, $1,018.97, and $3,004.75, while the maximum was $25,691.16.
- The median `LogAmount` was 3.1355 for legitimate transactions and 2.3814 for fraud transactions. Amount behavior differed by class but did not separate the classes by itself.
- `V17`, `V14`, `V12`, `V10`, `V16`, and `V3` had the largest absolute PCA-feature correlations with `Class`.
- `V17` had the strongest negative correlation with `Class` at -0.3135, while `V11` had the strongest positive correlation at 0.1491. These values indicate association, not causation.
- Transaction volume peaked during hour 21 with 17,629 transactions. The highest observed hourly fraud rate was 1.45% during hour 2, based on 48 fraud records among 3,308 transactions.

## EDA Guidance for Future Modeling

- Classification splits and cross-validation must preserve the rare fraud proportion through stratification.
- Accuracy alone is misleading; precision, recall, F1, ROC-AUC, and PR-AUC are required.
- `Time`, `Amount`, and PCA features require thoughtful transformation or scaling inside leakage-safe pipelines.
- Fraud records must not be deleted merely because they appear as outliers.
- Class-weighted models provide an appropriate initial response to imbalance.
- Calculations and comparisons use full precision; rounding is presentation-only.
- PCA features may be statistically useful, but anonymization limits domain-level explanation and causal claims.

## Modeling Plan

- **Deliverable 2:** Predict `Amount` using Linear and Ridge regression with leakage-safe preprocessing and cross-validation. `Class`, `Amount`, and `LogAmount` will not be predictors.
- **Deliverable 3 classification:** Compare class-weighted Logistic Regression and a tuned class-weighted Decision Tree using an untouched stratified test set.
- **Deliverable 3 clustering:** Apply K-Means without `Class`, then reattach labels only for external cluster profiling.
- **Deliverable 3 pattern mining:** Discretize a focused training-only feature set and clearly distinguish enriched-sample support from population prevalence.
- **Deliverable 4:** Consolidate validated findings, limitations, ethical considerations, recommendations, a written report, and a presentation.

## Challenges

- Severe class imbalance makes accuracy an unreliable primary measure.
- Anonymized PCA features constrain direct financial interpretation.
- Dense visualizations require reproducible sampling for readability without changing analytical calculations.
- Statistical extremes may be valid or fraud-related, so outlier handling requires context rather than automatic deletion.
- The raw dataset must remain local because it is large and distributed through Kaggle.

## Ethical Considerations

Fraud prediction involves sensitive financial behavior. False negatives can create financial losses, while false positives can block legitimate customers. Anonymization protects original feature details but does not eliminate risks related to bias, privacy, model drift, or unequal error rates. Results should support human review and responsible risk controls rather than automatic punitive decisions.

## How to Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `Project_Deliverable_1.ipynb`, select the `.venv` kernel, and use **Kernel > Restart Kernel and Run All Cells**.

The notebook writes:

- `results/deliverable_1_dataset_profile.csv`
- `results/deliverable_1_cleaning_summary.csv`
- seven PNG files under `images/deliverable_1/`

## Rubric Coverage

- **Dataset selection and description:** project purpose, dataset suitability, limitations, and ethical context
- **Data inspection:** structure, data types, `info()`, statistics, target values, and data dictionary
- **Cleaning:** missingness audit, exact-duplicate removal, class-impact comparison, and extreme-value review
- **EDA:** imbalance, amount, time, selected PCA distributions, class correlation, and dense scatter sampling
- **Documentation:** section objectives, major-block comments, interpretations, next-step connections, relative paths, and reproducible artifacts

## Dataset Reference

Machine Learning Group - ULB. *Credit Card Fraud Detection*. Kaggle.<br>
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
