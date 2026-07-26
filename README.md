# Advanced Data Mining for Data-Driven Insights and Predictive Modeling

**Course:** 2026 Summer - Advanced Big Data and Data Mining<br>
**Course Number:** MSCS-634-M20<br>
**Date:** July 24-26, 2026

## Group Members

1. Ashish Mahajan
2. Murali Krishna Vattikunta
3. Sreesh Sattiyamourthy
4. Sreeharsha Varma Tinnanuri

## Project Overview

This project applies a complete data-mining workflow to credit-card transactions. The four deliverables cover data cleaning and exploration, regression, classification, clustering, association-rule mining, and integrated recommendations. The work emphasizes reproducibility, leakage prevention, appropriate evaluation under extreme class imbalance, and responsible human review.

## Dataset

The project uses only the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- `Time`: seconds elapsed since the first transaction
- `V1` through `V28`: anonymized PCA-transformed features
- `Amount`: transaction amount
- `Class`: `0` for legitimate and `1` for fraud

The original data contained 284,807 rows and 31 columns. Exact-duplicate removal produced 283,726 rows: 283,253 legitimate and 473 fraud transactions. Fraud represents 0.1667% of the cleaned data.

The raw CSV is approximately 144 MiB and exceeds GitHub's regular 100 MiB file limit. It remains local and is excluded through `.gitignore`.

The notebooks first check `data/creditcard.csv`, then `creditcard.csv`. When neither exists, they use the public KaggleHub dataset:

```python
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)
```

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
|   `-- README.md
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
|-- report/
|   |-- report_source.md
|   `-- MSCS_634_Final_Project_Report.docx
`-- presentation/
    |-- MSCS_634_Final_Project_Presentation.pptx
    |-- presentation_script.md
    `-- video_link.txt
```

## Results

### Data Preparation and EDA

- No missing values were found, so imputation was unnecessary.
- 1,081 exact duplicates were removed, including 19 fraud duplicates.
- Statistical extremes were retained because valid large transactions and fraud can both appear anomalous.
- Amount was strongly right-skewed; the 99.9th percentile was $3,004.75 and the maximum was $25,691.16.
- `V17`, `V14`, `V12`, `V10`, `V16`, and `V3` had the largest absolute PCA correlations with `Class`.
- Seven focused figures document class imbalance, amount behavior, time patterns, and feature relationships.

### Regression

The regression task predicts `Amount` without using `Amount`, `LogAmount`, or `Class` as predictors. Features include `V1`-`V28`, `Hour`, `DayIndex`, `HourSin`, `HourCos`, and `PCA_L2_Norm`. Linear and Ridge Regression use standardized pipelines and a `log1p` target transformation. A training-derived 99.9th-percentile bound prevents unsupported inverse-transform extrapolation without altering actual targets.

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Ridge Regression | $57.9663 | $165.4704 | 0.5332 |
| Linear Regression | $57.9788 | $165.7208 | 0.5318 |

Ridge selected `alpha=100`. Its five-fold training CV RMSE was $178.4252 with a standard deviation of $18.1263, and mean CV R-squared was 0.4999. Ridge is the recommended regression baseline because it produced the lower untouched-test RMSE and higher R-squared, although the improvement was small.

### Classification

Classification uses a stratified 80/20 split. Logistic Regression and Decision Tree models use `class_weight="balanced"`. The tree was tuned only on training data using five-fold `StratifiedKFold`, `GridSearchCV`, and average precision. Its selected parameters were `criterion="entropy"`, `max_depth=8`, and `min_samples_leaf=20`; mean CV average precision was 0.7742.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9836 | 0.0761 | 0.7895 | 0.1388 | 0.9025 | 0.7224 | 911 | 20 |
| Logistic Regression | 0.9752 | 0.0560 | 0.8737 | 0.1053 | 0.9671 | 0.6804 | 1,398 | 12 |

The Decision Tree is recommended for this run because it achieved higher PR-AUC, precision, and F1 with fewer false positives. Logistic Regression recovered more fraud cases, demonstrating that the final operating threshold must reflect the costs of missed fraud and unnecessary customer interruption.

### Clustering and Pattern Mining

K-Means evaluated `k=2` through `k=6` on a reproducible 50,000-row standardized sample without using `Class`. `k=4` was selected with silhouette score 0.0762 after excluding solutions whose smallest cluster contained under 5% of the sample. After labels were reattached for external analysis, Cluster 0 had the highest observed fraud concentration at 0.4875%. This is exploratory segmentation, not fraud detection.

FP-Growth used all 378 fraud rows from the classification training partition and 1,890 reproducibly sampled legitimate rows. Training-derived bins represented Amount, time of day, and six focused PCA variables. A total of 319 fraud-consequent rules met minimum support 0.03, confidence 0.60, and lift 1.20. The strongest displayed rule had support 0.1287, confidence 1.0000, and lift 6.0000. These values describe the enriched sample and do not estimate population prevalence or establish causation.

## Limitations and Challenges

- The dataset covers approximately two days of historical transactions.
- PCA feature meanings are anonymized.
- Extreme class imbalance makes precision difficult at the default threshold.
- Regression can underpredict rare high-value transactions.
- Association-rule support comes from an enriched training sample.
- Fraud behavior may change through concept drift.

## How to Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order with **Kernel > Restart Kernel and Run All Cells**:

1. `Project_Deliverable_1.ipynb`
2. `Project_Deliverable_2.ipynb`
3. `Project_Deliverable_3.ipynb`
4. `Project_Deliverable_4.ipynb`

## Submission Artifacts

- [Final report](report/MSCS_634_Final_Project_Report.docx)
- [Presentation](presentation/MSCS_634_Final_Project_Presentation.pptx)
- [Video link](https://go.screenpal.com/watch/cOiO2anUS93)

## References

- Machine Learning Group - ULB. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Kaggle.
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [mlxtend Frequent Pattern Mining](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/)
