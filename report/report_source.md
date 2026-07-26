# Advanced Data Mining for Data-Driven Insights and Predictive Modeling

**Course:** 2026 Summer - Advanced Big Data and Data Mining  
**Course Number:** MSCS-634-M20  
**Residency:** July 24-26, 2026  

## Group Members
- Ashish Mahajan
- Murali Krishna Vattikunta
- Sreesh Sattiyamourthy
- Sreeharsha Varma Tinnanuri

## Introduction

This project applies a complete data-mining workflow to a highly imbalanced financial transaction dataset. The analysis progresses from cleaning and exploration to regression, classification, clustering, association-rule mining, and integrated recommendations.

## Business Problem

The primary practical problem is identifying fraudulent transactions while limiting unnecessary disruption to legitimate activity. A separate regression task evaluates how well anonymized characteristics estimate transaction amount.

## Dataset Description and Justification

The Kaggle Credit Card Fraud Detection dataset contained 284,807 original rows and 31 columns. After removing exact duplicates, 283,726 rows remained, including 473 fraud records (0.1667%). Its scale, numerical features, target label, and severe imbalance support the required mining methods.

## Data Preparation

The analysis removed 1,081 exact duplicate rows and found no missing values. Statistical extremes were retained because large valid transactions and fraud behavior may be anomalous. Original Time and Amount values were preserved.

## Exploratory Data Analysis

EDA showed a strongly right-skewed amount distribution, extreme class imbalance, time-varying transaction volume, and focused PCA relationships with the fraud label. These findings motivated stratified splitting, transformed amount features, and imbalance-aware evaluation.

## Feature Engineering

Engineered features included Hour, DayIndex, LogAmount, cyclical HourSin and HourCos terms, and PCA_L2_Norm. Regression excluded Amount, LogAmount, and Class from predictors. Classification used LogAmount and time-derived features. All preprocessing that learned parameters remained within training workflows.

## Regression Modeling

Linear Regression and Ridge Regression used StandardScaler pipelines and a log1p target transformation. Ridge alpha was selected from 0.01, 0.1, 1, 10, and 100 through five-fold training-only cross-validation.

## Classification Modeling

Class-weighted Logistic Regression and Decision Tree models addressed imbalance without changing the original records. Logistic scaling was fitted inside its pipeline, and the test set remained untouched during training.

## Hyperparameter Tuning

The Decision Tree used GridSearchCV with five-fold StratifiedKFold and average precision scoring. The selected settings were criterion=entropy, max_depth=8, and min_samples_leaf=20; mean training CV average precision was 0.7742.

## Results

### Regression Evaluation

Ridge Regression produced the lowest test RMSE at $165.4704, MAE at $57.9663, and R-squared of 0.5332. The most influential standardized Ridge coefficients included V2, PCA_L2_Norm, V1, V5, V6. The result is a useful baseline, but rare large amounts remain difficult to estimate.

### Classification Evaluation

Decision Tree is recommended because it achieved PR-AUC 0.7224, recall 0.7895, precision 0.0761, F1 0.1388, and ROC-AUC 0.9025. It produced 911 false positives and 20 false negatives, so human review and threshold analysis remain necessary.

### Clustering Analysis

K-Means was used as an unsupervised segmentation method to group transactions with similar numerical patterns. The 32 inputs were V1 through V28, LogAmount, HourSin, HourCos, and DayIndex. Class was excluded from model fitting, and StandardScaler placed all inputs on a comparable scale before K-Means calculated Euclidean distances.
The analysis evaluated k=2 through k=6 on a reproducible 50,000-transaction sample. Inertia measured within-cluster compactness, silhouette score measured cohesion and separation, and minimum cluster percentage identified highly imbalanced solutions. Although k=6 had the highest silhouette score (0.0872), its smallest cluster contained only 1.00% of the sample. k=4 was selected with silhouette 0.0762 and a 6.97% smallest cluster, providing a more balanced exploratory segmentation.
Class labels were reattached only after fitting to profile the clusters. Cluster 0 contained 3,487 transactions and 17 fraud records, producing the highest observed fraud rate (0.4875%) versus 0.1540% across the sample. Cluster 2 had the highest mean amount ($130.90) but an observed fraud rate of 0.0131%. PCA reduced the 32-dimensional fitted data to two dimensions only for visualization. These profiles describe transaction segments; they do not make K-Means a fraud detector.

### Association Rule Mining

FP-Growth used all training fraud rows and five legitimate rows per fraud row. Continuous values were discretized using training-derived boundaries for Amount, time of day, and focused PCA variables (V17, V14, V12, V10, V16, V3). 319 fraud-consequent rules met the final thresholds. The strongest displayed antecedent was V10_Low AND V12_Low AND V14_Low AND V16_Low AND V3_Low, with support 0.1287, confidence 1.0000, and lift 6.0000. Enriched-sample support is not population prevalence.

### Integrated Findings

Classification provides the direct fraud-screening evidence. Regression quantifies the separate difficulty of amount estimation. Clustering supplies exploratory segments, and association rules provide supporting descriptions of co-occurring conditions.

## Practical Recommendations

Use Decision Tree as a monitored screening layer. Select thresholds using false-positive and false-negative costs, route alerts to human reviewers, retain audit logs, and monitor precision, recall, PR-AUC, calibration, and drift. Use clustering and rules only as supporting analytical context.

## Ethical Considerations

Anonymization supports privacy but limits interpretability. False positives can inconvenience customers, while false negatives can create financial harm. Protected attributes are absent, so demographic fairness cannot be measured. Responsible use requires access controls, human review, threshold governance, monitoring, and documented appeals or correction processes.

## Limitations

The historical dataset covers approximately two days, contains anonymized PCA features, and is extremely imbalanced. Default classification thresholds may not reflect operational costs. Enriched association sampling changes support interpretation, and clustering does not establish fraud categories.

## Future Improvements

Future work should evaluate temporal validation, probability calibration, cost-sensitive threshold tuning, drift detection, periodic retraining, and newly approved data that improve interpretability without compromising privacy.

## Conclusion

The project demonstrates that reproducible preprocessing, leakage prevention, metric selection, and cautious interpretation matter as much as algorithm choice. The recommended classifier can support investigation, but final decisions require human oversight and continuing validation.

## References

Kaggle. Credit Card Fraud Detection. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
scikit-learn Developers. scikit-learn User Guide. https://scikit-learn.org/stable/user_guide.html
mlxtend Developers. Frequent Pattern Mining. https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/