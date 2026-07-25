# Residency Day 1: Project Deliverable 1: Data Collection, Cleaning, and Exploration

**Abstract / Project Overview**
This project focuses on the Credit Card Fraud Detection Dataset, which contains real-world credit card transaction data collected from European cardholders back in September 2013. The main objective here is to accurately identify fraudulent transactions by applying various data mining and machine learning techniques.
**Dataset Snapshot**
•	Total Records: 284,807 transactions
•	Total Attributes: 31 columns
•	Input Features: 30 numerical variables
•	Target Variable: Class

**Key Attributes**
•	Time: Seconds elapsed between each transaction and the first transaction in the dataset.
•	V1 - V28: Principal components obtained using PCA transformation (anonymized for privacy).
•	Amount: Total transaction amount.
•	Class: Target variable where 0 denotes a normal/legitimate transaction and 1 denotes a fraud transaction.

**Why this dataset?**
Fraud detection is a real-world problem that requires a solid grasp of data preprocessing, feature engineering, classification models, and anomaly detection. Since the dataset has a massive number of records and features, it is perfect for implementing advanced ML techniques.

**Data Cleaning and Preprocessing**
We performed the following data cleaning and preprocessing steps to prepare the dataset for modeling:
Step 1: Loading & Inspection
•	Imported the raw CSV file using pandas.
•	Checked the overall structure, dimensions (.shape), column names, and data types (.info()).
•	Ran statistical summaries (.describe()) to inspect feature distributions and scale.
Step 2: Missing Value Check
•	Checked all columns for NaN / null values using .isnull().sum().
•	Result: Zero missing values were found, so no imputation strategies (like mean/median replacement) were required.
Step 3: Removing Duplicates
•	Checked for duplicate rows using df.duplicated().
•	Dropped duplicate records to make sure repeated entries don't skew our model's performance or cause data leakage.
Step 4: Data Type Verification
•	Verified that all features are in numerical format (float64 / int64).
•	Since everything is already numeric, no categorical encoding (like One-Hot or Label Encoding) was needed.
Step 5: Class Imbalance Analysis
•	Calculated the frequency distribution of legitimate vs. fraudulent transactions using .value_counts().
•	Result: Found extreme class imbalance fraudulent cases make up less than 1% of the total dataset.
•	Plan for Modeling: To handle this imbalance, we will apply techniques like:
o	SMOTE (Synthetic Minority Over-sampling Technique)
o	Random Undersampling
o	Adjusting class_weight parameter in ML models

**Exploratory Data Analysis (EDA)**
EDA was carried out using matplotlib and seaborn libraries to visualize underlying trends.
•	Transaction Amount Analysis: Plotted distribution plots for the Amount feature. Observed that most transaction values are small, with a few extreme high-value outliers.
•	Fraud vs. Normal Comparison: Separated the dataset into fraud and non-fraud subsets to compare distribution patterns across features.
•	Correlation Analysis: Generated a heatmap correlation matrix to analyze feature relationships. Noticed that several V1-V28 PCA features show strong negative/positive correlations with the target Class.
•	Time-Based Analysis: Analyzed transaction frequency against Time and engineered an Hour feature to identify peak fraud hours.

**Key Insights Obtained****
1.	Scalability: The large volume of records makes it an ideal dataset for scalable data mining experiments.
2.	Severe Imbalance: Fraud cases are extremely rare, so plain accuracy metrics will be misleading; we must use Precision, Recall, F1-Score, and ROC-AUC.
3.	Amount Variances: Fraudulent transactions exhibit different spending patterns compared to regular transactions.
4.	PCA Utility: Even though V1-V28 features are anonymized, their statistical distributions show distinct variance between normal and fraudulent classes.
5.	Time Patterns: Time-based engineering gives useful contextual signals for detecting irregular transaction bursts.
6.	Clean Baseline: Post-cleaning, the dataset is robust and ready for feature engineering, clustering, and classification algorithms.
5. Challenges Encountered & Workarounds
   
**Challenge 1: Severe Class Imbalance**
•	Problem: Normal transactions heavily outnumber fraud cases, causing the model to overfit to the majority class.
•	Workaround: Highlighted this issue during EDA. Resampling methods (SMOTE / Undersampling) and weighted loss functions will be implemented in the next phase.
**Challenge 2: Anonymized Features (V1–V28)**
•	Problem: Features are PCA-transformed due to domain privacy, making domain-specific feature interpretation tricky.
•	Workaround: Retained all PCA components as they hold high variance and statistical significance crucial for fraud prediction.
**Challenge 3: High Dataset Volume**
•	Problem: Processing 280k+ rows can slow down certain computations and plot renderings.
•	Workaround: Used vectorised panda’s operations and optimized memory usage, avoiding redundant loops and heavy transformations during EDA.
**Challenge 4: Subtle Fraud Patterns**
•	Problem: Fraudulent transactions don't follow rigid rules and closely mimic normal user behavior.
•	Workaround: Combined correlation heatmaps, boxplots, and feature engineering to isolate subtle differences before passing data to ML algorithms.

**Conclusion**
Phase 1 of the project covering data acquisition, data cleaning, and Exploratory Data Analysis has been completed successfully. The dataset was thoroughly audited, duplicate records were dropped, data types were verified, and key distributional patterns were identified.
The findings from this phase will directly inform the next steps, which include Feature Scaling (StandardScaler), SMOTE Resampling, Model Building (Logistic Regression, Random Forest, XGBoost), and final evaluation using relevant metrics.
<img width="468" height="638" alt="image" src="https://github.com/user-attachments/assets/a67415ee-07ea-4de2-b434-690aade83da0" />
