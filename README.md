## 1. Background

### 1.1 Purpose and Goals of the ADS
Home Credit is an international financial institution focused on expanding financial inclusion by providing consumer loans to the "unbanked population"—individuals with little-to-no credit history. Historically, Home Credit has leveraged statistical and machine learning models to predict a client’s loan repayment capability.

To enhance this initiative, Home Credit hosted a Kaggle competition, **Home Credit Default Risk**, tasking participants with developing models to predict loan defaults. This helps Home Credit:
- Minimize financial risk by identifying low-risk clients.
- Prevent unjust rejections for clients capable of repayment.

This analysis focuses on an Automated Decision System (ADS) built by Kaggle competitor and ML scientist **Nikita Kozodoi**.

### 1.2 Trade-offs and Multiple Goals
While the primary goal of the ADS is to predict loan defaults, trade-offs exist:
- **Higher Accuracy** → More conservative risk assessments, potentially excluding some who could benefit from loans.
- **Greater Inclusion** → Accepting higher-risk clients, potentially increasing default rates.

---

## 2. Input and Output

### 2.1 Data Description
Kozodoi's ADS aggregates data from multiple sources:

- `application_{train|test}.csv`: Main datasets. `train` includes the `TARGET` variable.
- `bureau.csv`: Clients' previous credits reported to Credit Bureau.
- `bureau_balance.csv`: Monthly balance data for each bureau record.
- `POS_CASH_balance.csv`: Balances from previous POS and cash loans.
- `credit_card_balance.csv`: Previous credit card balances.
- `previous_application.csv`: Prior Home Credit applications.
- `installments_payments.csv`: History of installment repayments.

These datasets were merged to form a unified dataset.

### 2.2 Input Features

- **Number of Features**: 1808 (after merging all datasets).
- **Selected Example Features**:
  - `SK_ID_CURR` (Loan ID)
  - `TARGET` (0: repaid, 1: defaulted)
  - `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `CNT_CHILDREN`, `AMT_CREDIT`, etc.
  - `FLAG_DOCUMENT_2` to `FLAG_DOCUMENT_21` indicating submitted documents.

- **Missing Values**:
  - Many features (e.g. housing dimensions) had >50% missing values.
  - `bureau.csv` had 70%+ missing in specific financial columns.
  - Other datasets showed varying levels of sparsity.

- **Pairwise Correlation**:
  - Gender, income, children, credit, and marital status were explored.
  - Most features showed **weak correlation** to `TARGET` (max ~0.18).

### 2.3 Output
The output is a **probability score** (via LightGBM ensemble) estimating a client’s default risk. Thresholds can then be applied to classify clients into high/low risk, influencing:
- Loan approval
- Interest rate
- Repayment schedule

---

## 3. Implementation and Validation

### 3.1 Data Cleaning and Preprocessing

- **Handling Missing Values**: Missing values imputed with `0`, given the binary nature of many features.
- **Feature Engineering**: Ratios, durations, and historical insights engineered.
- **Feature Reduction**: Retained top 500 features for modeling.
- **Encoding**: Label encoding for categorical variables.
- **Data Merging**: Unified dataset created using unique identifiers.

### 3.2 Model Implementation

- **Model**: LightGBM
- **Hyperparameters**:
  - `num_leaves`: 70
  - `max_depth`: 7
  - `min_split_gain`: 0.01
  - `subsample`: 0.9
  - `colsample_bytree`: 0.8
  - `reg_alpha` & `reg_lambda`: 0.1

### 3.3 Validation

- **Cross-Validation**: Stratified K-Fold
- **Evaluation Metric**: AUC-ROC
  - Training AUC: 0.92–0.95
  - Validation AUC: 0.78–0.80
- **Feature Importance**: Used to select top 500 features.
- **Performance Tracking**: Mean validation AUC ~0.792

---

## 4. Outcomes

### 4.1 Accuracy Analysis

- **Metrics Used**: Accuracy, Precision, Recall, FPR, FNR
- **Observations**:
  - Accuracy: High for both genders (80–90%)
  - Precision ↑ as threshold ↑
  - Recall: Often <20%, occasionally <10%
  - FNR: High, especially for women (indicating under-detection of defaulters)
  - Men had higher FPR; women had lower recall and higher FNR.

### 4.2 Fairness Analysis

- **Metrics**:
  - **Demographic Parity**
  - **Equal Opportunity**
  - **Predictive Equality**
  - **Equalized Odds**

- **Findings**:
  - At threshold 0.5: All fairness metrics show optimal (low) values.
  - At threshold 0.7: Increase in unfairness, especially in Equal Opportunity and Equalized Odds.
  - Best fairness performance observed at threshold 0.5.

### 4.3 Additional Performance Analysis

- Top-500-feature model performs similarly to all-feature model.
- All-feature model slightly better at lower thresholds, but top-500 model is more efficient.

---

## 5. Summary

### 5.1 Appropriateness of Data

- The dataset was provided via Kaggle (not chosen by the author).
- Class imbalances exist (e.g. gender: more females, more male defaulters).
- Despite imperfections, the dataset is generally appropriate for the task.

### 5.2 Robustness, Accuracy, and Fairness

- **Accuracy**: Strong across gender and thresholds.
- **Fairness**:
  - Good parity and equality metrics at low/mid thresholds.
  - Unfairness creeps in at higher thresholds (e.g. equal opportunity gaps).

- **Stakeholder Impact**:
  - Applicants benefit from fairness.
  - Regulators ensure legal compliance.
  - Home Credit benefits from both fairness and accuracy.

### 5.3 Comfort in Deployment

While robust and accurate, the model raises **fairness concerns** at higher thresholds. It is **not ready for public or industry deployment** without:
- Improvements to fairness metrics
- Implementation of transparency measures
- Ongoing monitoring and retraining

### 5.4 Recommendations for Improvement

- Evaluate a wider range of thresholds
- Reduce feature set further
- Explore subgroup-balanced training
- Improve code documentation and transparency

---

## References

- [Kaggle Competition Data](https://www.kaggle.com/competitions/home-credit-default-risk/data)  
- [Extensive EDA Notebook](https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda/notebook)  
- [Kozodoi’s GitHub Repository](https://github.com/kozodoi/Kaggle_Home_Credit)

---
