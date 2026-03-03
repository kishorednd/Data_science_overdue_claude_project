# End-to-End Data Science Project: Overdue Payment Prediction

## 1. Problem Understanding
- **Business objective:** predict if a customer will become overdue so collections and risk teams can prioritize intervention.
- **Detected problem type:** `binary_classification`.
- **Target variable selected:** `Overdue_Status` with classes [0.0, 1.0].
- **Target prevalence:** overdue rate is **23.17%** across 10,000 records.

## 2. Data Exploration
- Dataset size: **10000 rows x 20 prepared columns**.
- Missing values identified before imputation:
  - Age: 0 (0.00%)
  - Annual_Income: 150 (1.50%)
  - Credit_Score: 100 (1.00%)
  - Loan_Amount: 80 (0.80%)
  - Loan_Term_Months: 0 (0.00%)
  - Interest_Rate: 0 (0.00%)
  - Monthly_Payment: 280 (2.80%)
  - Account_Age_Months: 0 (0.00%)
  - Number_of_Existing_Loans: 0 (0.00%)
  - Number_of_Late_Payments: 0 (0.00%)
  - Months_Since_Last_Payment: 120 (1.20%)
  - Debt_to_Income_Ratio: 90 (0.90%)
  - Gender: 226 (2.26%)
  - City: 1280 (12.80%)
  - Employment_Status: 206 (2.06%)
  - Overdue_Status: 50 (0.50%)
  - app_year: 0 (0.00%)
- Visual artifacts generated:
  - `artifacts/city_distribution.svg`
  - `artifacts/overdue_by_employment.svg`
- Distribution snapshot (Debt_to_Income_Ratio histogram):
  - `    0.22 -     6.23 | ########## (488)`
  - `    6.23 -    12.24 | ####################### (1141)`
  - `   12.24 -    18.25 | ############################# (1404)`
  - `   18.25 -    24.27 | ############################## (1451)`
  - `   24.27 -    30.28 | ############################# (1450)`
  - `   30.28 -    36.29 | ######################## (1179)`
  - `   36.29 -    42.30 | ################### (934)`
  - `   42.30 -    48.31 | ############## (722)`
  - `   48.31 -    54.32 | ########## (510)`
  - `   54.32 -    60.33 | ###### (327)`
  - `   60.33 -    66.35 | #### (207)`
  - `   66.35 -    72.36 | ### (187)`

## 3. Data Cleaning & Preparation
- Standardized categorical inconsistencies (e.g., `MALE`, `M`, `male` unified).
- Parsed mixed date formats into numeric year/month/weekday features.
- Imputed numeric nulls with medians and categorical nulls with modes.
- Capped numeric outliers using IQR winsorization.
- One-hot encoded categorical features and standardized numeric fields.

## 4. Feature Engineering
- Added `Loan_to_Income`, `Payment_to_Income`, `Late_per_Loan`, `Credit_Income_Interaction`.
- These features represent affordability stress and repayment behavior signals.

## 5. Model Training
- Split data into train/test (80/20).
- Trained models: Baseline majority, Logistic Regression, Gaussian Naive Bayes, KNN.
- Logistic Regression hyperparameter search over learning rate/epochs selected `lr=0.08`, `epochs=220` based on validation F1.

## 6. Model Evaluation
- Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC to balance false positives and false negatives.
- Best model: **GaussianNB** (sorted by F1).

## 7. Model Comparison
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| GaussianNB | 0.6280 | 0.3751 | 0.7926 | 0.5092 | 0.7227 |
| LogisticRegression | 0.7690 | 0.6344 | 0.1211 | 0.2034 | 0.7022 |
| KNN | 0.7555 | 0.4815 | 0.0534 | 0.0961 | 0.5654 |
| BaselineMajority | 0.7565 | 0.0000 | 0.0000 | 0.0000 | 0.4916 |

## 8. Predictions
- Generated test-set predictions and risk bands in `artifacts/test_predictions.csv`.
- Risk bands: High (>=0.70), Medium (0.40-0.69), Low (<0.40).

## 9. Insights & Business Conclusion
- Main risk drivers include late payment behavior, affordability-related ratios, and loan burden variables.
- Use predicted risk tiers to prioritize proactive reminders, restructuring offers, and collections workflows.
- Track ROI by comparing recovered amounts and prevented delinquencies across intervention strategies.

### Feature Importance (Top 10)
- Credit_Score: 0.2272
- Credit_Income_Interaction: 0.2052
- Employment_Status__unemployed: 0.1333
- Number_of_Late_Payments: 0.1197
- Debt_to_Income_Ratio: 0.1192
- Months_Since_Last_Payment: 0.0916
- Late_per_Loan: 0.0738
- City__new york: 0.0224
- City__los angeles: 0.0224
- Age: 0.0142

### Risks, Limitations, and Improvements
- Synthetic/limited feature space may not capture all delinquency drivers (behavioral, macroeconomic, channel interactions).
- Monitor for data drift and threshold drift each month; recalibrate cutoffs by business capacity.
- Next iteration: gradient boosting, probability calibration, reject inference, and fairness diagnostics.

## 10. Deployment & Monitoring Strategy
- Batch score new applications daily and push risk band to CRM/collections queue.
- Operational metrics: model AUC/F1, capture rate in top risk decile, intervention conversion, and false-positive cost.
- Governance: champion-challenger retraining every quarter or when drift thresholds are breached.

## Appendix: Preprocessing Parameters
- Numeric medians used for imputation:
  - Age: 39.0000
  - Annual_Income: 36236.5098
  - Credit_Score: 651.6265
  - Loan_Amount: 8115.2252
  - Loan_Term_Months: 36.0000
  - Interest_Rate: 12.0570
  - Monthly_Payment: 228.1850
  - Account_Age_Months: 16.0000
  - Number_of_Existing_Loans: 1.0000
  - Number_of_Late_Payments: 2.0000
  - Months_Since_Last_Payment: 1.3696
  - Debt_to_Income_Ratio: 26.2969
  - app_year: 2021.0000
  - app_month: 7.0000
  - app_weekday: 3.0000
- Categorical modes:
  - Gender: male
  - City: new york
  - Employment_Status: employed

- Outlier caps (IQR lower/upper):
  - Age: (-2.5000, 81.5000)
  - Annual_Income: (-39455.5787, 122089.1119)
  - Credit_Score: (383.8582, 917.7625)
  - Loan_Amount: (-13318.6966, 33264.1162)
  - Loan_Term_Months: (-12.0000, 84.0000)
  - Interest_Rate: (-1.3791, 25.3760)
  - Monthly_Payment: (-445.0179, 1040.6375)
  - Account_Age_Months: (-34.5000, 73.5000)
  - Number_of_Existing_Loans: (-0.5000, 3.5000)
  - Number_of_Late_Payments: (-2.0000, 6.0000)
  - Months_Since_Last_Payment: (-2.6167, 5.8680)
  - Debt_to_Income_Ratio: (-17.6619, 72.3578)
  - app_year: (2017.0000, 2025.0000)
  - app_month: (-7.5000, 20.5000)
  - app_weekday: (-5.0000, 11.0000)