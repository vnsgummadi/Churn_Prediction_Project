Telecom Customer Churn Prediction
================================

Project Goal
------------
Predict which customers are likely to churn so that a business can proactively
target retention offers. This project implements a clean, end-to-end machine
learning pipeline with a recall-first decision policy, prioritizing the
identification of churners over raw accuracy.

---------------------------------------------------------------------

Repository Contents
-------------------
- churn_full_pipeline.py
  Final production-style Python script implementing the complete pipeline:
  preprocessing, training, evaluation, and artifact saving.

- Saved artifacts produced by the pipeline:
  - churn_pipeline_final.joblib  : trained preprocessing + model pipeline
  - feature_coefficients.csv     : feature-to-coefficient mapping
  - metrics_summary.csv          : final evaluation metrics
  - test_predictions.csv         : test set probabilities and predictions

- requirements.txt
  Python dependencies required to run the project.

- churn_full_pipeline.ipynb (optional)
  Exploratory notebook used for experimentation and analysis.

---------------------------------------------------------------------

Key Results (Test Set)
---------------------
The decision threshold was selected based on business requirements,
prioritizing recall over precision.

- Probability threshold: 0.4
- Recall: 0.866
- Precision: 0.470
- F1-score: 0.609
- ROC AUC: 0.847

This operating point reflects a deliberate tradeoff:
accepting more false positives to minimize missed churners.

---------------------------------------------------------------------

Dataset
-------
- Source: IBM Telco Customer Churn dataset (public sample)
- Size: ~7,000 customers
- Feature types:
  - Categorical: contract type, internet service, payment method, etc.
  - Numerical: tenure, monthly charges, total charges

---------------------------------------------------------------------

Approach (End-to-End)
---------------------
1. Data Cleaning
   - Converted TotalCharges from string to numeric
   - Handled missing values
   - Removed customer identifier column

2. Feature Engineering
   - Tenure buckets
   - Average monthly charge proxy
   - Monthly charge to average charge ratio

3. Preprocessing
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features

4. Modeling
   - Logistic Regression
   - class_weight = 'balanced' to address class imbalance

5. Decision Policy
   - Probability-based predictions
   - Threshold tuning to prioritize recall

6. Evaluation
   - Precision, Recall, F1-score
   - Confusion matrix
   - ROC AUC
   - Precision–Recall tradeoff analysis

7. Interpretability
   - Model coefficients extracted from the trained pipeline
   - Identification of key churn drivers and retention signals

---------------------------------------------------------------------

Why Logistic Regression?
------------------------
- Interpretable and explainable
- Strong baseline for tabular data
- Enables clear business insights via coefficients
- Easier to justify decisions compared to black-box models

---------------------------------------------------------------------

Example Churn Drivers Identified
--------------------------------
Features increasing churn probability:
- Fiber optic internet service
- Month-to-month contracts
- Higher total charges
- Streaming services enabled
- Electronic check payment method

Features decreasing churn probability:
- Longer customer tenure
- Two-year contracts
- Automatic bank transfer payments

These drivers align well with real-world telecom churn behavior.

---------------------------------------------------------------------

How to Run
----------
1. (Recommended) Create and activate a virtual environment.

2. Install dependencies:
   pip install -r requirements.txt

   Example requirements:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - joblib

3. Run the pipeline:
   python churn_full_pipeline.py

All outputs will be generated automatically in the models/ directory.

---------------------------------------------------------------------

Important Note on Large Files
-----------------------------
GitHub does not allow files larger than 100 MB.

If trained model files are large:
- Do not commit them directly
- Add models/ to .gitignore
- Upload model files to external storage (Drive, Dropbox, GitHub Releases)
- Add a download link in this README if needed

Alternatively, Git LFS can be used to track large binary files.

---------------------------------------------------------------------

Reproducibility Checklist
-------------------------
- Python 3.8 or higher
- Dependencies installed via requirements.txt
- Script runs end-to-end without manual intervention
- Outputs saved consistently in models/

---------------------------------------------------------------------

Author
------
Venkata Nikhil Sai Gummadi,
MS Data Science, 
Arizona State University,
Mail: vnsgummadi@gmail.com,
+1 623-999-7076.


