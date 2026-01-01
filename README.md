This README is written for recruiters, reviewers, and technical stakeholders

________________________________________________________________________________________________________________________

### Bank Customer Churn Prediction - End-to-End Machine Learning Project

**Project Overview**

Customer churn is a critical business problem that directly affects revenue, customer lifetime value, and growth planning. 
This project develops an end-to-end machine learning pipeline to **predict bank customer churn**, identify **key churn drivers**, and support **data-driven retention strategies.**

The solution follows **production-grade design principles**, including modular code structure, pipeline-based preprocessing, and reproducible model evaluation.
________________________________________________________________________________________________________________________

#### **Business Objective**

* Predict whether a customer is likely to churn.
* Identify key behavioral and financial churn drivers.
* Compare multiple machine learning models.
* Provide interpretable insights for decision-makers.

________________________________________________________________________________________________________________________
#### **Project Structure**

customerchurnprediction/
│
├── data/
│   ├── raw/                # Original dataset
│   ├── engineered/         # Feature-engineered data
│   ├── model_input/        # Train/test inputs
│   └── models/             # Saved model artifacts
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── modelling.py
│   ├── evaluation.py
│   └── utils.py
│
├── README.md
└── requirements.txt


________________________________________________________________________________________________________________________

**Methodology**
1. Data Exploration
* Distribution analysis
* Class imbalance assessment
* Initial churn patterns

2. Feature Engineering
* Customer tenure behavior
* Product usage intensity
* Balance-based risk indicators
* Composite churn risk score

3. Preprocessing
* Numerical scaling
* Categorical encoding
* Missing value imputation
* Leakage-safe pipelines

4. Modelling
* Logistic Regression(baseline, interpretable)
* Random Forest
* Gradient Boosting

5. Evaluation
* ROC-AUC(primary metric)
* confusion matrix
* Classification report
* Model comparison table

________________________________________________________________________________________________________________________

**Key Results**

* Gradient Boosting achieved the best ROC-AUC performance.
* Behavioral indicators(activity status, tenure, product usage) were stronger churn predictors than demographics.
* Feature engineering significantly improved model performance.

________________________________________________________________________________________________________________________

**Business Insights**
* Inactive customers with low tenure are at highest churn risk.
* Customers holding a single product are more likely to churn.
* Balance behavior provides early churn signals.

________________________________________________________________________________________________________________________

**Technologies Used**
* Python
* Pandas, Numpy
* Scikit-learn
* Jupyter Notebook
* Joblib

________________________________________________________________________________________________________________________

**Next Steps**
* Cost-sensitive churn modelling
* Retention campaign simulation
* Model deployment via API or dashboard

________________________________________________________________________________________________________________________