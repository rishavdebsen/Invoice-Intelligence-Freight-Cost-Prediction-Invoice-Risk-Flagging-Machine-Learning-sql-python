# Vendor Invoice Intelligence

AI-driven **freight cost prediction** and **invoice risk flagging** built on vendor invoice data. The project combines SQL feature engineering, machine learning models, reusable inference scripts and a Streamlit portal.

## Problem

Freight is a non-trivial part of landed cost. Poor freight estimates distort margins and inventory planning, and abnormal vendor invoices create financial leakage and manual review workload. This project:

- Forecasts freight cost for a vendor invoice from quantity and dollar value
- Flags invoices that need manual approval based on cost, freight and delivery patterns

## Dataset

SQLite database `inventory.db` with five tables: `purchases`, `purchase_prices`, `vendor_invoice`, `begin_inventory`, `end_inventory`.

- Freight model: `vendor_invoice` (5,543 invoices)
- Invoice flagging model: `vendor_invoice` joined with aggregated purchase data
- The database (about 72 MB) is not stored in this repo. Place it at `data/inventory.db` before running.

## Approach

### 1. Freight cost prediction (regression)
1. Explored the relationship between Quantity, Dollars and Freight (correlation, heatmap, scatter plots)
2. Trained and compared Linear Regression, Decision Tree and Random Forest
3. Selected the best model by lowest MAE and saved it with joblib

### 2. Invoice risk flagging (classification)
1. Joined `vendor_invoice` with aggregated purchase data in SQL
2. Engineered features: invoice quantity, invoice dollars, freight, total item quantity, total item dollars
3. Labelled risky invoices using business rules: invoice vs item dollar mismatch and long receiving delay
4. Scaled features with StandardScaler and trained a Random Forest Classifier
5. Tuned hyperparameters with GridSearchCV (5-fold cross-validation)
6. Evaluated with accuracy, precision, recall, F1 and confusion matrix; reviewed feature importance
7. Saved the best model and the scaler with joblib

## Freight model results

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 24.11 | 124.72 | 96.99% |
| Decision Tree | 32.97 | 150.31 | 95.63% |
| Random Forest | 26.13 | 134.79 | 96.48% |

**Best model:** Linear Regression, because freight is close to linear in invoice dollars. It is also the simplest and easiest to explain to stakeholders.

## Project structure

```
machine learning project/
├── data/
│   └── inventory.db              # not tracked
├── notebooks/                    # EDA and experiments
├── freight_cost_prediction/
│   ├── data_preprocessing.py     # load data, prepare features, split
│   ├── model_evaluation.py       # train and evaluate regressors
│   └── train.py                  # freight training pipeline
├── invoice_flagging/
│   ├── data_preprocessing.py     # SQL feature table, labels, scaling
│   ├── modeling_evaluation.py    # Random Forest + GridSearchCV, evaluation
│   └── train.py                  # invoice flagging training pipeline
├── inference/
│   ├── predict_freight.py        # predict freight for new invoices
│   └── predict_invoice_flag.py   # flag new invoices
├── models/                       # saved models and scaler (.pkl)
├── app.py                        # Streamlit portal
└── README.md
```

## Getting started

### 1. Install dependencies
```
pip install pandas numpy scikit-learn joblib matplotlib seaborn streamlit
```

### 2. Add the data
Place `inventory.db` in the `data/` folder.

### 3. Train the models
Run from the project root so relative paths resolve:
```
python freight_cost_prediction/train.py
python invoice_flagging/train.py
```
Models are saved to `models/`.

### 4. Run predictions from Python
```
python inference/predict_freight.py
python inference/predict_invoice_flag.py
```

### 5. Launch the portal
```
streamlit run app.py
```

## Streamlit portal

- **Freight Cost Prediction:** enter quantity and invoice dollars to get an estimated freight cost
- **Invoice Manual Approval Flag:** enter invoice details to see whether the invoice is safe for auto-approval or needs manual review

## Tech stack

Python, SQL (SQLite), pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit, Jupyter, Git

## Skills demonstrated

SQL joins and aggregations, feature engineering, regression and classification, model comparison, hyperparameter tuning (GridSearchCV), cross-validation, feature scaling, model evaluation, model persistence, modular code structure, and building a simple ML app.

## Author

**Rishav Deb Sen**
GitHub: [github.com/rishavdebsen](https://github.com/rishavdebsen)
LinkedIn: [linkedin.com/in/rishav-1824a392a](https://linkedin.com/in/rishav-1824a392a)