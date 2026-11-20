# Customer churn prediction and fairness audit

Predicting which customers are likely to leave, then checking whether the models are
reliable and fair.

## What's inside

| Notebook | What it does |
|---|---|
| `case studies.ipynb` | Cleans two churn datasets and trains a model for each: **k-nearest neighbours** for a telecom company and a **random forest** for a bank. Reports accuracy, precision, recall and F1, plus the features that drive churn. |
| `Task2_Part2_Deliberation_Analysis.ipynb` | Tests those models properly: **stratified 5-fold cross-validation** (with scaling inside the pipeline to avoid leakage), **learning curves**, and a **Fairlearn bias audit** comparing recall across gender, senior-citizen status, country and age band. |

## Datasets

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: IBM Telco customer churn sample data
- `Bank Customer Churn Prediction.csv`: bank customer churn data

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn fairlearn jupyter
jupyter notebook
```

Open either notebook and run all cells. The CSV files sit next to the notebooks.

## Skills shown

Data cleaning and encoding · classification with kNN and random forests · model
evaluation with cross-validation and learning curves · feature importance · fairness
auditing with Fairlearn
