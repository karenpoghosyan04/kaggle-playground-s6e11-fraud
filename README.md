# Kaggle Playground S6E11: Credit Card Fraud Detection

This is my **first Kaggle competition**. I'm using it to learn how a competition works from start to finish: getting the data, setting up cross-validation, training models and submitting predictions. I want to be ready for harder competitions later.

- **Competition:** [Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11)
- **Task:** predict whether a transaction is fraud (`is_fraud`)
- **Metric:** ROC AUC
- **Data:** about 700k training rows and 300k test rows, with 24 features. About 1.7% of transactions are fraud.

## Results so far

Both models use 5-fold stratified cross-validation, the same folds (`random_state=42`) and early stopping on AUC. Neither model has been tuned or given extra features yet.

| Fold | LightGBM | XGBoost |
|---|---|---|
| 1 | 0.94287 | 0.94475 |
| 2 | 0.94722 | 0.94794 |
| 3 | 0.94953 | 0.95053 |
| 4 | 0.94562 | 0.94611 |
| 5 | 0.93973 | 0.94116 |
| **Out-of-fold AUC** | **0.94487** | **0.94606** |

**XGBoost did better than LightGBM in every fold**, with an out-of-fold AUC of 0.94606 against 0.94487. The gap is small, so how I build features will probably matter more than which of the two models I pick. The notebook now uses XGBoost as the baseline.

## Project layout

| File | What it is |
|---|---|
| `playground_s6e11.ipynb` | Loads data, quick EDA, XGBoost 5-fold CV and writes the submission |
| `requirements.txt` | Python packages used |

The data and `submission.csv` are not in the repo. The notebook downloads the data with `kagglehub`.

## How to run

1. Accept the competition rules on Kaggle and set up your Kaggle API credentials.
2. Install the packages:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run `playground_s6e11.ipynb` from top to bottom. It writes `submission.csv`.

## Next steps

- [ ] Feature engineering (interactions, ratios, aggregations)
- [ ] Hyperparameter tuning (for example with Optuna)
- [ ] Try CatBoost
- [ ] Blend models using out-of-fold predictions
