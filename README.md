# Kaggle Playground S6E11: Credit Card Fraud Detection

This is my **first Kaggle competition**. I'm using it to learn how a competition works from start to finish: getting the data, setting up cross-validation, training models and submitting predictions. I want to be ready for harder competitions later.

- **Competition:** [Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11)
- **Task:** predict whether a transaction is fraud (`is_fraud`)
- **Metric:** ROC AUC
- **Data:** about 700k training rows and 300k test rows, with 24 features. About 1.7% of transactions are fraud.

## Results so far

All models use 5-fold stratified cross-validation, the same folds (`random_state=42`) and early stopping on AUC. No model has been tuned yet.

| Fold | LightGBM | XGBoost | CatBoost | **Blend (XGB + CatBoost)** |
|---|---|---|---|---|
| 1 | 0.94287 | 0.94475 | 0.94433 | **0.94514** |
| 2 | 0.94722 | 0.94794 | 0.94737 | **0.94828** |
| 3 | 0.94953 | 0.95053 | 0.95062 | **0.95120** |
| 4 | 0.94562 | 0.94611 | 0.94627 | **0.94688** |
| 5 | 0.93973 | 0.94116 | 0.94176 | **0.94209** |
| **Out-of-fold AUC** | 0.94487 | 0.94606 | 0.94601 | **0.94668** |

- **XGBoost did better than LightGBM in every fold** (0.94606 against 0.94487).
- **CatBoost alone is about as good as XGBoost** (0.94601), but its predictions are different enough (correlation 0.977) that a **50/50 rank blend of the two beats XGBoost in every fold** (+0.00062). The submission now uses this blend. CatBoost is slow on CPU: about 23 minutes for 5 folds, against about 1 minute for XGBoost.

### Feature ideas that did not help

Tested with XGBoost on the same folds. None of them gave a clear gain, so none are in the notebook.

| Idea | OOF AUC | vs XGBoost baseline |
|---|---|---|
| Count of the 6 risk flags | 0.94606 | ±0 |
| Ratios (amount/balance, amount × merchant risk, txns per hour) | 0.94592 | −0.00014 |
| Category pairs as new categorical columns | 0.94240 | −0.00366 |
| Frequency encoding of every column | 0.94582 | −0.00024 |
| Target encoding of single categories (in-fold) | 0.94597 | −0.00009 |
| Target encoding of category pairs (in-fold) | 0.94580 | −0.00026 |
| Frequency + pair target encoding | 0.94621 | +0.00015 (mixed across folds) |

## Project layout

| File | What it is |
|---|---|
| `playground_s6e11.ipynb` | Loads data, quick EDA, XGBoost and CatBoost 5-fold CV, blends them and writes the submission |
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

- [x] Feature engineering (interactions, ratios, encodings): no gain, see above
- [x] Try CatBoost
- [x] Blend models using out-of-fold predictions
- [ ] Hyperparameter tuning (for example with Optuna)
- [ ] Add LightGBM to the blend
