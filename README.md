# Smart-Demand: Sales Volume Prediction System

Random Forest model trained on the Brazilian E-Commerce (Olist) dataset. Predicts total sales volume per product category per month from pricing, quality and market factors.

COMP6577001 - Machine Learning | Final Project

Note: the Railway deployment is offline, so run it locally with the steps below.

## Results

The first version of this model reported R2 0.9541 and MAE 8.92, and that number turned out to be misleading for two reasons.

The first is leakage. `num_products` counts how many products the category sold in the month being predicted, so in practice you only know it after that month is over. It carried 49 percent of the feature importance. The seasonality index had the same problem, since it was computed over every month including the test ones.

The second is the split. The data is monthly and ordered in time, but the split was random, so the model was training on months that came after the months it was tested on.

`03_Leak_Free_Evaluation.ipynb` redoes the evaluation with a time-ordered split (the last 4 months as test, 225 rows) and only features that are known before the month starts.

| Model, time-ordered test | MAE | R2 |
|---|---|---|
| Naive: next month = last month | 11.63 | 0.9361 |
| Random Forest, original features (leaky) | 9.47 | 0.9541 |
| Random Forest, leak-free features | 16.41 | 0.8777 |
| Random Forest, leak-free, predicts the change | 17.98 | 0.8555 |

Once the leaky features are gone, predicting last month's number again beats the model. With roughly two years of monthly history per category there is not much for the model to learn beyond the previous month, and neither version beats that yet. The median category sells 23 units a month in the test period, so even the baseline MAE of 11.63 is a large relative error.

The app in this repository still serves the original model, so treat its predictions as a demo of the pipeline rather than a forecast.

For reference, here are the original numbers on the random split:

| Model | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|
| Random Forest | 0.9541 | 8.92 | 17.22 |
| Linear Regression | 0.9445 | 9.45 | 18.26 |
| Naive: next month = last month | 0.8898 | 14.54 | 26.68 |

## Notebooks

**01_Initial_Model_ProductLevel.ipynb**

The first attempt predicted quantity_sold per individual product per month. It only reached R2 0.5033, because most products sell 1 to 5 units a month and the data is too noisy at that level.

**02_Final_Model_CategoryLevel.ipynb**

The target moved to units per product category per month, which is much smoother. This is the model the app serves, evaluated on a random split.

**03_Leak_Free_Evaluation.ipynb**

Repeats the data preparation, then evaluates on a time-ordered split with leak-free features against the naive baseline. The notebooks read the Olist CSV files from Google Drive, so point `DATA_PATH` at your own copy.

## Struktur Folder

```
SmartDemand_App/
├── main.py
├── index.html
├── requirements.txt
├── 01_Initial_Model_ProductLevel.ipynb
├── 02_Final_Model_CategoryLevel.ipynb
├── 03_Leak_Free_Evaluation.ipynb
└── models/
    ├── random_forest_model.joblib
    ├── linear_regression_model.joblib
    ├── label_encoder.joblib
    └── config.json
```

## Run Lokal

```
pip install -r requirements.txt
python main.py
```

Buka browser di http://localhost:8000

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: HTML / CSS / Vanilla JS
- Model: scikit-learn (Random Forest Regressor)
