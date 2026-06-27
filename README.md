# Smart-Demand: Sales Volume Prediction System

**Live App:** https://smartdemand-production-df02.up.railway.app

Random Forest machine learning model trained on the Brazilian E-Commerce (Olist) dataset. Predicts total sales volume per product category per month based on pricing, quality, and market factors.

COMP6577001 - Machine Learning | Final Project

## Struktur Folder

```
SmartDemand_App/
├── main.py
├── index.html
├── requirements.txt
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

## Deployment

App ini di-deploy ke **Railway**.

1. Push semua file ke GitHub repository (bisa private)
2. Buka https://railway.app → New Project → Deploy from GitHub repo
3. Connect repository GitHub
4. Settings otomatis terdeteksi dari `requirements.txt`, atau set manual:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Tunggu deploy selesai (3–5 menit)
6. Klik Settings → Networking → Generate Domain untuk dapat link publik

## Catatan File Model

Download 4 file dari Google Drive folder `SmartDemand_Dataset/models/`:

- random_forest_model.joblib (~50MB)
- linear_regression_model.joblib
- label_encoder.joblib
- config.json

Taruh di folder `models/` sebelum deploy. Untuk GitHub, file >50MB perlu Git LFS:
```
git lfs install
git lfs track "*.joblib"
git add .gitattributes
```

## Model Performance

| Model | Test R² | Test MAE | Test RMSE |
|---|---|---|---|
| Random Forest | 0.9541 | 8.92 | 17.22 |
| Linear Regression | 0.9445 | 9.45 | 18.26 |

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: HTML / CSS / Vanilla JS
- Model: scikit-learn (Random Forest Regressor)
- Hosting: Railway
