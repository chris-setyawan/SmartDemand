# Smart-Demand — Deployment Guide (FastAPI)

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

```bash
pip install -r requirements.txt
python main.py
```

Buka browser di http://localhost:8000

## Deploy ke Render.com (Gratis)

1. Upload semua file ke GitHub repository (bisa private)
2. Buka https://render.com → New → Web Service
3. Connect repository GitHub
4. Isi settings:
   - Build Command : `pip install -r requirements.txt`
   - Start Command : `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Klik Deploy → tunggu 2-3 menit
6. Dapat link publik untuk user testing

## Catatan File Model

Download 4 file dari Google Drive folder SmartDemand_Dataset/models/:
- random_forest_model.joblib  (~50MB)
- linear_regression_model.joblib
- label_encoder.joblib
- config.json

Taruh di folder models/ sebelum deploy.
Untuk GitHub, file >50MB perlu Git LFS:
  git lfs install
  git lfs track "*.joblib"
  git add .gitattributes
