# HotelGuard – Prediksi Pembatalan Reservasi Hotel (Decision Tree)
TA-05 Pembelajaran Mesin. Dataset: Hotel Booking Demand (Kaggle, `jessemostipak/hotel-booking-demand`).

## Jalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# letakkan dataset di data/hotel_bookings.csv (sudah disertakan di repo)
python train.py        # EDA, training Model A & B, simpan model + gambar
python app.py          # buka http://127.0.0.1:5000
```
## Deploy ke Render
1. Push folder ini ke GitHub (termasuk `data/hotel_bookings.csv`, ±17 MB).
2. Render → New → Blueprint (membaca `render.yaml`) atau Web Service:
   Build `pip install -r requirements.txt && python train.py`, Start `gunicorn app:app`.
3. Railway: New Project → Deploy from GitHub; `Procfile` terbaca otomatis (tambahkan build `python train.py` bila `models/` tidak di-commit).
## Struktur
`train.py` pipeline lengkap · `app.py` Flask · `templates/` `static/` UI · `models/` joblib · `notebook.ipynb` analisis
