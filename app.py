"""
Aplikasi Web Flask - Prediksi Pembatalan Reservasi Hotel (Decision Tree)
Jalankan lokal :  python app.py   ->  http://127.0.0.1:5000
Produksi       :  gunicorn app:app
"""
import json, os
import joblib, pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ---- muat model & metadata (dilakukan sekali saat start) ----
MODEL_B = joblib.load(os.path.join(BASE, "models", "model_pruned.joblib"))   # model terbaik (pruned)
MODEL_A = joblib.load(os.path.join(BASE, "models", "model_full.joblib"))     # pembanding (overfit)
META = json.load(open(os.path.join(BASE, "models", "meta.json")))
METRICS = json.load(open(os.path.join(BASE, "models", "metrics.json")))
CAT, NUM = META["cat_cols"], META["num_cols"]
FEATS = [f.replace("cat__", "").replace("remainder__", "")
         for f in MODEL_B.named_steps["pre"].get_feature_names_out()]

LABELS = {
    "hotel": "Jenis Hotel", "arrival_date_month": "Bulan Kedatangan", "meal": "Paket Makan",
    "market_segment": "Segmen Pasar", "distribution_channel": "Kanal Distribusi",
    "reserved_room_type": "Tipe Kamar Dipesan", "deposit_type": "Jenis Deposit",
    "customer_type": "Tipe Pelanggan", "lead_time": "Lead Time (hari sebelum check-in)",
    "stays_in_weekend_nights": "Malam Akhir Pekan", "stays_in_week_nights": "Malam Hari Kerja",
    "adults": "Dewasa", "children": "Anak-anak", "babies": "Bayi",
    "is_repeated_guest": "Tamu Berulang (0/1)", "previous_cancellations": "Pembatalan Sebelumnya",
    "previous_bookings_not_canceled": "Booking Sebelumnya (tidak batal)",
    "booking_changes": "Jumlah Perubahan Booking", "days_in_waiting_list": "Hari di Waiting List",
    "adr": "Harga Rata-rata/Malam (ADR, EUR)", "required_car_parking_spaces": "Tempat Parkir Dibutuhkan",
    "total_of_special_requests": "Jumlah Permintaan Khusus",
}
GROUPS = [
    ("Detail Menginap", ["hotel", "arrival_date_month", "lead_time", "stays_in_weekend_nights",
                          "stays_in_week_nights", "meal", "reserved_room_type", "adr"]),
    ("Tamu", ["adults", "children", "babies", "customer_type", "is_repeated_guest",
              "previous_cancellations", "previous_bookings_not_canceled"]),
    ("Pemesanan", ["market_segment", "distribution_channel", "deposit_type", "booking_changes",
                   "days_in_waiting_list", "required_car_parking_spaces", "total_of_special_requests"]),
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
DEFAULTS = {"hotel": "City Hotel", "arrival_date_month": "July", "meal": "BB", "market_segment": "Online TA",
            "distribution_channel": "TA/TO", "reserved_room_type": "A", "deposit_type": "No Deposit",
            "customer_type": "Transient", "lead_time": 60, "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2, "adults": 2, "children": 0, "babies": 0, "is_repeated_guest": 0,
            "previous_cancellations": 0, "previous_bookings_not_canceled": 0, "booking_changes": 0,
            "days_in_waiting_list": 0, "adr": 100, "required_car_parking_spaces": 0,
            "total_of_special_requests": 0}
PRESETS = {
    "berisiko": {**DEFAULTS, "lead_time": 220, "deposit_type": "Non Refund", "market_segment": "Groups",
                 "distribution_channel": "TA/TO", "customer_type": "Transient-Party", "adr": 85},
    "aman": {**DEFAULTS, "lead_time": 5, "total_of_special_requests": 2, "required_car_parking_spaces": 1,
             "market_segment": "Direct", "distribution_channel": "Direct", "is_repeated_guest": 1,
             "previous_bookings_not_canceled": 3},
    "ragu": {**DEFAULTS, "lead_time": 120, "market_segment": "Online TA", "total_of_special_requests": 0,
             "previous_cancellations": 1},
}


def months_ordered():
    return [m for m in MONTHS if m in META["cat"]["arrival_date_month"]]


def parse(form):
    """Validasi & konversi input menjadi satu baris DataFrame."""
    row = {}
    for c in CAT:
        v = str(form.get(c, DEFAULTS[c]))
        if v not in META["cat"][c]:
            raise ValueError(f"Nilai '{v}' tidak valid untuk {LABELS[c]}")
        row[c] = v
    for c in NUM:
        try:
            v = float(form.get(c, DEFAULTS[c]))
        except (TypeError, ValueError):
            raise ValueError(f"{LABELS[c]} harus berupa angka")
        if v < 0:
            raise ValueError(f"{LABELS[c]} tidak boleh negatif")
        row[c] = v
    if row["adults"] + row["children"] + row["babies"] <= 0:
        raise ValueError("Jumlah tamu minimal 1 orang")
    return pd.DataFrame([row])[CAT + NUM]


def explain_path(model, df):
    """Terjemahkan jalur keputusan dari akar ke daun menjadi aturan 'jika-maka'."""
    Xt = model.named_steps["pre"].transform(df)
    tree = model.named_steps["tree"]
    t = tree.tree_
    node_ids = tree.decision_path(Xt).indices
    steps = []
    for nid in node_ids[:-1]:
        f, thr = FEATS[t.feature[nid]], t.threshold[nid]
        val = float(Xt[0, t.feature[nid]]) if not hasattr(Xt, "toarray") else float(Xt.toarray()[0, t.feature[nid]])
        left = val <= thr
        if "_" in f and thr == 0.5 and any(f.startswith(c + "_") for c in CAT):   # fitur one-hot
            col = next(c for c in CAT if f.startswith(c + "_"))
            cat_val = f[len(col) + 1:]
            text = f"{LABELS[col]} {'BUKAN' if left else 'adalah'} “{cat_val}”"
        else:
            text = f"{LABELS.get(f, f)} {'≤' if left else '>'} {thr:.1f}" + f"  (input: {val:g})"
        steps.append(dict(text=text, samples=int(t.n_node_samples[nid])))
    leaf = node_ids[-1]
    v = t.value[leaf][0]; v = v / v.sum()
    return steps, dict(samples=int(t.n_node_samples[leaf]), p_cancel=float(v[1]), depth=len(steps))


def run(form):
    df = parse(form)
    pb = float(MODEL_B.predict_proba(df)[0, 1])
    pa = float(MODEL_A.predict_proba(df)[0, 1])
    steps, leaf = explain_path(MODEL_B, df)
    level = "tinggi" if pb >= 0.65 else "sedang" if pb >= 0.4 else "rendah"
    return dict(label="BATAL" if pb >= 0.5 else "TIDAK BATAL", prob_cancel=pb, prob_keep=1 - pb,
                level=level, model_a=dict(label="BATAL" if pa >= 0.5 else "TIDAK BATAL", prob_cancel=pa),
                path=steps, leaf=leaf)


@app.route("/")
def index():
    return render_template("index.html", groups=GROUPS, labels=LABELS, cat=META["cat"], num=META["num"],
                           months=months_ordered(), defaults=DEFAULTS, presets=PRESETS, page="home")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True) or request.form
        return jsonify(ok=True, **run(data))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception as e:                       # pragma: no cover
        return jsonify(ok=False, error=f"Terjadi kesalahan: {e}"), 500


@app.route("/model")
def model_page():
    return render_template("model.html", m=METRICS, page="model")


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
