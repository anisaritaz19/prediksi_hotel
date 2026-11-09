"""
TA-05 - Klasifikasi Pembatalan Reservasi Hotel dengan Decision Tree
Jalankan:  python train.py
Output  :  models/*.joblib, models/meta.json, models/metrics.json, static/img/*.png
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, roc_auc_score)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
IMG = "static/img/"
C0, C1 = "#2a9d8f", "#e76f51"   # tidak batal / batal

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_csv("data/hotel_bookings.csv")
print("Shape:", df.shape)
print(df.head())

# ------------------------------------------------------------------ 2. EDA
# 2.1 distribusi target
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="is_canceled", data=df, palette=[C0, C1])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontweight="bold")
ax.set_xticklabels(["Tidak Batal (0)", "Batal (1)"])
plt.title("Distribusi Kelas Target: is_canceled"); plt.xlabel("Kelas"); plt.ylabel("Jumlah")
plt.tight_layout(); plt.savefig(IMG+"01_countplot_target.png", dpi=150); plt.close()

# 2.2 missing values
mv = df.isna().sum(); mv = mv[mv > 0].sort_values()
plt.figure(figsize=(7, 3.5))
sns.barplot(x=mv.values, y=mv.index, palette="rocket")
for i, v in enumerate(mv.values): plt.text(v, i, f" {v:,}", va="center")
plt.title("Jumlah Missing Values per Kolom"); plt.xlabel("Jumlah NaN")
plt.tight_layout(); plt.savefig(IMG+"02_missing_values.png", dpi=150); plt.close()

# 2.3 hubungan fitur kategorikal vs target
month_order = ["January","February","March","April","May","June","July","August",
               "September","October","November","December"]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col in zip(axes.ravel(), ["hotel", "deposit_type", "market_segment", "customer_type"]):
    r = df.groupby(col)["is_canceled"].mean().sort_values(ascending=False)*100
    sns.barplot(x=r.index, y=r.values, ax=ax, palette="viridis")
    for i, v in enumerate(r.values): ax.text(i, v+1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_title(f"Tingkat Pembatalan menurut {col}"); ax.set_ylabel("% batal"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.savefig(IMG+"03_eda_kategorikal.png", dpi=150); plt.close()

# 2.4 numerik vs target
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(data=df, x="lead_time", hue="is_canceled", bins=50, palette=[C0, C1], ax=axes[0], element="step")
axes[0].set_title("Lead Time vs Pembatalan")
sns.boxplot(data=df, x="is_canceled", y="adr", palette=[C0, C1], ax=axes[1]); axes[1].set_ylim(0, 400)
axes[1].set_title("ADR (harga/malam) vs Pembatalan")
r = df.groupby("total_of_special_requests")["is_canceled"].mean()*100
sns.barplot(x=r.index, y=r.values, palette="mako", ax=axes[2])
axes[2].set_title("Special Requests vs % Batal"); axes[2].set_ylabel("% batal")
plt.tight_layout(); plt.savefig(IMG+"04_eda_numerik.png", dpi=150); plt.close()

# 2.5 batal per bulan
r = df.groupby("arrival_date_month")["is_canceled"].mean().reindex(month_order)*100
plt.figure(figsize=(11, 4)); sns.lineplot(x=r.index, y=r.values, marker="o", color=C1, linewidth=2.5)
plt.xticks(rotation=40); plt.ylabel("% batal"); plt.title("Tingkat Pembatalan per Bulan Kedatangan")
plt.tight_layout(); plt.savefig(IMG+"05_batal_per_bulan.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 3. PREPROCESSING
# Fitur dibuang: reservation_status & reservation_status_date (DATA LEAKAGE - hanya diketahui
# setelah hasil reservasi), assigned_room_type (ditetapkan saat check-in), agent/company (ID),
# country (kardinalitas tinggi), arrival_date_* selain bulan.
data = df.copy()
data["children"] = data["children"].fillna(data["children"].median())       # missing -> median
data["country"]  = data["country"].fillna(data["country"].mode()[0])        # missing -> modus (untuk dokumentasi)
data = data[(data["adults"] + data["children"] + data["babies"]) > 0]       # tamu 0 orang = tidak valid
data = data[data["adr"] >= 0]                                               # harga negatif = outlier error

cat_cols = ["hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel",
            "reserved_room_type", "deposit_type", "customer_type"]
num_cols = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
            "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
            "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
            "total_of_special_requests"]
X, y = data[cat_cols + num_cols], data["is_canceled"]
print("Setelah preprocessing:", X.shape, "| missing:", int(X.isna().sum().sum()))

# korelasi numerik
plt.figure(figsize=(11, 8.5))
sns.heatmap(data[num_cols+["is_canceled"]].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Heatmap Korelasi Fitur Numerik"); plt.tight_layout(); plt.savefig(IMG+"06_korelasi.png", dpi=150); plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")

def make(**kw):
    return Pipeline([("pre", pre), ("tree", DecisionTreeClassifier(random_state=42, **kw))])

# ------------------------------------------------------------------ 4. MODEL A & B
model_A = make(max_depth=None).fit(X_train, y_train)                            # tanpa pruning
model_B = make(max_depth=5, min_samples_leaf=50).fit(X_train, y_train)          # pruning

# sweep kedalaman (bukti overfitting)
depths = list(range(1, 31)); tr, te = [], []
for d in depths:
    m = make(max_depth=d).fit(X_train, y_train)
    tr.append(m.score(X_train, y_train)); te.append(m.score(X_test, y_test))
plt.figure(figsize=(9, 5))
plt.plot(depths, tr, "o-", color=C1, label="Akurasi Latih"); plt.plot(depths, te, "s-", color="#264653", label="Akurasi Uji")
plt.fill_between(depths, te, tr, color=C1, alpha=0.12)
plt.axvline(5, ls="--", color="green", label="Model B (depth=5)")
plt.xlabel("max_depth"); plt.ylabel("Akurasi"); plt.title("Kurva Overfitting: Akurasi vs Kedalaman Pohon")
plt.legend(); plt.tight_layout(); plt.savefig(IMG+"07_overfitting_curve.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 5. EVALUASI
metrics, names = {}, {"A": ("Model A (tanpa pruning)", model_A), "B": ("Model B (pruned)", model_B)}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.figure()
roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
for ax, (k, (nm, m)) in zip(axes, names.items()):
    pred = m.predict(X_test); proba = m.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    print(f"\n===== {nm} =====\nTrain acc {m.score(X_train,y_train):.4f} | Test acc {accuracy_score(y_test,pred):.4f}")
    print(confusion_matrix(y_test, pred)); print(classification_report(y_test, pred, target_names=["Tidak Batal","Batal"]))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if k == "B" else "Oranges", ax=ax,
                xticklabels=["Tidak Batal","Batal"], yticklabels=["Tidak Batal","Batal"], annot_kws={"size": 14})
    ax.set_title(nm); ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya")
    fpr, tpr, _ = roc_curve(y_test, proba); auc = roc_auc_score(y_test, proba)
    roc_ax.plot(fpr, tpr, lw=2.5, color=C1 if k == "A" else "#1d6fb8", label=f"{nm} (AUC={auc:.3f})")
    rep = classification_report(y_test, pred, output_dict=True)
    metrics[k] = dict(name=nm, depth=int(m.named_steps["tree"].get_depth()), leaves=int(m.named_steps["tree"].get_n_leaves()),
                      train_acc=m.score(X_train, y_train), test_acc=accuracy_score(y_test, pred), auc=auc,
                      precision_1=rep["1"]["precision"], recall_1=rep["1"]["recall"], f1_1=rep["1"]["f1-score"],
                      precision_0=rep["0"]["precision"], recall_0=rep["0"]["recall"], f1_0=rep["0"]["f1-score"],
                      cm=cm.tolist())
fig.tight_layout(); fig.savefig(IMG+"08_confusion_matrix.png", dpi=150)
roc_ax.plot([0, 1], [0, 1], "--", color="gray", label="Tebakan acak")
roc_ax.set_xlabel("False Positive Rate"); roc_ax.set_ylabel("True Positive Rate"); roc_ax.set_title("Kurva ROC: Model A vs Model B")
roc_ax.legend(loc="lower right"); roc_fig.tight_layout(); roc_fig.savefig(IMG+"09_roc.png", dpi=150); plt.close("all")

# ------------------------------------------------------------------ 6. VISUALISASI MODEL B
feat = model_B.named_steps["pre"].get_feature_names_out()
feat = [f.replace("cat__", "").replace("remainder__", "") for f in feat]
tree_B = model_B.named_steps["tree"]
plt.figure(figsize=(34, 13))
plot_tree(tree_B, feature_names=feat, class_names=["Tidak Batal", "Batal"], filled=True, rounded=True,
          fontsize=7, proportion=False, impurity=True)
plt.title("Visualisasi Pohon Keputusan - Model B (max_depth=5)", fontsize=18)
plt.tight_layout(); plt.savefig(IMG+"10_tree_modelB.png", dpi=170); plt.close()

imp = pd.Series(tree_B.feature_importances_, index=feat).sort_values(ascending=False)
imp = imp[imp > 0].head(12)[::-1]
plt.figure(figsize=(9, 5.5)); plt.barh(imp.index, imp.values, color=sns.color_palette("viridis", len(imp)))
for i, v in enumerate(imp.values): plt.text(v, i, f" {v:.3f}", va="center")
plt.title("Feature Importance - Model B"); plt.xlabel("Nilai Pentingnya")
plt.tight_layout(); plt.savefig(IMG+"11_feature_importance.png", dpi=150); plt.close()

rules = export_text(tree_B, feature_names=feat, max_depth=5)
open("reports/tree_rules_modelB.txt", "w").write(rules); print(rules[:1800])
metrics["importance"] = {k: float(v) for k, v in imp[::-1].items()}
metrics["sweep"] = dict(depths=depths, train=tr, test=te)
metrics["n_rows_raw"] = int(len(df)); metrics["n_rows_clean"] = int(len(data))
metrics["cancel_rate"] = float(data.is_canceled.mean())

# ------------------------------------------------------------------ 7. SIMPAN
joblib.dump(model_B, "models/model_pruned.joblib", compress=3)
joblib.dump(model_A, "models/model_full.joblib", compress=3)
meta = dict(
    cat={c: sorted(map(str, data[c].unique())) for c in cat_cols},
    num={c: dict(min=float(data[c].min()), max=float(data[c].quantile(0.999)), median=float(data[c].median())) for c in num_cols},
    cat_cols=cat_cols, num_cols=num_cols)
json.dump(meta, open("models/meta.json", "w"), indent=1)
json.dump(metrics, open("models/metrics.json", "w"), indent=1)
print("\nSELESAI. Model B disimpan di models/model_pruned.joblib")
