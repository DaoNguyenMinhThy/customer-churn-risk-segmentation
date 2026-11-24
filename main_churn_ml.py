import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams["axes.unicode_minus"] = False

#1. ĐỌC DỮ LIỆU & KHÁM PHÁ SƠ BỘ
df = pd.read_csv("Churn_Modelling.csv")
print("1. THÔNG TIN DỮ LIỆU")
print(f"Tổng số mẫu: {df.shape[0]}, Tổng số cột: {df.shape[1]}")
print(f"Số lượng giá trị khuyết (NaN): {df.isnull().sum().sum()}")

#Bỏ 3 cột định danh không mang thông tin dự đoán
df_clean = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

X = df_clean.drop(columns=["Exited"])
y = df_clean["Exited"]

ty_le_roi_bo = y.mean() * 100
print(f"Tỷ lệ khách ở lại (0): {100 - ty_le_roi_bo:.2f}% ({sum(y==0)} khách)")
print(f"Tỷ lệ khách rời bỏ (1): {ty_le_roi_bo:.2f}% ({sum(y==1)} khách)")

#Vẽ Hình 1: Phân phối nhãn lệch lớp
plt.figure(figsize=(6, 4))
counts = y.value_counts()
bars = plt.bar(["Ở lại (0)", "Rời bỏ (1)"], counts.values, color=["#2b5c8f", "#d9534f"], width=0.5)
plt.title("Phân phối nhãn khách hàng (Dữ liệu lệch lớp)", fontweight="bold")
plt.ylabel("Số lượng khách hàng")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 120, f"{h} ({h/len(y)*100:.2f}%)", ha="center", fontweight="bold")
plt.ylim(0, 9200)
plt.tight_layout()
plt.savefig("Hinh_1_Phan_phoi_nhan.png", dpi=300)
plt.close()

#2. CHIA TẬP HỌC / KIỂM TRA & XÂY DỰNG PIPELINE TIỀN XỬ LÝ
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTập học: {X_tr.shape[0]} mẫu | Tập kiểm tra: {X_te.shape[0]} mẫu")

cat_cols = ["Geography", "Gender"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ]
)

#3. SO SÁNH 5 MÔ HÌNH BẰNG KIỂM ĐỊNH CHÉO 5-FOLD TRÊN TẬP HỌC
print("\n2. KẾT QUẢ KIỂM ĐỊNH CHÉO 5-FOLD (TẬP HỌC)")
print(f"Mô hình lười: Accuracy = {(1 - y_tr.mean()):.4f}, Recall = 0.0000, F1 = 0.0000")

models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest (200 cây)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM (Kernel RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_summary = []

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    scores = cross_validate(pipe, X_tr, y_tr, cv=5, scoring=scoring, n_jobs=-1)
    cv_summary.append({
        "Mô hình": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    })

df_cv = pd.DataFrame(cv_summary)
print(df_cv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

#4. TINH CHỈNH SIÊU THAM SỐ & TỐI ƯU NGƯỠNG QUYẾT ĐỊNH (CHỈ DÙNG TẬP HỌC)
print("\n3. DÒ LƯỚI THAM SỐ & TỐI ƯU NGƯỠNG TRÊN TẬP HỌC")
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_leaf": [1, 3]
}

grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_tr, y_tr)
best_model = grid.best_estimator_

print("Tham số tối ưu (GridSearchCV theo F1):", grid.best_params_)
print(f"F1 trung bình CV tốt nhất (ngưỡng 0.5): {grid.best_score_:.4f}")

#Dò ngưỡng quyết định tối ưu (tránh gian lận tập test)
y_tr_proba_cv = cross_val_predict(best_model, X_tr, y_tr, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
best_thresh, best_f1_cv = 0.5, 0.0
for t in np.arange(0.20, 0.65, 0.05):
    f1_t = f1_score(y_tr, (y_tr_proba_cv >= t).astype(int))
    if f1_t > best_f1_cv:
        best_f1_cv = f1_t
        best_thresh = round(float(t), 2)

print(f"Ngưỡng tối ưu tìm được trên CV tập học: t = {best_thresh} (F1 CV = {best_f1_cv:.4f})")

#5. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP KIỂM TRA
print("\n4. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM TRA (2500 MẪU)")
y_proba_te = best_model.predict_proba(X_te)[:, 1]
y_pred_default = (y_proba_te >= 0.50).astype(int)
y_pred_opt = (y_proba_te >= best_thresh).astype(int)

for label, y_pred in [("Ngưỡng mặc định (t = 0.50)", y_pred_default), (f"Ngưỡng tối ưu (t = {best_thresh})", y_pred_opt)]:
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    print(f"\n{label}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"Precision: {precision_score(y_te, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_te, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_te, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_te, y_proba_te):.4f}")

#Vẽ Hình 2: So sánh Ma trận nhầm lẫn trước và sau tối ưu ngưỡng
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, y_p, title in zip(
    axes,
    [y_pred_default, y_pred_opt],
    ["Ngưỡng mặc định t = 0.50", f"Ngưỡng tối ưu t = {best_thresh}"]
):
    cm = confusion_matrix(y_te, y_p)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ở lại (0)", "Rời bỏ (1)"])
    ax.set_yticklabels(["Ở lại (0)", "Rời bỏ (1)"])
    ax.set_xlabel("Mô hình dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title(title, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Hinh_2_Ma_tran_nham_lan.png", dpi=300)
plt.close()

#Vẽ Hình 3: Đường cong ROC so sánh các mô hình trên tập kiểm tra
plt.figure(figsize=(7, 5))
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)]).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc_val = roc_auc_score(y_te, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="Đoán ngẫu nhiên (AUC = 0.5000)")
plt.xlabel("Tỷ lệ báo động nhầm (False Positive Rate)")
plt.ylabel("Tỷ lệ tìm ra khách rời bỏ (Recall / TPR)")
plt.title("Đường cong ROC-AUC trên tập kiểm tra", fontweight="bold")
plt.legend(loc="lower right", fontsize=8.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Hinh_3_Duong_cong_ROC.png", dpi=300)
plt.close()

#6. PHÂN TÍCH ĐỘ QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE)
ohe_names = best_model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(ohe_names)
importances = best_model.named_steps["clf"].feature_importances_

df_imp = pd.DataFrame({"Đặc trưng": all_feature_names, "Độ quan trọng": importances}).sort_values("Độ quan trọng", ascending=True)
print("\n5. ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (RANDOM FOREST)")
print(df_imp.sort_values("Độ quan trọng", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

plt.figure(figsize=(7.5, 4.8))
plt.barh(df_imp["Đặc trưng"], df_imp["Độ quan trọng"], color="#1f4e78")
plt.xlabel("Mức độ giảm Gini trung bình (Feature Importance)")
plt.title("Mức độ quan trọng của các đặc trưng trong Random Forest", fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("Hinh_4_Do_quan_trong_dac_trung.png", dpi=300)
plt.close()

#7. PHÂN KHÚC NHÓM KHÁCH HÀNG RỜI BỎ BẰNG K-MEANS (HỌC KHÔNG GIÁM SÁT)
print("\n6. PHÂN KHÚC NHÓM KHÁCH HÀNG RỜI BỎ BẰNG K-MEANS")
df_churn = df_clean[df_clean["Exited"] == 1].copy()
cluster_features = ["Age", "Balance", "CreditScore"]
X_cluster = df_churn[cluster_features]

sc_km = StandardScaler()
X_cluster_scaled = sc_km.fit_transform(X_cluster)

inertias = []
sil_scores = []
K_range = range(2, 8)
for k in K_range:
    km_temp = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_cluster_scaled)
    inertias.append(km_temp.inertia_)
    sil_scores.append(silhouette_score(X_cluster_scaled, km_temp.labels_))
    print(f"K = {k}: Inertia = {km_temp.inertia_:.2f} | Silhouette = {silhouette_score(X_cluster_scaled, km_temp.labels_):.4f}")

# Chọn K=3 cụm để phân nhóm hành vi rõ ràng
K_opt = 3
km_final = KMeans(n_clusters=K_opt, n_init=10, random_state=42).fit(X_cluster_scaled)
df_churn["Cum"] = km_final.labels_

centers_orig = sc_km.inverse_transform(km_final.cluster_centers_)
df_centers = pd.DataFrame(centers_orig, columns=cluster_features)
df_centers["Số khách"] = df_churn["Cum"].value_counts().sort_index().values
df_centers.index = [f"Cụm {i}" for i in range(K_opt)]
print("\nTâm các cụm khách hàng rời bỏ (đã quy đổi về đơn vị gốc):")
print(df_centers.to_string(float_format=lambda x: f"{x:.2f}"))

#Vẽ Hình 5: Elbow, Silhouette và Biểu đồ phân cụm
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

axes[0].plot(list(K_range), inertias, "o-", color="#1f4e78", linewidth=2)
axes[0].set_title("Phương pháp Elbow", fontweight="bold")
axes[0].set_xlabel("Số cụm K")
axes[0].set_ylabel("Inertia")
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_range), sil_scores, "s-", color="#d9534f", linewidth=2)
axes[1].set_title("Chỉ số Silhouette theo K", fontweight="bold")
axes[1].set_xlabel("Số cụm K")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(alpha=0.3)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i in range(K_opt):
    sub = df_churn[df_churn["Cum"] == i]
    axes[2].scatter(sub["Age"], sub["Balance"] / 1000, s=18, alpha=0.65, color=colors[i], label=f"Cụm {i} ({len(sub)} KH)")
axes[2].scatter(centers_orig[:, 0], centers_orig[:, 1] / 1000, c="black", marker="X", s=160, label="Tâm cụm")
axes[2].set_title("Phân cụm khách rời bỏ (K = 3)", fontweight="bold")
axes[2].set_xlabel("Độ tuổi (Age)")
axes[2].set_ylabel("Số dư tài khoản (nghìn USD)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Hinh_5_Phan_cum_KMeans.png", dpi=300)
plt.close()

print("\nĐã xuất 5 biểu đồ vào thư mục")