import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Cấu hình trang Web
st.set_page_config(
    page_title="Dự đoán Khách hàng Rời bỏ & Phân khúc Nguy cơ",
    page_icon="🏦",
    layout="wide"
)

OPTIMAL_THRESH = 0.35

@st.cache_resource
def load_and_train_models():
    df = pd.read_csv("Churn_Modelling.csv")
    df_clean = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    X = df_clean.drop(columns=["Exited"])
    y = df_clean["Exited"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    cat_cols = ["Geography", "Gender"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
        ]
    )

    # Mô hình Random Forest tối ưu sau GridSearchCV
    best_model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])
    best_model.fit(X_tr, y_tr)

    # Đánh giá trên tập kiểm tra tại ngưỡng tối ưu t = 0.35
    y_proba_te = best_model.predict_proba(X_te)[:, 1]
    y_pred_te = (y_proba_te >= OPTIMAL_THRESH).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred_te),
        "Precision": precision_score(y_te, y_pred_te),
        "Recall": recall_score(y_te, y_pred_te),
        "F1-score": f1_score(y_te, y_pred_te),
        "ROC-AUC": roc_auc_score(y_te, y_proba_te)
    }

    # Huấn luyện K-Means (K = 3) trên nhóm khách hàng rời bỏ
    df_churn = df_clean[df_clean["Exited"] == 1].copy()
    cluster_features = ["Age", "Balance", "CreditScore"]
    sc_km = StandardScaler()
    X_cluster_scaled = sc_km.fit_transform(df_churn[cluster_features])

    km_final = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_cluster_scaled)
    centers_orig = sc_km.inverse_transform(km_final.cluster_centers_)

    # Tự động gán nhãn chân dung & chiến lược theo đặc điểm tâm cụm
    cluster_profiles = {}
    for idx, (age_c, bal_c, cred_c) in enumerate(centers_orig):
        if bal_c < 20000:
            cluster_profiles[idx] = {
                "name": f"Cụm {idx} – Nhóm tài khoản rỗng / ít giao dịch",
                "desc": f"Tuổi TB ~{age_c:.1f} | Số dư TB ~${bal_c:,.0f} | Điểm tín dụng TB ~{cred_c:.0f}",
                "action": "Gửi ưu đãi miễn phí chuyển khoản, hoàn tiền thanh toán hóa đơn để kích hoạt lại thẻ."
            }
        elif cred_c < 620:
            cluster_profiles[idx] = {
                "name": f"Cụm {idx} – Nhóm số dư lớn nhưng điểm tín dụng thấp",
                "desc": f"Tuổi TB ~{age_c:.1f} | Số dư TB ~${bal_c:,.0f} | Điểm tín dụng TB ~{cred_c:.0f}",
                "action": "Cử chuyên viên tư vấn cơ cấu khoản vay, hỗ trợ cải thiện và nâng hạng điểm tín dụng."
            }
        else:
            cluster_profiles[idx] = {
                "name": f"Cụm {idx} – Nhóm khách hàng VIP (Giá trị cao)",
                "desc": f"Tuổi TB ~{age_c:.1f} | Số dư TB ~${bal_c:,.0f} | Điểm tín dụng TB ~{cred_c:.0f}",
                "action": "Ưu tiên chăm sóc 1-1, áp dụng lãi suất tiết kiệm bậc thang đặc quyền và tặng thẻ hội viên."
            }

    return best_model, sc_km, km_final, cluster_profiles, metrics, df_clean

best_model, sc_km, km_final, cluster_profiles, metrics, df_clean = load_and_train_models()

# GIAO DIỆN CHÍNH
st.title("🏦 Hệ thống Dự đoán Khách hàng Rời bỏ & Phân khúc Nhóm Nguy cơ")
st.caption(f"Mô hình chủ lực: Random Forest (300 cây) | Ngưỡng quyết định tối ưu: t = {OPTIMAL_THRESH} | Phân cụm hành vi: K-Means (K = 3)")

tab1, tab2, tab3 = st.tabs([
    "🔍 Dự đoán & Tư vấn Cá nhân",
    "📂 Quét Danh sách Hàng loạt (CSV)",
    "📊 Hiệu năng Mô hình & Phân khúc"
])

# TAB 1: DỰ ĐOÁN CÁ NHÂN
with tab1:
    st.subheader("Nhập thông tin hồ sơ khách hàng")
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Điểm tín dụng (CreditScore)", min_value=300, max_value=900, value=650)
        geography = st.selectbox("Quốc gia (Geography)", ["France", "Germany", "Spain"])
        gender = st.selectbox("Giới tính (Gender)", ["Female", "Male"])
        age = st.slider("Độ tuổi (Age)", min_value=18, max_value=95, value=45)

    with col2:
        tenure = st.slider("Số năm gắn bó (Tenure)", min_value=0, max_value=10, value=5)
        balance = st.number_input("Số dư tài khoản - USD (Balance)", min_value=0.0, max_value=300000.0, value=120000.0, step=1000.0)
        num_products = st.selectbox("Số lượng sản phẩm sử dụng (NumOfProducts)", [1, 2, 3, 4], index=0)

    with col3:
        has_cr_card = st.selectbox("Sở hữu thẻ tín dụng (HasCrCard)", [1, 0], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
        is_active = st.selectbox("Tương tác thường xuyên (IsActiveMember)", [1, 0], index=1, format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
        est_salary = st.number_input("Thu nhập ước tính hàng năm - USD (EstimatedSalary)", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)

    if st.button("🚀 Phân tích Nguy cơ Rời bỏ", type="primary"):
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": est_salary
        }])

        prob_churn = best_model.predict_proba(input_df)[0, 1]
        is_high_risk = prob_churn >= OPTIMAL_THRESH

        # Xác định cụm hành vi gần nhất bằng K-Means
        km_input = sc_km.transform(input_df[["Age", "Balance", "CreditScore"]])
        cluster_id = int(km_final.predict(km_input)[0])
        profile = cluster_profiles[cluster_id]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Xác suất rời bỏ dịch vụ", f"{prob_churn * 100:.2f}%", delta=f"Ngưỡng cảnh báo: {OPTIMAL_THRESH * 100:.0f}%", delta_color="inverse")
            if is_high_risk:
                st.error("⚠️ CẢNH BÁO: Khách hàng thuộc nhóm NGUY CƠ RỜI BỎ CAO")
            else:
                st.success("✅ AN TOÀN: Khách hàng có xu hướng tiếp tục gắn bó")

        with res_col2:
            st.markdown(f"**Phân khúc hành vi (K-Means):** `{profile['name']}`")
            st.caption(profile["desc"])
            if is_high_risk:
                st.info(f"💡 **Đề xuất chiến lược giữ chân:** {profile['action']}")
            else:
                st.write("💡 **Khuyến nghị:** Tiếp tục duy trì trải nghiệm dịch vụ tiêu chuẩn và chăm sóc định kỳ.")

# TAB 2: QUÉT HÀNG LOẠT
with tab2:
    st.subheader("Tải lên danh sách khách hàng (.csv) để sàng lọc tự động")
    st.write("Tệp CSV cần có các cột đặc trưng giống bộ dữ liệu `Churn_Modelling.csv`.")

    uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

        if all(col in batch_df.columns for col in required_cols):
            probs = best_model.predict_proba(batch_df[required_cols])[:, 1]
            preds = (probs >= OPTIMAL_THRESH).astype(int)

            km_batch = sc_km.transform(batch_df[["Age", "Balance", "CreditScore"]])
            clusters = km_final.predict(km_batch)

            result_df = batch_df.copy()
            result_df["Xac_suat_roi_bo (%)"] = np.round(probs * 100, 2)
            result_df["Du_doan_nguy_co"] = np.where(preds == 1, "Nguy cơ cao", "An toàn")
            result_df["Phan_khuc_KMeans"] = [cluster_profiles[c]["name"] for c in clusters]

            high_risk_count = int(preds.sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng hồ sơ đã quét", f"{len(result_df):,}")
            c2.metric("Khách hàng nguy cơ cao", f"{high_risk_count:,}")
            c3.metric("Tỷ lệ cảnh báo", f"{high_risk_count / len(result_df) * 100:.2f}%")

            st.dataframe(result_df.sort_values("Xac_suat_roi_bo (%)", ascending=False), use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải xuống kết quả dự đoán (.csv)",
                data=csv_out,
                file_name="Ket_qua_du_doan_Churn.csv",
                mime="text/csv"
            )
        else:
            st.error("Tệp CSV thiếu một số cột đặc trưng bắt buộc!")

# TAB 3: THÔNG SỐ MÔ HÌNH
with tab3:
    st.subheader("Chỉ số đánh giá Random Forest tối ưu trên tập kiểm tra (2.500 mẫu)")
    m_cols = st.columns(5)
    for col, (k, v) in zip(m_cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    st.divider()
    st.subheader("Chân dung 3 nhóm khách hàng rời bỏ (K-Means, K = 3)")
    summary_rows = []
    for cid in sorted(cluster_profiles.keys()):
        summary_rows.append({
            "Mã cụm": f"Cụm {cid}",
            "Định danh nhóm": cluster_profiles[cid]["name"],
            "Đặc điểm trung bình": cluster_profiles[cid]["desc"],
            "Chiến lược giữ chân": cluster_profiles[cid]["action"]
        })
    st.table(pd.DataFrame(summary_rows))