import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CLEAN_CSV_FILE = "clean_openalex_papers.csv"
FORECAST_FOCUS_CSV = "forecast_2025_2027_top6_universities.csv"
FORECAST_ALL_CSV = "forecast_2025_2027_all20_universities.csv"

FOCAL_VN_UNIS = ["HUST", "HCMUT", "PTIT"]
FOCAL_GLOBAL_UNIS = ["MIT", "Stanford", "CMU"]
FOCAL_ALL = FOCAL_VN_UNIS + FOCAL_GLOBAL_UNIS


def build_institution_year_panel(csv_path=CLEAN_CSV_FILE):
    """Tổng hợp dữ liệu từ 7,598 bài báo sang bảng Panel (20 Trường x 6 Năm = 120 dòng)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy '{csv_path}'. Hãy đảm bảo file clean_openalex_papers.csv nằm cùng thư mục!"
        )

    print(f"📂 [Bước 1/4] Đang đọc và tổng hợp dữ liệu từ '{csv_path}'...")
    df = pd.read_csv(csv_path)

    df["pub_dt"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["day_of_year"] = df["pub_dt"].dt.dayofyear.fillna(180)
    snapshot_dt = pd.Timestamp("2025-01-15")
    df["months_exposed"] = np.maximum(0.8, (snapshot_dt - df["pub_dt"]).dt.days / 30.44)
    df["cites_per_month"] = df["cited_by_count"] / df["months_exposed"]

    df["is_core_ai"] = df["subfield"].isin(
        ["Artificial Intelligence", "Computer Vision and Pattern Recognition"]
    ).astype(int)
    df["is_intl"] = (df["countries_distinct_count"] > 1).astype(int)
    df["is_oa_int"] = df["is_oa"].astype(int)

    df["is_top10"] = np.where(
        df["months_exposed"] >= 2.5,
        (df["citation_percentile"] >= 0.90).astype(int),
        ((df["citation_percentile"] >= 0.85) | (df["cites_per_month"] >= 2.2)).astype(int),
    )

    panel = df.groupby(["region", "institution", "publication_year"], as_index=False).agg(
        crawled_papers=("work_id", "count"),
        core_ai_papers=("is_core_ai", "sum"),
        top10_crawled=("is_top10", "sum"),
        avg_fwci=("fwci", "mean"),
        avg_percentile=("citation_percentile", "mean"),
        avg_refs=("referenced_works_count", "mean"),
        avg_authors=("authors_count", "mean"),
        avg_insts=("institutions_distinct_count", "mean"),
        intl_collab_rate=("is_intl", lambda s: s.mean() * 100.0),
        oa_rate=("is_oa_int", lambda s: s.mean() * 100.0),
        min_day=("day_of_year", "min"),
        avg_exposure_months=("months_exposed", "mean"),
    )

    panel.sort_values(["institution", "publication_year"], inplace=True, ignore_index=True)

    # Khôi phục tổng sản lượng bài báo năm (total_papers_est) từ mật độ ngày xuất bản khi chạm trần 80 bài
    panel["raw_window_days"] = np.where(
        panel["crawled_papers"] >= 80,
        np.maximum(24.0, 366.0 - panel["min_day"]),
        365.0,
    )
    panel["smooth_window_days"] = panel.groupby("institution")["raw_window_days"].transform(
        lambda s: s.rolling(2, min_periods=1).mean()
    )
    panel["total_papers_est"] = np.where(
        panel["crawled_papers"] >= 80,
        np.round(panel["crawled_papers"] * (365.0 / panel["smooth_window_days"])).astype(int),
        panel["crawled_papers"],
    )

    # Tính số lượng bài báo lọt Top 10% ước tính trên tổng sản lượng năm
    panel["top10_rate_pct"] = (panel["top10_crawled"] / np.maximum(1, panel["crawled_papers"])) * 100.0
    panel["top10_rate_smooth"] = panel.groupby("institution")["top10_rate_pct"].transform(
        lambda s: 0.65 * s + 0.35 * s.shift(1).bfill()
    )
    panel["top10_papers_est"] = np.maximum(
        panel["top10_crawled"],
        np.round(panel["total_papers_est"] * (panel["top10_rate_smooth"] / 100.0)),
    ).astype(int)

    # Hiệu chỉnh nhẹ FWCI năm 2024 cho các trường có thời gian phơi sáng trích dẫn < 2 tháng
    panel["fwci_adj"] = panel.groupby("institution")["avg_fwci"].transform(
        lambda s: np.where(
            panel.loc[s.index, "publication_year"] == 2024,
            0.45 * s + 0.55 * s.shift(1).bfill(),
            s,
        )
    ).round(3)

    # Tính Điểm Tin cậy Dữ liệu (Data_Confidence_Score: 0 - 100%) làm sample_weight
    maturity_score = np.clip(panel["avg_exposure_months"] / 12.0, 0.45, 1.0) * 100.0
    sample_score = np.clip(panel["crawled_papers"] / 30.0, 0.50, 1.0) * 100.0
    panel["data_confidence"] = (0.65 * maturity_score + 0.35 * sample_score).round(1)

    # Xây dựng Chỉ số Vị thế Học thuật Tổng hợp (Academic_Index) & Nhãn Phân tầng Vị thế (Top ? %)
    panel["academic_index"] = (
        28.0 * np.log10(panel["total_papers_est"] + 1.0)
        + 22.0 * np.log10(panel["top10_papers_est"] + 1.0)
        + 12.0 * np.log10(panel["core_ai_papers"] + 1.0)
        + 6.5 * np.clip(panel["fwci_adj"], 0, 5.0)
        + 0.15 * panel["intl_collab_rate"]
    ).round(2)

    def assign_position_tier(score):
        if score >= 168.0:
            return "1. Top 5% Thế giới (World Elite)"
        if score >= 120.0:
            return "2. Top 15% Thế giới (Global Leading)"
        if score >= 88.0:
            return "3. Top 30% Thế giới (High-Impact Regional)"
        return "4. Top 50% Thế giới (Emerging Tech Tier)"

    panel["position_tier"] = panel["academic_index"].apply(assign_position_tier)
    print(f"  ✅ Đã tổng hợp 7,598 bài báo thành {len(panel)} bản ghi Panel (20 Trường ĐH x 6 Năm: 2019–2024).")
    print(f"  🛡️ Điểm tin cậy dữ liệu trung bình (Data_Confidence_Score): {panel['data_confidence'].mean():.1f}%")
    return panel


def add_lag_features(panel_df):
    """Tạo các biến trễ (Lag-1) và tốc độ tăng trưởng để làm đầu vào cho Mô hình Lớp 1."""
    df_m = panel_df.copy()
    df_m.sort_values(["institution", "publication_year"], inplace=True, ignore_index=True)

    df_m["lag1_total_papers"] = df_m.groupby("institution")["total_papers_est"].shift(1)
    df_m["lag1_top10_papers"] = df_m.groupby("institution")["top10_papers_est"].shift(1)
    df_m["lag1_core_ai"] = df_m.groupby("institution")["core_ai_papers"].shift(1)
    df_m["lag1_fwci"] = df_m.groupby("institution")["fwci_adj"].shift(1)
    df_m["lag1_intl_rate"] = df_m.groupby("institution")["intl_collab_rate"].shift(1)
    df_m["lag1_refs"] = df_m.groupby("institution")["avg_refs"].shift(1)

    # Tốc độ tăng trưởng bài báo trung bình của trường
    df_m["inst_growth_momentum"] = (
        df_m.groupby("institution")["total_papers_est"]
        .transform(lambda s: s.pct_change().rolling(2, min_periods=1).mean())
        .fillna(0.06)
        .clip(-0.15, 0.35)
    )
    df_m["is_vietnam"] = (df_m["region"] == "Vietnam").astype(int)

    # Loại bỏ năm đầu tiên (2019) khi đưa vào huấn luyện vì không có biến trễ năm 2018
    df_train_ready = df_m.dropna(subset=["lag1_total_papers", "lag1_top10_papers"]).copy()
    return df_m, df_train_ready


def run_two_tier_ml_pipeline():
    panel = build_institution_year_panel()
    full_panel, model_df = add_lag_features(panel)

    # BƯỚC 2: HUẤN LUYỆN LỚP 1 - LINEAR REGRESSION (DỰ BÁO SỐ BÀI BÁO & TOP 10%)
    print("\n" + "=" * 80)
    print("📈 [Bước 2/4] LỚP 1: MÔ HÌNH LINEAR REGRESSION (DỰ BÁO SẢN LƯỢNG & BÀI TOP 10%)")
    print("=" * 80)

    tier1_features = [
        "publication_year",
        "is_vietnam",
        "lag1_total_papers",
        "lag1_top10_papers",
        "lag1_core_ai",
        "lag1_fwci",
        "lag1_intl_rate",
        "lag1_refs",
        "inst_growth_momentum",
    ]

    train_mask = model_df["publication_year"] <= 2023
    test_mask = model_df["publication_year"] == 2024

    X_train_t1 = model_df.loc[train_mask, tier1_features]
    X_test_t1 = model_df.loc[test_mask, tier1_features]
    w_train_t1 = model_df.loc[train_mask, "data_confidence"].values / 100.0

    # Khởi tạo 4 mô hình Linear Regression cho 4 chỉ số mục tiêu ở Lớp 1
    targets_t1 = {
        "total_papers_est": "Tổng số bài báo CS/AI năm",
        "top10_papers_est": "Số bài báo lọt Top 10% xuất sắc",
        "core_ai_papers": "Số bài báo AI & Computer Vision lõi",
        "fwci_adj": "Chỉ số tác động trích dẫn FWCI",
    }

    lin_models = {}
    for target_col, target_desc in targets_t1.items():
        y_tr = model_df.loc[train_mask, target_col]
        y_te = model_df.loc[test_mask, target_col]

        reg = LinearRegression()
        reg.fit(X_train_t1, y_tr, sample_weight=w_train_t1)
        lin_models[target_col] = reg

        y_pred_te = reg.predict(X_test_t1)
        r2 = r2_score(y_te, y_pred_te)
        mae = mean_absolute_error(y_te, y_pred_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))

        print(f"🔹 Mục tiêu: {target_desc} ({target_col})")
        print(f"   • R² Score (Tập Test) : {r2:.4f} ({r2*100:.2f}%) | MAE: ±{mae:.2f} | RMSE: {rmse:.2f}")

    # In bảng hệ số góc w (Coefficients) của mô hình Dự báo Số bài lọt Top 10%
    coef_top10_df = pd.DataFrame({
        "Đặc trưng đầu vào (Feature)": tier1_features,
        "Trọng số w ( Dự báo Bài báo)": np.round(lin_models["total_papers_est"].coef_, 4),
        "Trọng số w (Dự báo Top 10%)": np.round(lin_models["top10_papers_est"].coef_, 4),
    }).sort_values(by="Trọng số w (Dự báo Top 10%)", ascending=False)
    print("\n📊 Bảng Trọng số Tuyến tính (Weights w) của Lớp 1:")
    print(coef_top10_df.to_string(index=False))

    # ==========================================================================
    # BƯỚC 3: HUẤN LUYỆN LỚP 2 - SOFTMAX REGRESSION (DỰ ĐOÁN VỊ THẾ TOP ? %)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🎯 [Bước 3/4] LỚP 2: MÔ HÌNH SOFTMAX REGRESSION (DỰ ĐOÁN VỊ THẾ HỌC THUẬT TOP ? %)")
    print("=" * 80)

    # Đầu vào của Lớp 2 chính là các chỉ số sản lượng bài báo, bài Top 10%, AI lõi, FWCI và hợp tác quốc tế
    tier2_features = [
        "total_papers_est",
        "top10_papers_est",
        "core_ai_papers",
        "fwci_adj",
        "intl_collab_rate",
    ]

    X_train_t2 = model_df.loc[train_mask, tier2_features]
    y_train_t2 = model_df.loc[train_mask, "position_tier"]
    X_test_t2 = model_df.loc[test_mask, tier2_features]
    y_test_t2 = model_df.loc[test_mask, "position_tier"]

    scaler_t2 = StandardScaler()
    X_train_t2_scaled = scaler_t2.fit_transform(X_train_t2)
    X_test_t2_scaled = scaler_t2.transform(X_test_t2)

    # Huấn luyện Softmax Regression (Multinomial Cross-Entropy Loss với L-BFGS)
    softmax_model = LogisticRegression(
        solver="lbfgs", C=2.5, max_iter=1000, random_state=42
    )
    softmax_model.fit(X_train_t2_scaled, y_train_t2, sample_weight=w_train_t1)

    y_pred_t2 = softmax_model.predict(X_test_t2_scaled)
    acc_t2 = accuracy_score(y_test_t2, y_pred_t2)

    print(f"  • Độ chính xác phân loại vị thế (Test Accuracy): {acc_t2 * 100:.2f}%")
    print("  • Báo cáo phân loại chi tiết (Precision / Recall / F1-Score):")
    print(classification_report(y_test_t2, y_pred_t2, digits=4))

    # BƯỚC 4: DỰ BÁO CUỐN CHIẾU 3 NĂM TỚI (2025, 2026, 2027) KẾT HỢP LỚP 1 -> LỚP 2
    print("=" * 80)
    print("🔮 [Bước 4/4] DỰ BÁO 3 NĂM TỚI (2025 – 2027): 3 ĐH VIỆT NAM (CÓ PTIT) & 3 ĐH THẾ GIỚI")
    print("=" * 80)

    current_state = full_panel[full_panel["publication_year"] == 2024].copy()
    forecast_records = []
    class_labels = list(softmax_model.classes_)

    for future_yr in [2025, 2026, 2027]:
        step_df = current_state.copy()
        step_df["publication_year"] = future_yr

        # Gán kết quả của năm liền trước (t-1) vào các biến trễ (Lag-1)
        step_df["lag1_total_papers"] = current_state["total_papers_est"].values
        step_df["lag1_top10_papers"] = current_state["top10_papers_est"].values
        step_df["lag1_core_ai"] = current_state["core_ai_papers"].values
        step_df["lag1_fwci"] = current_state["fwci_adj"].values
        step_df["lag1_intl_rate"] = current_state["intl_collab_rate"].values
        step_df["lag1_refs"] = current_state["avg_refs"].values

        X_future_t1 = step_df[tier1_features]

        # --- LỚP 1: DỰ ĐOÁN SỐ BÀI BÁO, SỐ BÀI TOP 10%, AI LÕI & FWCI ---
        raw_pred_papers = lin_models["total_papers_est"].predict(X_future_t1)
        # Đảm bảo tính thực tế theo đà tăng trưởng lịch sử của từng trường
        momentum_floor = current_state["total_papers_est"].values * (
            1.0 + np.maximum(0.04, current_state["inst_growth_momentum"].values * 0.65)
        )
        pred_papers = np.maximum(momentum_floor, raw_pred_papers).round().astype(int)

        raw_pred_top10 = lin_models["top10_papers_est"].predict(X_future_t1)
        min_top10 = np.maximum(
            1, np.round(pred_papers * (current_state["top10_papers_est"].values / np.maximum(1, current_state["total_papers_est"].values)))
        )
        pred_top10 = np.clip(
            np.round(0.55 * raw_pred_top10 + 0.45 * min_top10),
            1,
            np.round(pred_papers * 0.48),
        ).astype(int)

        raw_pred_core_ai = lin_models["core_ai_papers"].predict(X_future_t1)
        pred_core_ai = np.clip(
            np.round(0.6 * raw_pred_core_ai + 0.4 * current_state["core_ai_papers"].values * 1.06),
            2,
            80,
        ).astype(int)

        raw_pred_fwci = lin_models["fwci_adj"].predict(X_future_t1)
        pred_fwci = np.clip(
            0.5 * raw_pred_fwci + 0.5 * current_state["fwci_adj"].values * 1.02,
            0.85,
            5.50,
        ).round(2)

        # Cập nhật dự báo từ Lớp 1 vào bảng trạng thái năm future_yr
        step_df["total_papers_est"] = pred_papers
        step_df["top10_papers_est"] = pred_top10
        step_df["core_ai_papers"] = pred_core_ai
        step_df["fwci_adj"] = pred_fwci
        # Hợp tác quốc tế tăng nhẹ 1.2%/năm theo xu hướng hội nhập
        step_df["intl_collab_rate"] = np.clip(current_state["intl_collab_rate"].values + 1.2, 15.0, 85.0).round(1)

        # --- LỚP 2: ĐẨY DỰ BÁO TĂNG TRƯỞNG BÀI BÁO TỪ LỚP 1 VÀO SOFTMAX REGRESSION ---
        X_future_t2 = step_df[tier2_features]
        X_future_t2_scaled = scaler_t2.transform(X_future_t2)

        pred_tiers = softmax_model.predict(X_future_t2_scaled)
        pred_probas = softmax_model.predict_proba(X_future_t2_scaled) * 100.0

        step_df["Pred_Position_Tier"] = pred_tiers
        step_df["Confidence_Max_Prob (%)"] = np.round(pred_probas.max(axis=1), 1)

        for c_idx, c_name in enumerate(class_labels):
            short_col = c_name.split("(")[0].replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").strip()
            step_df[f"Prob_{short_col} (%)"] = np.round(pred_probas[:, c_idx], 1)

        step_df["Paper_Growth_YoY (%)"] = np.round(
            ((step_df["total_papers_est"] - current_state["total_papers_est"].values)
             / np.maximum(1, current_state["total_papers_est"].values)) * 100.0,
            1,
        )

        forecast_records.append(step_df)
        # Cuốn chiếu: Dùng kết quả dự báo của năm nay làm đầu vào (Lag-1) cho năm tiếp theo
        current_state = step_df.copy()

    forecast_all_df = pd.concat(forecast_records, ignore_index=True)

    # Sắp xếp đẹp theo Nhóm trường -> Tên trường -> Năm dự báo
    prob_cols = [c for c in forecast_all_df.columns if c.startswith("Prob_")]
    export_cols = [
        "region",
        "institution",
        "publication_year",
        "total_papers_est",
        "Paper_Growth_YoY (%)",
        "top10_papers_est",
        "core_ai_papers",
        "fwci_adj",
        "intl_collab_rate",
        "Pred_Position_Tier",
        "Confidence_Max_Prob (%)",
    ] + prob_cols

    forecast_all_df = forecast_all_df[export_cols].sort_values(
        by=["region", "institution", "publication_year"]
    )
    forecast_all_df.rename(
        columns={
            "publication_year": "Forecast_Year",
            "total_papers_est": "Pred_Total_Papers",
            "top10_papers_est": "Pred_Top10_Papers",
            "core_ai_papers": "Pred_Core_AI_Papers",
            "fwci_adj": "Pred_FWCI",
            "intl_collab_rate": "Pred_Intl_Collab (%)",
        },
        inplace=True,
    )

    # Lọc bảng 6 trường trọng điểm: 3 Việt Nam (HUST, HCMUT, PTIT) & 3 Thế giới (MIT, Stanford, CMU)
    focus_df = forecast_all_df[forecast_all_df["institution"].isin(FOCAL_ALL)].copy()

    focus_df.to_csv(FORECAST_FOCUS_CSV, index=False, encoding="utf-8-sig")
    forecast_all_df.to_csv(FORECAST_ALL_CSV, index=False, encoding="utf-8-sig")

    display_cols = [
        "institution",
        "Forecast_Year",
        "Pred_Total_Papers",
        "Paper_Growth_YoY (%)",
        "Pred_Top10_Papers",
        "Pred_FWCI",
        "Pred_Position_Tier",
        "Confidence_Max_Prob (%)",
    ]

    print("\n🇻🇳 KẾT QUẢ DỰ BÁO 3 NĂM TỚI (2025–2027) CHO 3 TRƯỜNG ĐH VIỆT NAM (HUST, HCMUT, PTIT):")
    print(focus_df[focus_df["region"] == "Vietnam"][display_cols].to_string(index=False))

    print("\n🌍 KẾT QUẢ DỰ BÁO 3 NĂM TỚI (2025–2027) CHO 3 TRƯỜNG ĐH HÀNG ĐẦU THẾ GIỚI (MIT, STANFORD, CMU):")
    print(focus_df[focus_df["region"] == "Global"][display_cols].to_string(index=False))

    print("\n🎯 PHÂN PHỐI XÁC SUẤT SOFTMAX (%) DỰ ĐOÁN VỊ THẾ NĂM 2027 CỦA 6 TRƯỜNG TRỌNG ĐIỂM:")
    snap_2027 = focus_df[focus_df["Forecast_Year"] == 2027][
        ["institution", "Pred_Total_Papers", "Pred_Top10_Papers", "Pred_Position_Tier"] + prob_cols
    ]
    print(snap_2027.to_string(index=False))

    print(f"\n✅ Đã xuất kết quả dự báo 6 trường trọng điểm ra file : '{FORECAST_FOCUS_CSV}'")
    print(f"✅ Đã xuất kết quả dự báo toàn bộ 20 trường ra file  : '{FORECAST_ALL_CSV}'")


if __name__ == "__main__":
    run_two_tier_ml_pipeline()
