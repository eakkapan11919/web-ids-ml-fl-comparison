import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Web-based IDS Prototype",
    layout="wide"
)


# =========================================================
# Language Dictionary
# =========================================================
TEXT = {
    "en": {
        "title": "Web-based Intrusion Detection System Prototype",
        "subtitle": "Upload network traffic feature data to detect Benign, Attack type, or Unknown Attack.",
        "language": "Language / ภาษา",

        "model_settings": "Model Settings",
        "select_model": "Select Model",
        "model": "Model",
        "stage1": "Stage 1: Benign / Attack Detection",
        "stage2": "Stage 2: Attack Type Classification",
        "model_desc_xgb": "XGBoost: better at reducing missed attacks and improving Infiltration recall.",
        "model_desc_lgbm": "LightGBM: best overall balance in Accuracy, Macro F1, Weighted F1, and lower false alarm.",
        "missing_lgbm": "LightGBM model files were not found. Save the LightGBM .pkl files first to enable this option.",

        "threshold_mode": "Threshold Mode",
        "select_mode": "Select Detection Mode",
        "security_mode": "Security Mode - Detect more attacks",
        "balanced_mode": "Balanced Mode - Default",
        "low_false_alarm_mode": "Low False Alarm Mode",
        "custom_mode": "Custom",

        "security_desc": "High sensitivity: detects more attacks but may increase false alarms.",
        "balanced_desc": "Recommended balanced setting.",
        "low_false_alarm_desc": "Reduces false alarms but may miss more attacks.",
        "custom_desc": "Custom threshold selected by user.",

        "binary_threshold": "Binary Threshold",
        "unknown_threshold": "Unknown Attack Threshold",
        "custom_binary_threshold": "Custom Binary Threshold",

        "upload_file": "Upload CSV or Parquet file",
        "upload_info": "Please upload a CSV or Parquet file to start detection.",
        "cannot_read_file": "Cannot read uploaded file.",
        "model_file_missing": "Model file not found. Please check that all required .pkl files are in the same folder as this Streamlit file.",

        "preview": "Uploaded Data Preview",
        "uploaded_rows": "Uploaded rows",
        "uploaded_columns": "Uploaded columns",

        "detection_summary": "Detection Summary",
        "total_records": "Total Records",
        "benign": "Benign",
        "attack": "Attack",
        "unknown_attack": "Unknown Attack",

        "attack_type_summary": "Attack Type Summary",
        "prediction": "Prediction",
        "prediction_en": "Prediction (English)",
        "prediction_th": "Prediction (Thai)",
        "count": "Count",

        "original_label_dist": "Original Label Distribution",
        "original_label": "Original Label",
        "grouped_true_label_dist": "Grouped True Label Distribution",
        "true_group": "True Group",

        "evaluation_metrics": "Evaluation Metrics",
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",

        "classification_report": "Classification Report",
        "confusion_matrix": "Confusion Matrix",
        "prediction_results": "Prediction Results",
        "rows_to_display": "Number of rows to display",
        "download": "Download Prediction Results as CSV",

        "no_label": "No Label column found. Evaluation metrics cannot be calculated, but prediction results are available.",
        "label_meaning": "Prediction Label Meaning",
    },

    "th": {
        "title": "ต้นแบบระบบตรวจจับการบุกรุกบนเว็บ",
        "subtitle": "อัปโหลดไฟล์คุณลักษณะของทราฟฟิกเครือข่าย เพื่อตรวจจับข้อมูลปกติ ประเภทการโจมตี หรือการโจมตีที่ไม่ทราบประเภท",
        "language": "ภาษา / Language",

        "model_settings": "การตั้งค่าโมเดล",
        "select_model": "เลือกโมเดล",
        "model": "โมเดล",
        "stage1": "ขั้นที่ 1: ตรวจจับข้อมูลปกติ / การโจมตี",
        "stage2": "ขั้นที่ 2: จำแนกประเภทการโจมตี",
        "model_desc_xgb": "XGBoost: เด่นด้านลดการโจมตีที่หลุด และจับ Infiltration ได้ดีกว่า",
        "model_desc_lgbm": "LightGBM: คะแนนรวมดีที่สุด ทั้ง Accuracy, Macro F1, Weighted F1 และลด false alarm ได้ดีกว่า",
        "missing_lgbm": "ยังไม่พบไฟล์โมเดล LightGBM ต้อง save ไฟล์ .pkl ของ LightGBM ก่อน จึงจะเลือกได้",

        "threshold_mode": "โหมดการตรวจจับ",
        "select_mode": "เลือกโหมดการตรวจจับ",
        "security_mode": "โหมดความปลอดภัยสูง - จับการโจมตีมากขึ้น",
        "balanced_mode": "โหมดสมดุล - ค่าแนะนำ",
        "low_false_alarm_mode": "โหมดลดการแจ้งเตือนผิดพลาด",
        "custom_mode": "กำหนดเอง",

        "security_desc": "ไวต่อการตรวจจับสูง: จับการโจมตีได้มากขึ้น แต่อาจแจ้งเตือนผิดพลาดมากขึ้น",
        "balanced_desc": "ค่าแนะนำแบบสมดุล",
        "low_false_alarm_desc": "ลดการแจ้งเตือนผิดพลาด แต่อาจทำให้การโจมตีหลุดมากขึ้น",
        "custom_desc": "ผู้ใช้กำหนดค่า threshold เอง",

        "binary_threshold": "ค่า Binary Threshold",
        "unknown_threshold": "ค่า Unknown Attack Threshold",
        "custom_binary_threshold": "กำหนดค่า Binary Threshold เอง",

        "upload_file": "อัปโหลดไฟล์ CSV หรือ Parquet",
        "upload_info": "กรุณาอัปโหลดไฟล์ CSV หรือ Parquet เพื่อเริ่มตรวจจับ",
        "cannot_read_file": "ไม่สามารถอ่านไฟล์ที่อัปโหลดได้",
        "model_file_missing": "ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าไฟล์ .pkl ที่จำเป็นทั้งหมดอยู่ในโฟลเดอร์เดียวกับไฟล์ Streamlit นี้",

        "preview": "ตัวอย่างข้อมูลที่อัปโหลด",
        "uploaded_rows": "จำนวนแถวที่อัปโหลด",
        "uploaded_columns": "จำนวนคอลัมน์ที่อัปโหลด",

        "detection_summary": "สรุปผลการตรวจจับ",
        "total_records": "ข้อมูลทั้งหมด",
        "benign": "ข้อมูลปกติ",
        "attack": "การโจมตี",
        "unknown_attack": "การโจมตีที่ไม่ทราบประเภท",

        "attack_type_summary": "สรุปประเภทผลการทำนาย",
        "prediction": "ผลการทำนาย",
        "prediction_en": "ผลการทำนาย (อังกฤษ)",
        "prediction_th": "ผลการทำนาย (ไทย)",
        "count": "จำนวน",

        "original_label_dist": "การกระจาย Label เดิม",
        "original_label": "Label เดิม",
        "grouped_true_label_dist": "การกระจายกลุ่ม Label จริง",
        "true_group": "กลุ่มจริง",

        "evaluation_metrics": "ค่าประเมินผล",
        "accuracy": "ความถูกต้อง",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",

        "classification_report": "รายงานการจำแนกประเภท",
        "confusion_matrix": "ตาราง Confusion Matrix",
        "prediction_results": "ผลการทำนายรายแถว",
        "rows_to_display": "จำนวนแถวที่ต้องการแสดง",
        "download": "ดาวน์โหลดผลการทำนายเป็น CSV",

        "no_label": "ไม่พบคอลัมน์ Label จึงไม่สามารถคำนวณค่าประเมินผลได้ แต่ยังสามารถดูผลการทำนายได้",
        "label_meaning": "ความหมายของผลการทำนาย",
    }
}


LABEL_TO_THAI = {
    "Benign": "ข้อมูลปกติ",
    "Bot": "บอต",
    "BruteForce": "เดารหัสผ่าน",
    "DDoS": "โจมตีแบบ DDoS",
    "DoS": "โจมตีแบบ DoS",
    "Infiltration": "การแทรกซึม",
    "WebAttack": "โจมตีเว็บ",
    "UnknownAttack": "การโจมตีที่ไม่ทราบประเภท",
    "OtherAttack": "การโจมตีอื่น ๆ"
}


def translate_label(label, lang):
    if lang == "th":
        return LABEL_TO_THAI.get(label, label)
    return label


def translate_report_index(index_name, lang):
    if lang == "en":
        return index_name

    if index_name in LABEL_TO_THAI:
        return LABEL_TO_THAI[index_name]

    translate_rows = {
        "accuracy": "ความถูกต้องรวม",
        "macro avg": "ค่าเฉลี่ยแบบ Macro",
        "weighted avg": "ค่าเฉลี่ยแบบ Weighted"
    }

    return translate_rows.get(index_name, index_name)


def group_attack_multiclass(label):
    s = str(label).strip().lower()

    if s == "benign":
        return "Benign"
    elif "bot" in s:
        return "Bot"
    elif "ddos" in s or "loic" in s or "hoic" in s:
        return "DDoS"
    elif (
        "dos" in s
        or "hulk" in s
        or "goldeneye" in s
        or "slowloris" in s
        or "slowhttptest" in s
    ):
        return "DoS"
    elif "web" in s or "xss" in s or "sql" in s:
        return "WebAttack"
    elif (
        "ssh-bruteforce" in s
        or "ftp-bruteforce" in s
        or "bruteforce" in s
        or "brute force" in s
    ):
        return "BruteForce"
    elif "infilteration" in s or "infiltration" in s:
        return "Infiltration"
    else:
        return "OtherAttack"


# =========================================================
# Sidebar: Language
# =========================================================
language_choice = st.sidebar.selectbox(
    "ภาษา / Language",
    ["ไทย", "English"]
)

lang = "th" if language_choice == "ไทย" else "en"
T = TEXT[lang]


# =========================================================
# Title
# =========================================================
st.title(T["title"])
st.write(T["subtitle"])


# =========================================================
# Load Models and Artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    base_path = Path(".")

    # Shared preprocessing objects
    scaler = joblib.load(base_path / "multiclass_scaler.pkl")
    median_values = joblib.load(base_path / "multiclass_median_values.pkl")
    feature_columns = joblib.load(base_path / "multiclass_feature_columns.pkl")
    attack_label_encoder = joblib.load(base_path / "attack_type_label_encoder.pkl")

    models = {}

    # Required XGBoost model
    models["Two-stage XGBoost"] = {
        "stage1": joblib.load(base_path / "xgb_stage1_binary_model.pkl"),
        "stage2": joblib.load(base_path / "xgb_stage2_attack_type_model.pkl"),
        "binary_threshold": joblib.load(base_path / "xgb_binary_threshold.pkl"),
        "unknown_threshold": joblib.load(base_path / "xgb_unknown_threshold.pkl"),
        "description_en": TEXT["en"]["model_desc_xgb"],
        "description_th": TEXT["th"]["model_desc_xgb"],
    }

    # Optional LightGBM model
    lgbm_files = [
        base_path / "lgbm_stage1_binary_model.pkl",
        base_path / "lgbm_stage2_attack_type_model.pkl",
        base_path / "lgbm_binary_threshold.pkl",
        base_path / "lgbm_unknown_threshold.pkl",
    ]

    lgbm_available = all(path.exists() for path in lgbm_files)

    if lgbm_available:
        models["Two-stage LightGBM"] = {
            "stage1": joblib.load(base_path / "lgbm_stage1_binary_model.pkl"),
            "stage2": joblib.load(base_path / "lgbm_stage2_attack_type_model.pkl"),
            "binary_threshold": joblib.load(base_path / "lgbm_binary_threshold.pkl"),
            "unknown_threshold": joblib.load(base_path / "lgbm_unknown_threshold.pkl"),
            "description_en": TEXT["en"]["model_desc_lgbm"],
            "description_th": TEXT["th"]["model_desc_lgbm"],
        }

    return (
        models,
        scaler,
        median_values,
        feature_columns,
        attack_label_encoder,
        lgbm_available
    )


try:
    (
        models,
        scaler,
        median_values,
        feature_columns,
        attack_label_encoder,
        lgbm_available
    ) = load_artifacts()

except FileNotFoundError as e:
    st.error(T["model_file_missing"])
    st.code(str(e))
    st.stop()


# =========================================================
# Sidebar: Model Settings
# =========================================================
st.sidebar.header(T["model_settings"])

selected_model_name = st.sidebar.selectbox(
    T["select_model"],
    list(models.keys())
)

selected_model = models[selected_model_name]

stage1_model = selected_model["stage1"]
stage2_model = selected_model["stage2"]

saved_binary_threshold = selected_model["binary_threshold"]
saved_unknown_threshold = selected_model["unknown_threshold"]

if lang == "th":
    st.sidebar.write(f"{T['model']}: **{selected_model_name}**")
    st.sidebar.info(selected_model["description_th"])
else:
    st.sidebar.write(f"{T['model']}: **{selected_model_name}**")
    st.sidebar.info(selected_model["description_en"])

if not lgbm_available:
    st.sidebar.warning(T["missing_lgbm"])

st.sidebar.markdown("---")
st.sidebar.write(T["stage1"])
st.sidebar.write(T["stage2"])


# =========================================================
# Sidebar: Threshold
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header(T["threshold_mode"])

mode_options = {
    T["security_mode"]: "security",
    T["balanced_mode"]: "balanced",
    T["low_false_alarm_mode"]: "low_false_alarm",
    T["custom_mode"]: "custom"
}

selected_mode_text = st.sidebar.selectbox(
    T["select_mode"],
    list(mode_options.keys())
)

selected_mode = mode_options[selected_mode_text]

if selected_mode == "security":
    binary_threshold = 0.3
    mode_description = T["security_desc"]

elif selected_mode == "balanced":
    binary_threshold = 0.4
    mode_description = T["balanced_desc"]

elif selected_mode == "low_false_alarm":
    binary_threshold = 0.6
    mode_description = T["low_false_alarm_desc"]

else:
    binary_threshold = st.sidebar.slider(
        T["custom_binary_threshold"],
        min_value=0.10,
        max_value=0.90,
        value=float(saved_binary_threshold),
        step=0.05
    )
    mode_description = T["custom_desc"]

unknown_threshold = st.sidebar.slider(
    T["unknown_threshold"],
    min_value=0.10,
    max_value=0.90,
    value=float(saved_unknown_threshold),
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.write(f"{T['binary_threshold']}: **{binary_threshold}**")
st.sidebar.write(f"{T['unknown_threshold']}: **{unknown_threshold}**")
st.sidebar.info(mode_description)


# =========================================================
# Sidebar: Label Meaning
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader(T["label_meaning"])

legend_df = pd.DataFrame({
    "English Label": list(LABEL_TO_THAI.keys()),
    "ภาษาไทย": list(LABEL_TO_THAI.values())
})

st.sidebar.dataframe(legend_df, hide_index=True)


# =========================================================
# Upload File
# =========================================================
uploaded_file = st.file_uploader(
    T["upload_file"],
    type=["csv", "parquet"]
)

if uploaded_file is None:
    st.info(T["upload_info"])
    st.stop()


# =========================================================
# Read Uploaded File
# =========================================================
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)

except Exception as e:
    st.error(T["cannot_read_file"])
    st.code(str(e))
    st.stop()


df.columns = df.columns.str.strip()
original_df = df.copy()


# =========================================================
# Uploaded Data Preview
# =========================================================
st.subheader(T["preview"])
st.dataframe(df.head(10))

st.write(f"{T['uploaded_rows']}: **{len(df):,}**")
st.write(f"{T['uploaded_columns']}: **{len(df.columns):,}**")


# =========================================================
# Preprocessing
# =========================================================
drop_cols = [
    "Label",
    "target_binary",
    "target_group",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Timestamp"
]

drop_cols_existing = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols_existing)
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)

# Add missing training columns
for col in feature_columns:
    if col not in X.columns:
        X[col] = np.nan

# Keep same feature order as training
X = X[feature_columns]

# Fill missing values using training median
X = X.fillna(median_values)

# Scale features
X_scaled = scaler.transform(X).astype(np.float32)


# =========================================================
# Stage 1: Benign vs Attack
# =========================================================
stage1_attack_proba = stage1_model.predict_proba(X_scaled)[:, 1]
stage1_pred = (stage1_attack_proba >= binary_threshold).astype(int)


# =========================================================
# Stage 2: Attack Type Classification
# =========================================================
final_output = np.array(["Benign"] * len(X_scaled), dtype=object)
final_confidence = np.zeros(len(X_scaled))

attack_mask = stage1_pred == 1

if attack_mask.sum() > 0:
    attack_type_proba = stage2_model.predict_proba(X_scaled[attack_mask])

    attack_type_pred_num = np.argmax(attack_type_proba, axis=1)
    attack_type_confidence = np.max(attack_type_proba, axis=1)

    attack_type_pred_text = attack_label_encoder.inverse_transform(
        attack_type_pred_num
    )

    attack_type_final = []

    for label, conf in zip(attack_type_pred_text, attack_type_confidence):
        if conf < unknown_threshold:
            attack_type_final.append("UnknownAttack")
        else:
            attack_type_final.append(label)

    final_output[attack_mask] = attack_type_final
    final_confidence[attack_mask] = attack_type_confidence

benign_mask = stage1_pred == 0
final_confidence[benign_mask] = 1 - stage1_attack_proba[benign_mask]


# =========================================================
# Result DataFrame
# =========================================================
result_df = original_df.copy()
result_df["Selected_Model"] = selected_model_name
result_df["Binary_Threshold"] = binary_threshold
result_df["Unknown_Threshold"] = unknown_threshold
result_df["Stage1_Attack_Probability"] = stage1_attack_proba
result_df["Final_Output"] = final_output
result_df["Final_Output_Thai"] = [
    translate_label(label, "th") for label in final_output
]
result_df["Final_Confidence"] = final_confidence


# =========================================================
# Detection Summary
# =========================================================
st.subheader(T["detection_summary"])

total_records = len(result_df)
benign_count = (result_df["Final_Output"] == "Benign").sum()
unknown_count = (result_df["Final_Output"] == "UnknownAttack").sum()
attack_count = total_records - benign_count

col1, col2, col3, col4 = st.columns(4)

col1.metric(T["total_records"], f"{total_records:,}")
col2.metric(T["benign"], f"{benign_count:,}")
col3.metric(T["attack"], f"{attack_count:,}")
col4.metric(T["unknown_attack"], f"{unknown_count:,}")


# =========================================================
# Attack Type Summary
# =========================================================
st.subheader(T["attack_type_summary"])

summary = (
    result_df["Final_Output"]
    .value_counts()
    .rename_axis("Prediction_English")
    .reset_index(name="Count")
)

summary["Prediction_Thai"] = summary["Prediction_English"].apply(
    lambda x: translate_label(x, "th")
)

if lang == "th":
    summary_display = summary[["Prediction_Thai", "Prediction_English", "Count"]].copy()
    summary_display.columns = [
        T["prediction_th"],
        T["prediction_en"],
        T["count"]
    ]

    chart_df = summary_display[[T["prediction_th"], T["count"]]].set_index(
        T["prediction_th"]
    )

else:
    summary_display = summary[["Prediction_English", "Prediction_Thai", "Count"]].copy()
    summary_display.columns = [
        T["prediction_en"],
        T["prediction_th"],
        T["count"]
    ]

    chart_df = summary_display[[T["prediction_en"], T["count"]]].set_index(
        T["prediction_en"]
    )

st.dataframe(summary_display)
st.bar_chart(chart_df)


# =========================================================
# Original Label and Evaluation
# =========================================================
if "Label" in original_df.columns:
    st.subheader(T["original_label_dist"])

    original_label_dist = (
        original_df["Label"]
        .value_counts()
        .rename_axis(T["original_label"])
        .reset_index(name=T["count"])
    )

    st.dataframe(original_label_dist)

    # Convert original labels to grouped labels
    result_df["True_Group"] = original_df["Label"].apply(group_attack_multiclass)
    result_df["True_Group_Thai"] = result_df["True_Group"].apply(
        lambda x: translate_label(x, "th")
    )

    st.subheader(T["grouped_true_label_dist"])

    true_group_dist = (
        result_df["True_Group"]
        .value_counts()
        .rename_axis("True_Group_English")
        .reset_index(name="Count")
    )

    true_group_dist["True_Group_Thai"] = true_group_dist["True_Group_English"].apply(
        lambda x: translate_label(x, "th")
    )

    if lang == "th":
        true_group_display = true_group_dist[
            ["True_Group_Thai", "True_Group_English", "Count"]
        ].copy()

        true_group_display.columns = [
            "กลุ่มจริง (ไทย)",
            "กลุ่มจริง (อังกฤษ)",
            T["count"]
        ]

    else:
        true_group_display = true_group_dist[
            ["True_Group_English", "True_Group_Thai", "Count"]
        ].copy()

        true_group_display.columns = [
            "True Group (English)",
            "True Group (Thai)",
            T["count"]
        ]

    st.dataframe(true_group_display)

    # =====================================================
    # Evaluation Metrics
    # =====================================================
    st.subheader(T["evaluation_metrics"])

    y_true = result_df["True_Group"].values
    y_pred = result_df["Final_Output"].values

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    eval_col1, eval_col2, eval_col3 = st.columns(3)

    eval_col1.metric(T["accuracy"], f"{acc:.4f}")
    eval_col2.metric(T["macro_f1"], f"{macro_f1:.4f}")
    eval_col3.metric(T["weighted_f1"], f"{weighted_f1:.4f}")

    labels_order = [
        "Benign",
        "Bot",
        "BruteForce",
        "DDoS",
        "DoS",
        "Infiltration",
        "WebAttack",
        "UnknownAttack",
        "OtherAttack"
    ]

    labels_used = [
        label for label in labels_order
        if label in set(y_true) or label in set(y_pred)
    ]

    # =====================================================
    # Classification Report
    # =====================================================
    st.subheader(T["classification_report"])

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_used,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    if lang == "th":
        report_df.index = [
            translate_report_index(idx, "th") for idx in report_df.index
        ]

        report_df = report_df.rename(
            columns={
                "precision": "Precision / ความแม่นยำเมื่อทำนายเป็นคลาสนั้น",
                "recall": "Recall / ความสามารถในการจับคลาสนั้น",
                "f1-score": "F1-score",
                "support": "จำนวนข้อมูลจริง"
            }
        )

    st.dataframe(report_df)

    # =====================================================
    # Confusion Matrix
    # =====================================================
    st.subheader(T["confusion_matrix"])

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_used
    )

    if lang == "th":
        row_labels = [
            f"จริง: {translate_label(label, 'th')}" for label in labels_used
        ]

        col_labels = [
            f"ทำนาย: {translate_label(label, 'th')}" for label in labels_used
        ]

    else:
        row_labels = [
            f"True: {label}" for label in labels_used
        ]

        col_labels = [
            f"Pred: {label}" for label in labels_used
        ]

    cm_df = pd.DataFrame(
        cm,
        index=row_labels,
        columns=col_labels
    )

    st.dataframe(cm_df)

else:
    st.warning(T["no_label"])


# =========================================================
# Prediction Results
# =========================================================
st.subheader(T["prediction_results"])

show_rows = st.slider(
    T["rows_to_display"],
    min_value=10,
    max_value=1000,
    value=100,
    step=10
)

display_df = result_df.copy()

if lang == "th":
    preferred_cols = [
        "Selected_Model",
        "Binary_Threshold",
        "Unknown_Threshold"
    ]

    if "Label" in display_df.columns:
        preferred_cols.append("Label")

    if "True_Group_Thai" in display_df.columns:
        preferred_cols.append("True_Group_Thai")

    if "True_Group" in display_df.columns:
        preferred_cols.append("True_Group")

    preferred_cols += [
        "Final_Output_Thai",
        "Final_Output",
        "Final_Confidence",
        "Stage1_Attack_Probability"
    ]

    existing_preferred_cols = [
        col for col in preferred_cols
        if col in display_df.columns
    ]

    other_cols = [
        col for col in display_df.columns
        if col not in existing_preferred_cols
    ]

    display_df = display_df[existing_preferred_cols + other_cols]

    display_df = display_df.rename(
        columns={
            "Selected_Model": "โมเดลที่เลือก",
            "Binary_Threshold": "ค่า Binary Threshold",
            "Unknown_Threshold": "ค่า Unknown Threshold",
            "Label": "Label เดิม",
            "True_Group_Thai": "กลุ่มจริง (ไทย)",
            "True_Group": "กลุ่มจริง (อังกฤษ)",
            "Final_Output_Thai": "ผลการทำนาย (ไทย)",
            "Final_Output": "ผลการทำนาย (อังกฤษ)",
            "Final_Confidence": "ความมั่นใจของผลการทำนาย",
            "Stage1_Attack_Probability": "ความน่าจะเป็นว่าเป็นการโจมตี"
        }
    )

else:
    preferred_cols = [
        "Selected_Model",
        "Binary_Threshold",
        "Unknown_Threshold"
    ]

    if "Label" in display_df.columns:
        preferred_cols.append("Label")

    if "True_Group" in display_df.columns:
        preferred_cols.append("True_Group")

    preferred_cols += [
        "Final_Output",
        "Final_Output_Thai",
        "Final_Confidence",
        "Stage1_Attack_Probability"
    ]

    existing_preferred_cols = [
        col for col in preferred_cols
        if col in display_df.columns
    ]

    other_cols = [
        col for col in display_df.columns
        if col not in existing_preferred_cols
    ]

    display_df = display_df[existing_preferred_cols + other_cols]

st.dataframe(display_df.head(show_rows))


# =========================================================
# Download Results
# =========================================================
csv = result_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label=T["download"],
    data=csv,
    file_name="ids_prediction_results.csv",
    mime="text/csv"
)
