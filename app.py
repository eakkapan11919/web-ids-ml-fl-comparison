import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io
import zipfile
import tempfile
from collections import Counter
import json
import pyarrow.parquet as pq

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



# Camera-ready V3, retaining the original bilingual UI and label mapping.
from ids_pipeline import load_bundle, infer_chunk, BATCH_SIZE
import pyarrow.parquet as pq

BASE=Path(__file__).resolve().parent
MANIFEST=json.loads((BASE/"model_manifest.json").read_text(encoding="utf-8"))

@st.cache_resource
def get_models():
    return load_bundle(BASE)

try:
    prep, models = get_models()
except Exception as exc:
    st.error(f"Model loading error: {exc}")
    st.stop()

choice=st.sidebar.selectbox("ภาษา / Language",["ไทย","English"])
lang="th" if choice=="ไทย" else "en"
T=TEXT[lang]
st.title(T["title"])
st.write(T["subtitle"])
st.caption("Camera-ready grouped seed 42 | Offline CSV / Parquet network-flow analysis")
st.info("UnknownAttack หมายถึงโมเดลไม่มั่นใจในประเภทการโจมตี ไม่ได้พิสูจน์ว่าเป็น Zero-day | UnknownAttack means low confidence, not verified zero-day detection.")

st.sidebar.header(T["model_settings"])
chosen=st.sidebar.selectbox(T["select_model"],list(models.keys()))
st.sidebar.write(f'{T["model"]}: **{chosen}**')
if lang=="th":
    st.sidebar.info("โมเดลทั้งสองเปรียบเทียบบนข้อมูลเดียวกัน โดยไม่อ้างว่าโมเดลใดดีที่สุดทุกตัวชี้วัด")
else:
    st.sidebar.info("Both models use the same dataset split; neither is superior on every metric.")
st.sidebar.write(T["stage1"])
st.sidebar.write(T["stage2"])
st.sidebar.markdown("---")
st.sidebar.header(T["threshold_mode"])
modes={
    T["security_mode"]:"security",
    T["balanced_mode"]:"balanced",
    T["low_false_alarm_mode"]:"low_false_alarm",
    T["custom_mode"]:"custom",
}
chosen_mode=st.sidebar.selectbox(T["select_mode"],list(modes))
mode=modes[chosen_mode]
selected_default=float(MANIFEST["thresholds"]["XGBoost" if "XGBoost" in chosen else "LightGBM"]["binary"])
if mode=="security":
    binary_threshold=0.2
elif mode=="balanced":
    binary_threshold=selected_default
elif mode=="low_false_alarm":
    binary_threshold=0.6
else:
    binary_threshold=st.sidebar.slider(T["custom_binary_threshold"],0.05,0.95,float(selected_default),0.05)
unknown_threshold=st.sidebar.slider(T["unknown_threshold"],0.0,1.0,0.6,0.05)
st.sidebar.write(f'{T["binary_threshold"]}: **{binary_threshold:.2f}**')
st.sidebar.write(f'{T["unknown_threshold"]}: **{unknown_threshold:.2f}**')
st.sidebar.caption("Binary 0.30: selected using validation macro F1 (seed 42). Other modes: demonstrations, not validation-optimal settings. Unknown 0.60: exploratory.")
st.sidebar.markdown("---")
st.sidebar.subheader(T["label_meaning"])
legend=pd.DataFrame({"English Label":list(LABEL_TO_THAI),"ภาษาไทย":list(LABEL_TO_THAI.values())})
st.sidebar.dataframe(legend,hide_index=True)

tab_detect,tab_evidence=st.tabs(["ตรวจจับ / Detection","หลักฐานงานวิจัย / Research Evidence"])

with tab_evidence:
    st.subheader("วิธีเลือก Threshold / How thresholds were selected")
    st.write("ข้อมูล Train 62,537 / Validation 15,641 / Test 19,542 ตัวอย่าง แยกกลุ่ม Feature Hash เพื่อป้องกัน Feature Vector ที่เหมือนกันข้ามชุด")
    st.write("Binary Threshold ของ Two-stage XGBoost และ LightGBM เลือก 0.30 จาก Macro F1 บน Validation โดยเปรียบเทียบ 0.3, 0.4, 0.5, 0.6 ไม่ได้เลือกจาก Test")
    st.warning("UnknownAttack Threshold 0.60 เป็นค่าใช้สาธิต ไม่ได้ผ่านการยืนยันว่าเหมาะสมที่สุดสำหรับ Unseen Attacks. คะแนน LOCO ของการตรวจ Unknown ยังต่ำ")
    st.caption("The research tables below are fixed results from saved experiments. The upload settings do not change the tables or figures.")
    base=BASE/"research_evidence"
    st.subheader("Two-stage versus single-stage: model seeds 42, 43, 44")
    st.dataframe(pd.read_csv(base/"TABLE_II_REVISED_3SEEDS.csv").round(4),hide_index=True,width="stretch")
    st.subheader("Three independent grouped data splits")
    st.dataframe(pd.read_csv(base/"SPLIT_MEAN_SD_PERCENT.csv").round(4),hide_index=True,width="stretch")
    st.subheader("UnknownAttack: withheld WebAttack and Infiltration")
    st.dataframe(pd.read_csv(base/"UNKNOWN_VALIDATED_MAXPROB_VS_ENTROPY.csv").round(4),hide_index=True,width="stretch")
    st.caption("The rejection cutoffs above use the 99th percentile of known-class uncertainty on validation. Even under this criterion, held-out Infiltration detection is very poor. This does not demonstrate general zero-day detection.")
    st.image(str(base/"FIG_NEW_UNKNOWN_VALIDATED.png"),caption="Unknown rejection results with validation-based cutoffs",width="stretch")
    st.subheader("Centralized MLP versus FedAvg IID/Non-IID")
    st.dataframe(pd.read_csv(base/"CENTRALIZED_VS_FEDAVG_COMPARISON.csv").round(4),hide_index=True,width="stretch")
    st.caption("FedAvg is a separate MLP experiment, not a model used for webpage predictions. Centralized and FL optimization procedures differ.")
    st.subheader("Stage 1 ROC at multiple thresholds")
    st.image(str(base/"FIG2_ROC_GROUPED_VALIDATED_SEED42.png"),width="stretch")
    st.caption("วงกลม = Threshold 0.30 ที่เลือกจาก Validation; สี่เหลี่ยม = Threshold 0.40 ใช้เปรียบเทียบ. จุดแต่ละโมเดลอยู่ใกล้กันตามค่าจริง")
    points=pd.read_csv(base/"FIG2_OPERATING_POINTS.csv")
    st.dataframe(points[["model","stage1_binary_threshold","stage1_FPR","stage1_TPR","roc_auc"]].round(4),hide_index=True,width="stretch")
    st.caption("ROC แสดง Stage 1 (Binary Attack/Benign) บน Test ชุดเดียวกัน ไม่ใช่ ROC ของ Stage 2 หรือ UnknownAttack")
    st.caption("Dataset split grouping by identical numerical feature hashes does not establish day/session independence. Results have not demonstrated statistically significant superiority.")

def iter_upload(file):
    file.seek(0)
    if file.name.lower().endswith(".parquet"):
        parquet=pq.ParquetFile(file)
        total=parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
            yield batch.to_pandas(), total
    else:
        for frame in pd.read_csv(file,chunksize=BATCH_SIZE,low_memory=False):
            yield frame, None

with tab_detect:
    st.caption("Maximum upload is 250 MB per file. Large files are read in batches of 5,000 flows. Original multi-hundred-thousand-row Parquet files are supported if server memory permits.")
    uploaded=st.file_uploader(T["upload_file"],type=["csv","parquet"],key="big_upload")
    if uploaded is None:
        st.info(T["upload_info"])
        with st.expander("รูปแบบไฟล์ที่รองรับ / Required input schema"):
            st.write("ต้องมีคอลัมน์ Numeric Network-flow Features 77 ตัวตาม CSE-CIC-IDS2018")
            st.write("คอลัมน์ Label หรือ target_group ไม่จำเป็นต่อการทำนาย แต่จำเป็นสำหรับคำนวณ Accuracy / F1")
            st.write("ไฟล์ CSV / Parquet ขนาดสูงสุด 250 MB ต่อไฟล์; ประมวลผลทีละ 5,000 แถว และดาวน์โหลดผลทั้งไฟล์เป็น ZIP")
            st.write("Features ที่ต้องมี:", ", ".join(prep["columns"]))
    else:
        try:
            # Build one compressed prediction CSV incrementally, avoiding retaining full datasets.
            compressed=tempfile.TemporaryFile(mode="w+b")
            label_pairs=Counter()
            originals=Counter()
            groups=Counter()
            results=Counter()
            total=0
            preview=None
            first=True
            bar=st.progress(0,text="Processing flow batches")
            selected=models[chosen]
            with zipfile.ZipFile(compressed,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
                with z.open("ids_predictions.csv","w",force_zip64=True) as output:
                    for frame,expected in iter_upload(uploaded):
                        if frame.empty:
                            continue
                        frame.columns=[str(c).strip() for c in frame.columns]
                        preds=infer_chunk(frame,prep,selected,binary_threshold,unknown_threshold)
                        preds["Selected_Model"]=chosen
                        preds["Binary_Threshold"]=binary_threshold
                        preds["Unknown_Threshold"]=unknown_threshold
                        preds["Final_Output_Thai"]=preds["Final_Output"].map(LABEL_TO_THAI).fillna(preds["Final_Output"])
                        # Keep source columns for full reproducibility, like the original webpage.
                        combined=pd.concat([frame.reset_index(drop=True),preds],axis=1)
                        if preview is None:
                            preview=combined.head(10)
                        results.update(preds["Final_Output"].value_counts().to_dict())
                        if "target_group" in frame.columns:
                            truth=frame["target_group"].astype(str).str.strip()
                        elif "Label" in frame.columns:
                            originals.update(frame["Label"].astype(str).value_counts().to_dict())
                            truth=frame["Label"].map(group_attack_multiclass)
                        else:
                            truth=None
                        if truth is not None:
                            groups.update(truth.value_counts().to_dict())
                            label_pairs.update(zip(truth,preds["Final_Output"]))
                        out_bytes=combined.to_csv(index=False,header=first).encode("utf-8-sig" if first else "utf-8")
                        output.write(out_bytes)
                        first=False
                        total+=len(frame)
                        if expected:
                            bar.progress(min(1.0,total/expected),text=f"Processed {total:,}/{expected:,} flows")
                        else:
                            bar.progress(min(.99,(total%100000)/100000),text=f"Processed {total:,} flows")
            bar.progress(1.0,text=f"Completed {total:,} flows")
            st.success(f"ประมวลผลสำเร็จ {total:,} แถว / Processed {total:,} rows")
            st.caption(f"ใช้โมเดล {chosen}; Binary threshold {binary_threshold:.2f}; Unknown threshold {unknown_threshold:.2f}; อินพุตมี {len(prep['columns'])} คุณลักษณะที่ต้องการครบ")
            if not total:
                st.error("No rows to process")
            else:
                st.subheader(T["preview"])
                st.dataframe(preview,width="stretch")
                st.write(f'{T["uploaded_rows"]}: **{total:,}**')
                st.write(f'{T["uploaded_columns"]}: **{len(preview.columns):,}** (including prediction outputs)')
                benign=results["Benign"]
                unknown=results["UnknownAttack"]
                known=total-benign-unknown
                st.subheader(T["detection_summary"])
                c1,c2,c3,c4=st.columns(4)
                c1.metric(T["total_records"],f"{total:,}")
                c2.metric(T["benign"],f"{benign:,}")
                c3.metric("Known attack alarms",f"{known:,}")
                c4.metric(T["unknown_attack"],f"{unknown:,}")
                st.caption("Total = Benign + Known-type alarms + UnknownAttack alarms. UnknownAttack is not counted twice.")
                summary=pd.DataFrame([{"Prediction_English":k,"Prediction_Thai":translate_label(k,"th"),"Count":v} for k,v in results.most_common()])
                st.subheader(T["attack_type_summary"])
                st.dataframe(summary,width="stretch",hide_index=True)
                st.bar_chart(summary.set_index("Prediction_English")["Count"])
                if originals:
                    st.subheader(T["original_label_dist"])
                    st.dataframe(pd.DataFrame(originals.items(),columns=["Original Label","Count"]),hide_index=True)
                if groups:
                    st.subheader(T["grouped_true_label_dist"])
                    st.dataframe(pd.DataFrame(groups.items(),columns=["True Group","Count"]),hide_index=True)
                if label_pairs:
                    # Aggregated metrics exactly match evaluation on all rows without keeping all rows in RAM.
                    actual_labels=set(groups)
                    known_truth=set(prep["label_encoder"].classes_)
                    if actual_labels.issubset(known_truth):
                        y_true=[];y_pred=[];weights=[]
                        for (a,b),n in label_pairs.items():
                            y_true.append(a);y_pred.append(b);weights.append(n)
                        st.subheader(T["evaluation_metrics"])
                        a=accuracy_score(y_true,y_pred,sample_weight=weights)
                        macro=f1_score(y_true,y_pred,average="macro",sample_weight=weights,zero_division=0)
                        weighted=f1_score(y_true,y_pred,average="weighted",sample_weight=weights,zero_division=0)
                        c1,c2,c3=st.columns(3)
                        c1.metric(T["accuracy"],f"{a:.4f}")
                        c2.metric(T["macro_f1"],f"{macro:.4f}")
                        c3.metric(T["weighted_f1"],f"{weighted:.4f}")
                        used=sorted(set(y_true)|set(y_pred))
                        report=pd.DataFrame(classification_report(y_true,y_pred,labels=used,sample_weight=weights,output_dict=True,zero_division=0)).T
                        st.subheader(T["classification_report"])
                        st.dataframe(report,width="stretch")
                        st.subheader(T["confusion_matrix"])
                        matrix=confusion_matrix(y_true,y_pred,labels=used,sample_weight=weights)
                        st.dataframe(pd.DataFrame(matrix,index=used,columns=used),width="stretch")
                        st.caption("With UnknownAttack enabled, these upload metrics are NOT directly comparable with the seven-class Table II, which excludes unknown rejection.")
                    else:
                        st.warning("Some uploaded true labels are not within the seven training classes, so supervised metrics are omitted.")
                else:
                    st.info(T["no_label"])
                st.subheader(T["prediction_results"])
                st.dataframe(preview.head(10),width="stretch")
                # Streamlit download_button requires binary bytes or an accepted IO object.
                # tempfile.TemporaryFile is not an accepted data type, so read the ZIP
                # into bytes after streaming the CSV to disk; close the temp file.
                compressed.seek(0)
                zip_bytes=compressed.read()
                compressed.close()
                st.download_button(
                    "Download complete predictions (ZIP with CSV)",
                    data=zip_bytes,
                    file_name="ids_predictions.zip",
                    mime="application/zip",
                    on_click="ignore",
                )
                st.caption("ZIP includes the original columns and prediction outputs for every uploaded row; the table above displays only a preview to reduce memory.")
        except Exception as error:
            st.error(f"Input or prediction error: {error}")
