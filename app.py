from pathlib import Path
import json
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Two-Stage IDS | Camera-ready v2", layout="wide")

@st.cache_resource
def load_artifacts():
    meta=json.loads((BASE/"model_manifest.json").read_text(encoding="utf-8"))
    prep=joblib.load(BASE/"validation_preprocessing.joblib")
    models={
        "XGBoost":joblib.load(BASE/"xgboost_models.joblib"),
        "LightGBM":joblib.load(BASE/"lightgbm_models.joblib")
    }
    return meta,prep,models

def preprocess(data, prep):
    cols=list(prep["columns"])
    raw=data.copy()
    raw.columns=[str(x).strip() for x in raw.columns]
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate feature names after trimming whitespace")
    missing=[c for c in cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required training features: "+", ".join(missing[:12]))
    x=raw.loc[:,cols].copy()
    for c in cols:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.replace([np.inf,-np.inf],np.nan)
    all_nan=[c for c in cols if x[c].isna().all()]
    if all_nan:
        raise ValueError("Features contain no usable values: "+", ".join(all_nan[:12]))
    x=prep["imputer"].transform(x)
    return prep["scaler"].transform(x).astype(np.float32)

def predict(x,model,prep,binary_threshold,unknown_threshold):
    enc=prep["label_encoder"]
    benign=int(enc.transform(["Benign"])[0])
    attack_ids=np.array([i for i in range(len(enc.classes_)) if i!=benign])
    p_attack=model["stage1"].predict_proba(x)[:,1]
    p_type=model["stage2"].predict_proba(x)
    # The stage-2 model only produces six ATTACK labels in sorted encoder order.
    if p_type.shape[1]!=len(attack_ids):
        raise ValueError("Stage-2 class dimension does not match the saved label schema")
    attack=p_attack>=binary_threshold
    category=np.full(len(x),"Benign",dtype=object)
    confidence=p_type.max(axis=1)
    if attack.any():
        attack_labels=enc.inverse_transform(attack_ids[p_type[attack].argmax(axis=1)])
        category[attack]=np.where(confidence[attack]<unknown_threshold,"UnknownAttack",attack_labels)
    return pd.DataFrame({
        "Final_Output":category,
        "Stage1_Attack_Probability":p_attack,
        "Stage2_Max_Confidence":confidence,
        "Is_Attack_Alarm":attack
    })

st.title("Two-Stage Machine Learning Intrusion Detection System")
st.caption("Camera-ready experiment version | Offline network-flow CSV / Parquet input, not live traffic or PCAP")
st.info("UnknownAttack represents low classification confidence. It does not prove that an unseen or zero-day attack has been detected.")

try:
    manifest,prep,models=load_artifacts()
except Exception as exc:
    st.error("Could not load compatible Camera-ready model artifacts: "+str(exc))
    st.stop()

with st.sidebar:
    st.header("Model settings")
    selected=st.selectbox("Model",list(models.keys()))
    default=float(manifest["thresholds"][selected]["binary"])
    bthr=st.slider("Stage 1 attack threshold",0.05,0.95,default,0.05)
    uthr=st.slider("Stage 2 UnknownAttack threshold (experimental)",0.0,1.0,0.6,0.05)
    st.caption("The Stage 1 default was chosen from validation macro F1. The UnknownAttack threshold is exploratory and is not a validated zero-day detection setting.")

st.write("Model version:",manifest["version"])
st.write("Expected numeric features:",len(prep["columns"]))
uploaded=st.file_uploader("Upload extracted network-flow features",type=["csv","parquet"])
if uploaded:
    try:
        if uploaded.name.lower().endswith(".parquet"):
            df=pd.read_parquet(uploaded)
        else:
            df=pd.read_csv(uploaded)
        if df.empty:
            raise ValueError("The uploaded file has no rows")
        if len(df)>100000:
            raise ValueError("Please upload at most 100,000 rows per batch")
        x=preprocess(df,prep)
        predictions=predict(x,models[selected],prep,bthr,uthr)
        full=pd.concat([df.reset_index(drop=True),predictions],axis=1)
        benign=int((predictions.Final_Output=="Benign").sum())
        unknown=int((predictions.Final_Output=="UnknownAttack").sum())
        known_attack=len(predictions)-benign-unknown
        a,b,c,d=st.columns(4)
        a.metric("Total flows",len(df))
        b.metric("Benign",benign)
        c.metric("Known-type attack alarms",known_attack)
        d.metric("UnknownAttack alarms",unknown)
        st.caption("Benign, known-type attack alarms, and UnknownAttack alarms are disjoint. Their total equals the number of flows.")
        st.subheader("Prediction counts")
        st.dataframe(predictions.Final_Output.value_counts().rename_axis("Prediction").reset_index(name="Count"),use_container_width=True)
        st.subheader("Prediction preview")
        st.dataframe(full.head(200),use_container_width=True)
        lab=next((col for col in ["target_group","Label"] if col in df.columns),None)
        if lab is not None:
            real=df[lab].astype(str).str.strip()
            valid=set(prep["label_encoder"].classes_)
            if set(real.unique()).issubset(valid):
                st.subheader("Evaluation on labeled upload")
                st.write("Accuracy:",round(accuracy_score(real,predictions.Final_Output),4))
                st.dataframe(pd.DataFrame(classification_report(real,predictions.Final_Output,output_dict=True,zero_division=0)).T,use_container_width=True)
                cm_labels=sorted(set(real)|set(predictions.Final_Output))
                st.dataframe(pd.DataFrame(confusion_matrix(real,predictions.Final_Output,labels=cm_labels),index=cm_labels,columns=cm_labels),use_container_width=True)
                if unknown:
                    st.warning("This metric includes UnknownAttack as an additional predicted label, so it is not directly comparable with the seven-class Table II.")
            else:
                st.caption("Provided labels do not match the stored 7-class schema; metrics skipped.")
        st.download_button("Download predictions CSV",data=full.to_csv(index=False).encode("utf-8-sig"),file_name=f"{selected.lower()}_predictions.csv",mime="text/csv")
    except Exception as exc:
        st.error("Input or prediction error: "+str(exc))
else:
    st.caption("The upload must include the full set of 77 precomputed numeric flow features. Dataset rows with missing feature columns are rejected rather than silently imputed.")
