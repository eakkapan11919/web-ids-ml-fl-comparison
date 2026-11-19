"""Two-stage batch IDS: source of truth for Streamlit and local verification."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BATCH_SIZE=5000
def load_bundle(folder):
    folder=Path(folder)
    prep=joblib.load(folder/"validation_preprocessing.joblib")
    models={
        "Two-stage XGBoost":joblib.load(folder/"xgboost_models.joblib"),
        "Two-stage LightGBM":joblib.load(folder/"lightgbm_models.joblib")
    }
    return prep,models

def make_features(df,prep):
    names=list(prep["columns"])
    df=df.copy()
    df.columns=[str(s).strip() for s in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after trimming spaces")
    absent=[x for x in names if x not in df.columns]
    if absent:
        raise ValueError(f"Missing {len(absent)} required feature columns: "+", ".join(absent[:10]))
    x=df.loc[:,names].apply(pd.to_numeric,errors="coerce")
    x=x.replace([np.inf,-np.inf],np.nan)
    x=prep["imputer"].transform(x)
    return prep["scaler"].transform(x).astype("float32")

def infer_chunk(df,prep,model,binary_threshold=0.3,unknown_threshold=0.6):
    x=make_features(df,prep)
    enc=prep["label_encoder"]
    benign=int(enc.transform(["Benign"])[0])
    known_ids=np.array([i for i in range(len(enc.classes_)) if i!=benign])
    p1=model["stage1"].predict_proba(x)[:,1]
    alarm=p1>=binary_threshold
    labels=np.full(len(x),"Benign",dtype=object)
    confidence=np.maximum(1-p1,0.0)
    if alarm.any():
        p2=model["stage2"].predict_proba(x[alarm])
        if p2.shape[1]!=len(known_ids):
            raise ValueError("Wrong Stage-2 attack label schema")
        conf=p2.max(axis=1)
        category=enc.inverse_transform(known_ids[p2.argmax(axis=1)])
        labels[alarm]=np.where(conf<unknown_threshold,"UnknownAttack",category)
        confidence[alarm]=conf
    return pd.DataFrame({
        "Final_Output":labels,
        "Stage1_Attack_Probability":p1,
        "Final_Confidence":confidence,
        "Stage1_Alarm":alarm
    })
