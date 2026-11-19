# Two-Stage IDS Camera-ready v2 (review branch)

A Streamlit batch network-flow classification prototype powered by the **grouped-split seed 42** two-stage XGBoost and LightGBM models.

## Model details

Stage 1 is a binary Benign/Attack classifier. Stage 2 classifies passing traffic into six attack types. It was trained on attack-labeled training flows only. Prediction labels are Benign, Bot, BruteForce, DDoS, DoS, Infiltration, WebAttack, or UnknownAttack (low-confidence rejection).

The model bundle includes training-fitted median imputation, MinMax scaling, the original ordered feature schema and a LabelEncoder. Stage 1 operating thresholds were selected from validation data using macro F1. **The UnknownAttack slider is experimental; it has not demonstrated reliable unseen or zero-day detection.**

## Use

Run `streamlit run app.py`. Upload a CSV or Parquet file with all **77 numerical flow features** matching the CSE-CIC-IDS2018 training schema. Do not upload raw PCAP. If a recognized seven-class Label or target_group column is present, optional per-upload evaluation appears. Download the prediction CSV to analyze each flow.

## Scope and limitations

The prototype is offline batch prediction, not a real-time packet monitor. The displayed metrics on uploaded labeled data may include an extra UnknownAttack prediction, and must not be confused with the seven-class published experimental results. The one-seed model artifacts are separate from the three-seed aggregate evaluation and the federated MLP experiments. The held-out-class UnknownAttack recall remained low. The group-aware split protects exact feature-hash duplicates across splits but does not establish independence at network session level.

See `model_manifest.json` for run provenance.

## Deployment

This is a review-branch candidate. Do not merge to `main` before a deployment smoke test and review of model compatibility.
