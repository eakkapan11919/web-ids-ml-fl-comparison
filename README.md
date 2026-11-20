# Two-Stage IDS: Camera-ready v3

Web prototype based on the original bilingual Streamlit app, upgraded for the Camera-ready grouped-split seed-42 models.

## Features
- Thai/English UI and original attack label legend.
- XGBoost or LightGBM two-stage classification. Stage 1 detects Attack/Benign and Stage 2 classifies six attack labels only when Stage 1 flags Attack.
- Binary 0.30 default selected on Validation macro-F1 from candidates 0.3/0.4/0.5/0.6. Security 0.20 / Low false alarm 0.60 / Custom modes are exploratory operating-point choices, not validated optima.
- UnknownAttack 0.60 default is an exploratory low-confidence rejection; held-out unseen-attack detection remains poor.
- Streamed CSV and Parquet input in 5,000-row chunks; maximum uploaded file size 250MB, contingent on Streamlit Cloud memory. Includes all original input rows in downloadable compressed ZIP containing CSV.
- Original label distributions, evaluation metrics/confusion matrix from labeled uploads, translated labels, batch progress, complete results download.
- Research Evidence tab with immutable previously measured CSV tables and figures. These are not recomputed by moving sliders.

## Model and research caveats
- Inputs must include all 77 numeric feature columns, with exactly matching names after whitespace trim. No raw PCAP or live traffic.
- The model/Preprocessing is derived from the grouped split (62,537 train, 15,641 validation, 19,542 test), seed 42. The three-seed means and independent splits are experiment summaries, not the fitted model used in deployment.
- Feature-hash grouping prevents duplicate numeric feature vectors across the three stored splits. Independence at session/day level has not been established.
- Uploaded data metrics with UnknownAttack enabled do not reproduce the 7-class Table II directly.
- Federated Learning IID/Non-IID is a separate MLP experiment; not deployed by this Streamlit app.
- The research evidence is explanatory, not proof of significant superiority or zero-day detection.
- Test on Streamlit Cloud separately before merging into main.

## Run
pip install -r requirements.txt
streamlit run app.py

Use demo_upload_sample.parquet for a quick test.
