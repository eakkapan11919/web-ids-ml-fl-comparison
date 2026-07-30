# Web IDS ML-FL Comparison

A web-based offline Intrusion Detection System prototype using Machine Learning and Federated Learning evaluation.

## Project Title

A Comparative Evaluation of Machine Learning and Federated Learning for a Web-based Intrusion Detection Prototype

## Features

- Upload CSV or Parquet network traffic feature files
- Select Two-stage XGBoost or Two-stage LightGBM
- Binary detection: Benign vs Attack
- Attack type classification
- UnknownAttack output for uncertain attack types
- Detection threshold modes:
  - Security Mode: 0.3
  - Balanced Mode: 0.4
  - Low False Alarm Mode: 0.6
- Unknown Attack Threshold default: 0.6
- Evaluation metrics and confusion matrix
- Download prediction results as CSV

## Models

- XGBoost
- LightGBM
- Random Forest baseline
- Federated Learning evaluation using FedAvg MLP

## How to Run

pip install -r requirements.txt

streamlit run app.py

## Input Data Format

The uploaded file must be a CSV or Parquet file with CICFlowMeter-style features similar to the CSE-CIC-IDS2018 dataset.

Raw PCAP files are not supported directly.

## Dataset Notice

The original CSE-CIC-IDS2018 dataset is not included in this repository.

## Disclaimer

This project is an academic prototype and should not be used as a production security system without further validation.
