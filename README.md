###### \# Web IDS ML-FL Comparison

###### 

###### A web-based offline Intrusion Detection System prototype using Machine Learning and Federated Learning evaluation.

###### 

###### \## Deployed Web Application

###### 

###### The deployed Streamlit web prototype is available at:

###### 

###### https://web-ids-ml-fl-comparison-nfx9wkn8tpea6n4rug44oc.streamlit.app

###### 

###### \## GitHub Repository

###### 

###### https://github.com/eakkapan11919/web-ids-ml-fl-comparison

###### 

###### \## Project Title

###### 

###### \*\*A Comparative Evaluation of Machine Learning and Federated Learning for a Web-based Intrusion Detection Prototype\*\*

###### 

###### \## Project Overview

###### 

###### This project presents an academic prototype of a web-based Intrusion Detection System (IDS).  

###### The system is designed for offline/batch prediction, where users upload preprocessed network traffic feature files and receive prediction results from trained machine learning models.

###### 

###### The prototype focuses on comparing traditional Machine Learning models and Federated Learning evaluation results for intrusion detection tasks. The deployed web application uses selected machine learning models for prediction, while Federated Learning is used as part of the research comparison.

###### 

###### \## Main Features

###### 

###### \- Upload CSV or Parquet network traffic feature files

###### \- Support CICFlowMeter-style features similar to the CSE-CIC-IDS2018 dataset

###### \- Select between Two-stage XGBoost and Two-stage LightGBM

###### \- Perform binary detection: Benign vs Attack

###### \- Perform attack type classification

###### \- Output `UnknownAttack` when attack type confidence is uncertain

###### \- Support adjustable detection threshold modes:

###### &#x20; - Security Mode: 0.3

###### &#x20; - Balanced Mode: 0.4

###### &#x20; - Low False Alarm Mode: 0.6

###### \- Unknown Attack Threshold default: 0.6

###### \- Display prediction results

###### \- Display evaluation metrics

###### \- Display confusion matrix

###### \- Download prediction results as CSV

###### 

###### \## Models Used

###### 

###### The project includes the following model components:

###### 

###### \- Random Forest baseline

###### \- Two-stage XGBoost

###### \- Two-stage LightGBM

###### \- Federated Learning evaluation using FedAvg MLP

###### 

###### \## Web Application Model

###### 

###### The deployed Streamlit web application supports:

###### 

###### \- Two-stage XGBoost

###### \- Two-stage LightGBM

###### 

###### Random Forest and Federated Learning are included as part of the research comparison, but they are not the main deployed prediction models in the web prototype.

###### 

###### \## System Workflow

###### 

###### 1\. User uploads a CSV or Parquet file.

###### 2\. The system preprocesses the uploaded data using the saved preprocessing pipeline.

###### 3\. The selected model performs Stage 1 binary detection.

###### 4\. If the traffic is predicted as an attack, Stage 2 classifies the attack type.

###### 5\. If the attack type confidence is below the unknown threshold, the output is labeled as `UnknownAttack`.

###### 6\. The system displays prediction results and evaluation metrics.

###### 7\. The user can download the final prediction results as a CSV file.

###### 

###### \## How to Run Locally

###### 

###### ```bash

###### pip install -r requirements.txt

###### streamlit run app.py

###### ```

###### 

###### \## Input Data Format

###### 

###### The uploaded file must be a CSV or Parquet file containing CICFlowMeter-style network traffic features similar to the CSE-CIC-IDS2018 dataset.

###### 

###### Raw PCAP files are not supported directly.

###### 

###### \## Dataset Notice

###### 

###### The original CSE-CIC-IDS2018 dataset is not included in this repository because of its large file size.

###### 

###### This project uses CICFlowMeter-style network traffic features based on the CSE-CIC-IDS2018 dataset. Users should download the dataset separately and prepare CSV or Parquet files with CICFlowMeter-style features before uploading them to the web application.

###### 

###### \## Dataset Download

###### 

###### The dataset version used in this project was downloaded from Kaggle:

###### 

###### https://www.kaggle.com/datasets/dhoogla/csecicids2018

###### 

###### Official dataset information is also available from the AWS Open Data Registry:

###### 

###### https://registry.opendata.aws/cse-cic-ids2018/



###### \## Limitations

###### 

###### \- This system is an offline/batch prediction prototype.

###### \- The system does not perform real-time packet capture.

###### \- Raw PCAP files are not directly supported.

###### \- The web application depends on the feature format used during model training.

###### \- The prototype is intended for academic evaluation and demonstration.

###### 

###### \## Disclaimer

###### 

###### This project is an academic prototype and should not be used as a production security system without further validation, monitoring, and security testing.

