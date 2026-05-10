# AI-Based Predictive Maintenance & Fault Intelligence System for Telecom Infrastructure

An end-to-end machine learning project for predicting telecom infrastructure faults using industrial sensor telemetry. The project reframes the AI4I 2020 predictive maintenance dataset as operational telemetry from telecom infrastructure nodes such as 5G base station cooling units, RAN cabinet hardware, optical transport equipment, edge compute servers, power modules, and signal processing units.

The goal is to show how a reliability engineering team could use machine learning to detect early degradation, prioritize preventive maintenance, and protect network uptime before field equipment reaches a failure state.

## Telecom Relevance

Telecom networks depend on distributed physical infrastructure. A single degraded cooling module, overloaded power unit, or worn cabinet component can affect radio performance, optical transport availability, and service-level reliability. Predictive maintenance helps operations teams move from reactive incident handling to proactive fault prevention.

This project connects predictive analytics with telecom concerns such as:

- RAN cabinet reliability
- 5G site availability
- Network uptime and SLA protection
- Thermal stress in telecom edge equipment
- Failure-type intelligence for preventive dispatch
- Root-cause explainability for operations teams

## Telecom Infrastructure Context

The project is aligned with telecom infrastructure analytics, network assurance, and intelligent operations themes. It uses public predictive maintenance data as a proxy for telecom reliability telemetry and does not depend on proprietary network data.

## Project Structure

```text
telecom_fault_intelligence_ml/
  data/
    raw/
      ai4i2020.csv
    processed/
  models/
  notebooks/
    fault_prediction.ipynb
  reports/
    figures/
  src/
    config.py
    data_preprocessing.py
    features.py
    modeling.py
    visualization.py
    explainability.py
    train.py
    predict.py
  requirements.txt
  README.md
```

## Dataset

Dataset: Predictive Maintenance Dataset (AI4I 2020)

Source: Kaggle dataset file `ai4i2020.csv`

Original fields include air temperature, process temperature, rotational speed, torque, tool wear, binary machine failure, and failure-mode flags. In this project, those signals are interpreted as telemetry from telecom infrastructure equipment.

Example interpretation:

- Air temperature: site ambient temperature near telecom cabinet
- Process temperature: internal equipment or module temperature
- Rotational speed: cooling fan or rotating subsystem speed
- Torque: mechanical or load stress indicator
- Tool wear: component operating age or wear exposure
- Machine failure: telecom equipment failure event
- Failure types: thermal, power, wear, overstress, and random fault modes

## ML Pipeline

1. Load and clean AI4I telemetry
2. Rename features into telecom infrastructure terminology
3. Create engineered reliability features
4. Analyze imbalance in failure events
5. Train supervised fault prediction models with SMOTE
6. Train unsupervised anomaly detection with Isolation Forest
7. Train failure-type classification model
8. Evaluate models with recall, F1-score, ROC-AUC, and PR-AUC
9. Explain model decisions using SHAP
10. Save reusable model artifacts with joblib

## Algorithms Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Isolation Forest
- SMOTE for class imbalance handling
- SHAP for explainable root-cause analysis

## Feature Engineering

Additional telecom-oriented reliability features:

- Temperature delta between internal and ambient equipment conditions
- Estimated power load from speed and torque
- Wear-load interaction
- Thermal stress index
- Cooling efficiency index
- Speed-to-torque ratio

## How To Run

From this folder:

```powershell
cd D:\project\telecom_fault_intelligence_ml
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the notebook:

```powershell
jupyter notebook notebooks\fault_prediction.ipynb
```

Run the full training pipeline:

```powershell
python -m src.train
```

Run a sample prediction:

```powershell
python -m src.predict --ambient-temp-k 298.1 --processing-temp-k 308.6 --fan-speed-rpm 1400 --load-nm 46.3 --wear-min 160 --type H
```

## Outputs

The training pipeline saves:

- `models/telecom_failure_classifier.joblib`
- `models/failure_type_classifier.joblib`
- `models/isolation_forest_anomaly_detector.joblib`
- `models/model_comparison_metrics.csv`
- `data/processed/telecom_fault_dataset_processed.csv`
- figures generated from the notebook under `reports/figures/`

## Business Insights

Predictive fault intelligence can help telecom operations teams:

- Reduce unplanned site outages
- Improve service availability
- Prioritize high-risk maintenance visits
- Detect thermal or load stress earlier
- Protect SLA performance
- Improve infrastructure reliability across distributed 5G and transport assets

## Future Improvements

- Add time-series telemetry windows
- Add survival analysis for remaining useful life estimation
- Add drift monitoring for changing field conditions
- Integrate geospatial tower/site metadata
- Add cost-sensitive learning for critical network assets
- Validate with real telecom operations data when available
