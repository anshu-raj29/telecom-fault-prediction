import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import build_feature_frame


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "telecom_failure_classifier.joblib"
FAULT_TYPE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "failure_type_classifier.joblib"
ANOMALY_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest_anomaly_detector.joblib"


def predict_single_record(record: dict) -> dict:
    binary_model = joblib.load(MODEL_PATH)
    fault_type_model = joblib.load(FAULT_TYPE_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)

    df = pd.DataFrame([record])
    df = build_feature_frame(df)

    failure_probability = float(binary_model.predict_proba(df)[0, 1])
    failure_prediction = int(binary_model.predict(df)[0])
    fault_type = str(fault_type_model.predict(df)[0]) if failure_prediction else "none"
    anomaly_flag = int(anomaly_model.predict(df)[0] == -1)
    anomaly_score = float(-anomaly_model.decision_function(df)[0])

    return {
        "failure_prediction": failure_prediction,
        "failure_probability": round(failure_probability, 4),
        "predicted_failure_type": fault_type,
        "anomaly_flag": anomaly_flag,
        "anomaly_score": round(anomaly_score, 4),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict telecom infrastructure failure risk.")
    parser.add_argument("--ambient-temp-k", type=float, required=True)
    parser.add_argument("--processing-temp-k", type=float, required=True)
    parser.add_argument("--fan-speed-rpm", type=float, required=True)
    parser.add_argument("--load-nm", type=float, required=True)
    parser.add_argument("--wear-min", type=float, required=True)
    parser.add_argument("--type", type=str, default="M", choices=["L", "M", "H"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = {
        "ambient_cabinet_temp_k": args.ambient_temp_k,
        "processing_unit_temp_k": args.processing_temp_k,
        "cooling_fan_speed_rpm": args.fan_speed_rpm,
        "mechanical_load_nm": args.load_nm,
        "component_wear_min": args.wear_min,
        "Type": args.type,
    }
    print(predict_single_record(record))


if __name__ == "__main__":
    main()
