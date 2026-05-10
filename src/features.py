import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reliability-focused features from sensor telemetry."""
    df = df.copy()

    df["temp_delta_k"] = df["processing_unit_temp_k"] - df["ambient_cabinet_temp_k"]
    df["cabinet_temp_c"] = df["ambient_cabinet_temp_k"] - 273.15
    df["processing_temp_c"] = df["processing_unit_temp_k"] - 273.15

    df["estimated_power_load"] = (
        df["mechanical_load_nm"] * df["cooling_fan_speed_rpm"] / 9550
    )
    df["wear_load_interaction"] = df["component_wear_min"] * df["mechanical_load_nm"]
    df["thermal_stress_index"] = df["temp_delta_k"] * df["mechanical_load_nm"]
    df["cooling_efficiency_index"] = df["cooling_fan_speed_rpm"] / (
        df["processing_unit_temp_k"] + 1e-6
    )
    df["speed_torque_ratio"] = df["cooling_fan_speed_rpm"] / (
        df["mechanical_load_nm"] + 1e-6
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full model-ready frame with engineered features."""
    return add_engineered_features(df)
