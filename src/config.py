from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "telecom_fault_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TELECOM_COLUMN_NAMES = {
    "Air temperature [K]": "ambient_cabinet_temp_k",
    "Process temperature [K]": "processing_unit_temp_k",
    "Rotational speed [rpm]": "cooling_fan_speed_rpm",
    "Torque [Nm]": "mechanical_load_nm",
    "Tool wear [min]": "component_wear_min",
    "Machine failure": "equipment_failure",
    "TWF": "component_wear_failure",
    "HDF": "thermal_dissipation_failure",
    "PWF": "power_delivery_failure",
    "OSF": "overload_stress_failure",
    "RNF": "random_node_failure",
}

FEATURE_COLUMNS = [
    "ambient_cabinet_temp_k",
    "processing_unit_temp_k",
    "cooling_fan_speed_rpm",
    "mechanical_load_nm",
    "component_wear_min",
    "temp_delta_k",
    "thermal_stress_index",
    "estimated_power_load",
    "wear_load_interaction",
    "cooling_efficiency_index",
    "speed_torque_ratio",
    "cabinet_temp_c",
    "processing_temp_c",
]

CATEGORICAL_COLUMNS = ["Type"]

FAILURE_TYPE_LABELS = {
    "component_wear_failure": "Component Wear",
    "thermal_dissipation_failure": "Thermal Dissipation",
    "power_delivery_failure": "Power Delivery",
    "overload_stress_failure": "Overload Stress",
    "random_node_failure": "Random Node",
    "none": "No Failure",
}
