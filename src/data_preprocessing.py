import pandas as pd

from .config import (
    FAILURE_TYPE_COLUMNS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TELECOM_COLUMN_NAMES,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the AI4I 2020 predictive maintenance dataset."""
    return pd.read_csv(path)


def standardize_telecom_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename sensor fields so analysis reads as telecom infrastructure telemetry."""
    return df.rename(columns=TELECOM_COLUMN_NAMES)


def add_failure_type(df: pd.DataFrame) -> pd.DataFrame:
    """Create a single failure-type label from AI4I's failure indicator columns."""
    renamed_failure_cols = [TELECOM_COLUMN_NAMES[col] for col in FAILURE_TYPE_COLUMNS]

    def resolve_failure(row: pd.Series) -> str:
        for col in renamed_failure_cols:
            if row[col] == 1:
                return col
        return "none"

    df = df.copy()
    df["failure_type"] = df.apply(resolve_failure, axis=1)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleanup while preserving original telemetry values."""
    df = df.copy()
    df = standardize_telecom_columns(df)
    df = add_failure_type(df)
    df = df.drop(columns=["UDI", "Product ID"], errors="ignore")
    return df


def save_processed_data(df: pd.DataFrame, path=PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_and_prepare_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    return clean_dataset(df)
