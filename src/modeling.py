from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)


@dataclass
class TrainingArtifacts:
    models: dict[str, Any]
    metrics: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    best_model_name: str
    preprocessor: ColumnTransformer


def build_preprocessor() -> ColumnTransformer:
    numeric_features = FEATURE_COLUMNS
    categorical_features = CATEGORICAL_COLUMNS
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def get_supervised_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=280,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def split_binary_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y = df["equipment_failure"].astype(int)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_supervised_models(df: pd.DataFrame) -> TrainingArtifacts:
    X_train, X_test, y_train, y_test = split_binary_data(df)

    trained_models: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []

    for name, model in get_supervised_models().items():
        preprocessor = build_preprocessor()
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = _predict_proba_or_score(pipeline, X_test)
        trained_models[name] = pipeline
        rows.append(evaluate_binary_classifier(name, y_test, y_pred, y_proba))

    metrics = pd.DataFrame(rows).sort_values(["recall", "f1_score"], ascending=False)
    best_model_name = str(metrics.iloc[0]["model"])

    return TrainingArtifacts(
        models=trained_models,
        metrics=metrics,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        best_model_name=best_model_name,
        preprocessor=trained_models[best_model_name].named_steps["preprocess"],
    )


def _predict_proba_or_score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def evaluate_binary_classifier(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
    }


def train_failure_type_classifier(df: pd.DataFrame) -> Pipeline:
    fault_df = df[df["failure_type"] != "none"].copy()
    X = fault_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y = fault_df["failure_type"]

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def train_isolation_forest(df: pd.DataFrame) -> Pipeline:
    contamination = max(float(df["equipment_failure"].mean()), 0.01)
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    return model


def model_diagnostics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = _predict_proba_or_score(model, X_test)
    return {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "probabilities": y_proba,
        "predictions": y_pred,
    }


def save_artifacts(
    binary_model,
    fault_type_model,
    anomaly_model,
    metrics: pd.DataFrame,
    output_dir=MODELS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_model, output_dir / "telecom_failure_classifier.joblib")
    joblib.dump(fault_type_model, output_dir / "failure_type_classifier.joblib")
    joblib.dump(anomaly_model, output_dir / "isolation_forest_anomaly_detector.joblib")
    metrics.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
