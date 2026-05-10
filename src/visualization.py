import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

from .config import FIGURES_DIR


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (11, 6)
    plt.rcParams["axes.titleweight"] = "bold"


def save_current_figure(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=180, bbox_inches="tight")


def plot_class_distribution(df: pd.DataFrame):
    ax = sns.countplot(data=df, x="equipment_failure", hue="equipment_failure", palette="viridis", legend=False)
    ax.set_title("Telecom Infrastructure Failure Distribution")
    ax.set_xlabel("Equipment Failure")
    ax.set_ylabel("Number of Records")
    return ax


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    ax = sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, linewidths=0.4)
    ax.set_title("Sensor and Fault Correlation Map")
    return ax


def plot_metric_comparison(metrics: pd.DataFrame):
    plot_df = metrics.melt(id_vars="model", value_vars=["precision", "recall", "f1_score", "roc_auc"])
    ax = sns.barplot(data=plot_df, x="model", y="value", hue="variable", palette="mako")
    ax.set_title("Model Comparison Across Reliability Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    return ax


def plot_confusion_matrix(confusion_matrix, labels):
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title("Failure Prediction Confusion Matrix")
    return display


def plot_roc_curve(y_true, y_score, model_name: str):
    display = RocCurveDisplay.from_predictions(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    display.ax_.set_title(f"ROC Curve - {model_name} (AUC={auc_value:.3f})")
    return display


def plot_precision_recall_curve(y_true, y_score, model_name: str):
    display = PrecisionRecallDisplay.from_predictions(y_true, y_score)
    ap_value = average_precision_score(y_true, y_score)
    display.ax_.set_title(f"Precision-Recall Curve - {model_name} (AP={ap_value:.3f})")
    return display
