import joblib
import shap

from .config import FEATURE_COLUMNS, MODELS_DIR


def load_best_model(path=MODELS_DIR / "telecom_failure_classifier.joblib"):
    return joblib.load(path)


def build_shap_explainer(pipeline, X_background):
    """Create a SHAP explainer for the trained tree-based model when available."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed_background = preprocessor.transform(X_background)
    feature_names = get_transformed_feature_names(preprocessor)
    explainer = shap.Explainer(model, transformed_background, feature_names=feature_names)
    return explainer, preprocessor, feature_names


def get_transformed_feature_names(preprocessor):
    names = list(FEATURE_COLUMNS)
    cat_transformer = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_transformer.get_feature_names_out(["Type"]))
    return names + cat_names
