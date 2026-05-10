from .config import MODELS_DIR, PROCESSED_DATA_PATH
from .data_preprocessing import load_and_prepare_data, save_processed_data
from .features import build_feature_frame
from .modeling import (
    save_artifacts,
    train_failure_type_classifier,
    train_isolation_forest,
    train_supervised_models,
)


def main() -> None:
    df = load_and_prepare_data()
    df = build_feature_frame(df)
    save_processed_data(df, PROCESSED_DATA_PATH)

    artifacts = train_supervised_models(df)
    best_binary_model = artifacts.models[artifacts.best_model_name]
    fault_type_model = train_failure_type_classifier(df)
    anomaly_model = train_isolation_forest(df)

    save_artifacts(
        binary_model=best_binary_model,
        fault_type_model=fault_type_model,
        anomaly_model=anomaly_model,
        metrics=artifacts.metrics,
        output_dir=MODELS_DIR,
    )

    print("Training complete.")
    print(f"Best model: {artifacts.best_model_name}")
    print(artifacts.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
