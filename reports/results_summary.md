# Results Summary

This report records the generated outputs from the telecom fault prediction pipeline after running the project end to end.

## Training Run

The training pipeline completed successfully using the AI4I 2020 dataset reframed as telecom infrastructure telemetry.

Best recall-oriented model:

```text
Logistic Regression
```

Model comparison:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Avg Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8585 | 0.1810 | 0.8971 | 0.3012 | 0.9402 | 0.4493 |
| Gradient Boosting | 0.9595 | 0.4504 | 0.8676 | 0.5930 | 0.9775 | 0.8650 |
| Random Forest | 0.9780 | 0.6333 | 0.8382 | 0.7215 | 0.9797 | 0.8358 |
| XGBoost | 0.9720 | 0.5588 | 0.8382 | 0.6706 | 0.9808 | 0.8713 |

## Saved Model Artifacts

- `models/telecom_failure_classifier.joblib`
- `models/failure_type_classifier.joblib`
- `models/isolation_forest_anomaly_detector.joblib`
- `models/model_comparison_metrics.csv`

## Saved Result Tables

- `reports/tables/model_comparison_metrics.csv`
- `reports/tables/classification_report_full_dataset.csv`
- `reports/tables/confusion_matrix_full_dataset.csv`
- `reports/tables/failure_type_distribution.csv`
- `reports/tables/sample_prediction.json`

## Saved Graphs

- `reports/figures/correlation_heatmap.png`
- `reports/figures/sensor_distribution_by_failure.png`
- `reports/figures/telecom_sensor_pairplot.png`
- `reports/figures/model_comparison_metrics.png`
- `reports/figures/confusion_matrix_best_model.png`
- `reports/figures/roc_curve_best_model.png`
- `reports/figures/precision_recall_curve_best_model.png`
- `reports/figures/anomaly_scatter.png`
- `reports/figures/shap_summary.png`
- `reports/figures/shap_bar.png`
- `reports/figures/shap_dependence.png`
- `reports/figures/shap_waterfall_high_risk.png`

## Sample Prediction

Input:

```json
{
  "ambient_cabinet_temp_k": 298.1,
  "processing_unit_temp_k": 308.6,
  "cooling_fan_speed_rpm": 1400,
  "mechanical_load_nm": 46.3,
  "component_wear_min": 160,
  "Type": "H"
}
```

Output:

```json
{
  "failure_prediction": 0,
  "failure_probability": 0.2221
}
```
