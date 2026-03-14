# models.py
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

def train_and_evaluate(X, y, cont_iso, cont_lof, cont_svm, save_dir="saved_models"):
    """
    Train models, evaluate, and save them using joblib.
    """
    os.makedirs(save_dir, exist_ok=True)

    models = {
        'Isolation Forest': IsolationForest(
            contamination=cont_iso,
            random_state=42,
            n_estimators=100
        ),
        'Local Outlier Factor': LocalOutlierFactor(
            n_neighbors=20,
            contamination=cont_lof,
            novelty=True
        ),
        'One-Class SVM': OneClassSVM(
            kernel='rbf',
            nu=cont_svm
        )
    }

    results = {}
    saved_paths = {}

    for name, model in models.items():
        print(f"\nTraining {name} with tuned param = {model.contamination if hasattr(model,'contamination') else model.nu:.5f}")

        if name == 'Local Outlier Factor':
            model.fit(X)
            preds = model.predict(X)
        else:
            preds = model.fit_predict(X)

        anomalies = (preds == -1).astype(int)
        results[name] = anomalies
        print(f"\n{name} Results:")
        print(classification_report(y, anomalies, digits=4, zero_division=0))
        cm = confusion_matrix(y, anomalies)
        print("Confusion Matrix:\n", cm)
        print(f"ROC-AUC: {roc_auc_score(y, anomalies):.4f}")
        print(f"PR-AUC : {average_precision_score(y, anomalies):.4f}")
        safe_name = name.lower().replace(' ', '_')
        model_path = os.path.join(save_dir, f"{safe_name}_model.joblib")
        joblib.dump(model, model_path)
        saved_paths[name] = model_path
        print(f"Model saved → {model_path}")
    summary_path = os.path.join(save_dir, "model_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Saved Models:\n")
        for name, path in saved_paths.items():
            f.write(f"{name}: {path}\n")
        f.write("\nNote: Use joblib.load() to reload models for inference.\n")

    print(f"\nAll models saved in folder: {save_dir}/")
    print("Summary written to: model_summary.txt")

    return results