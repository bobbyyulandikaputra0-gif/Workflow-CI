"""modelling.py — Training model untuk MLflow Project (Workflow CI).

Dijalankan otomatis oleh GitHub Actions melalui `mlflow run` setiap kali trigger
terpantik. Tracking memakai file store lokal (./mlruns) karena berjalan headless
di runner CI, dan model disimpan sebagai artefak MLflow.
"""
import argparse
import os

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main(data_dir: str, n_estimators: int, max_depth: int):
    train = pd.read_csv(os.path.join(BASE_DIR, data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(BASE_DIR, data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=["y"]), train["y"]
    X_test, y_test = test.drop(columns=["y"]), test["y"]

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="ci_randomforest"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))

        print(f"CI training selesai | accuracy={accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="bank_marketing_preprocessing")
    # type=float karena MLflow Project mengoper parameter numerik sebagai float
    parser.add_argument("--n_estimators", type=float, default=200)
    parser.add_argument("--max_depth", type=float, default=20)
    args = parser.parse_args()
    main(args.data_dir, int(args.n_estimators), int(args.max_depth))
