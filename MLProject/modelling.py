"""MLflow Project entry — train Heart Disease model with autolog."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "heart_disease_preprocessing"
DEFAULT_TRAIN = DATA_DIR / "train.csv"
DEFAULT_TEST = DATA_DIR / "test.csv"


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [col for col in train_df.columns if col != "target"]
    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df["target"],
        test_df["target"],
    )


def train_model(train_path: Path = DEFAULT_TRAIN, test_path: Path = DEFAULT_TEST):
    mlflow.set_experiment("heart_disease_ci")
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    with mlflow.start_run(run_name="ci_random_forest"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print("CI training metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()
    train_model(args.train_path, args.test_path)
