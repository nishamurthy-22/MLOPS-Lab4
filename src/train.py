#Train a penguin species classifier and log it to MLflow.

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_data,
    load_data,
    split_data,
)

TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "penguin-species"

# Types skops needs permission to save; tree-based models (added later) use this one.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "numpy.dtype"]


def build_pipeline(model):
    #Wrap a model with preprocessing, so the saved model accepts raw penguin data.
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(y_true, y_pred):
    #Return accuracy and macro F1 (macro F1 weights all three species equally).
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))

    with mlflow.start_run(run_name="logistic_regression"):
        pipeline.fit(X_train, y_train)
        metrics = evaluate(y_test, pipeline.predict(X_test))

        mlflow.set_tag("dataset", "palmer_penguins")
        mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000, "test_size": 0.2})
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.dropna().head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        print(f"LogisticRegression: {metrics}")


if __name__ == "__main__":
    main()