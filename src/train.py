#Train and compare penguin species classifiers, logging each to MLflow,
#then register the best one in the Model Registry.

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_data,
    load_data,
    split_data,
)

TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "penguin-species"
REGISTERED_MODEL_NAME = "penguin-species-classifier"
CHAMPION_ALIAS = "champion"
RANDOM_STATE = 42

#Types skops needs permission to save: numpy.dtype comes from the OneHotEncoder,
#sklearn.tree._tree.Tree from the tree-based models, and the two calibration
#types from the calibrated SVM.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "numpy.dtype",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
]


#SVC no longer outputs probabilities by itself, so wrap it in a calibrator.
def calibrated_svc(C=1.0, kernel="rbf"):
    return CalibratedClassifierCV(SVC(C=C, kernel=kernel), ensemble=False)


#Each entry: run name -> (model class, hyperparameters)
MODELS = {
    "logistic_regression": (LogisticRegression, {"C": 1.0, "max_iter": 1000}),
    "svm_rbf": (calibrated_svc, {"C": 1.0, "kernel": "rbf"}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE}),
    "gradient_boosting": (
        GradientBoostingClassifier,
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": RANDOM_STATE},
    ),
}


#Wrap a model with preprocessing, so the saved model accepts raw penguin data.
def build_pipeline(model):
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


#Score the pipeline on 5 stratified folds of the training data.
def cross_validate_model(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=["accuracy", "f1_macro", "neg_log_loss"])
    return {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        #sklearn reports log loss as a negative number, so flip the sign back
        "cv_log_loss_mean": -scores["test_neg_log_loss"].mean(),
        "cv_log_loss_std": scores["test_neg_log_loss"].std(),
    }


#Score a fitted pipeline on the held-out test set.
def evaluate(pipeline, X, y):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    return {
        "test_accuracy": accuracy_score(y, y_pred),
        "test_f1_macro": f1_score(y, y_pred, average="macro"),
        "test_log_loss": log_loss(y, y_proba, labels=pipeline.classes_),
    }


#Cross-validate, train, test, and log one model as an MLflow run.
#Returns the metrics and the URI of the logged model.
def train_and_log(run_name, model_class, params, X_train, X_test, y_train, y_test):
    pipeline = build_pipeline(model_class(**params))

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"dataset": "palmer_penguins", "model_type": model_class.__name__})
        mlflow.log_params(params)

        cv_metrics = cross_validate_model(pipeline, X_train, y_train)
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)
        metrics = {**cv_metrics, **test_metrics}
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.dropna().head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return {"metrics": metrics, "model_uri": model_info.model_uri}


#Register the model with the lowest CV log loss and point the champion alias at it.
def register_best(results):
    best_name = min(results, key=lambda name: results[name]["metrics"]["cv_log_loss_mean"])
    best = results[best_name]

    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "selected_by", "lowest cv_log_loss_mean")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "run_name", best_name)

    return best_name, version.version


#Load the data, train and log every model in MODELS, then register the best one.
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    results = {}
    for run_name, (model_class, params) in MODELS.items():
        results[run_name] = train_and_log(run_name, model_class, params, X_train, X_test, y_train, y_test)
        m = results[run_name]["metrics"]
        print(
            f"{run_name:20s} cv_f1={m['cv_f1_macro_mean']:.3f}  "
            f"cv_log_loss={m['cv_log_loss_mean']:.3f}  test_f1={m['test_f1_macro']:.3f}"
        )

    best_name, version = register_best(results)
    print(f"\nBest model: {best_name}, registered as {REGISTERED_MODEL_NAME} v{version} (@{CHAMPION_ALIAS})")


if __name__ == "__main__":
    main()