#Tests for the model pipeline, metrics, and best-model selection in train.py.

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import clean_data, load_data, split_data
from src.train import MODELS, build_pipeline, cross_validate_model, evaluate, select_best

SPECIES = {"Adelie", "Chinstrap", "Gentoo"}


#Load and split the data once, and share it across the tests in this file.
@pytest.fixture(scope="module")
def data():
    return split_data(clean_data(load_data()))


#A small, fast pipeline trained once for the tests below.
@pytest.fixture(scope="module")
def fitted_pipeline(data):
    X_train, _, y_train, _ = data
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)


#The pipeline only ever predicts one of the three known species.
def test_pipeline_predicts_known_species(fitted_pipeline, data):
    _, X_test, _, _ = data
    preds = fitted_pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds) <= SPECIES


#A penguin with missing sex still gets the right prediction, because the pipeline fills it in.
def test_pipeline_handles_missing_sex(fitted_pipeline):
    penguin = pd.DataFrame([{
        "bill_length_mm": 50.0, "bill_depth_mm": 15.2, "flipper_length_mm": 218,
        "body_mass_g": 5700, "island": "Biscoe", "sex": None,
    }])
    assert fitted_pipeline.predict(penguin)[0] == "Gentoo"


#An island the model never saw does not crash it, because the encoder ignores unknown values.
@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_pipeline_handles_unseen_island(fitted_pipeline):
    penguin = pd.DataFrame([{
        "bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181,
        "body_mass_g": 3750, "island": "Anvers", "sex": "MALE",
    }])
    assert fitted_pipeline.predict(penguin)[0] in SPECIES


#Test-set metrics have the expected names and valid ranges.
def test_evaluate_returns_valid_metrics(fitted_pipeline, data):
    _, X_test, _, y_test = data
    metrics = evaluate(fitted_pipeline, X_test, y_test)
    assert set(metrics) == {"test_accuracy", "test_f1_macro", "test_log_loss"}
    assert 0 <= metrics["test_accuracy"] <= 1
    assert 0 <= metrics["test_f1_macro"] <= 1
    assert metrics["test_log_loss"] >= 0


#Cross-validation returns all six metrics, and the model clearly beats guessing.
def test_cross_validation_metrics(data):
    X_train, _, y_train, _ = data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    metrics = cross_validate_model(pipeline, X_train, y_train)
    assert len(metrics) == 6
    assert metrics["cv_f1_macro_mean"] > 0.9
    assert metrics["cv_log_loss_mean"] > 0


#Every model in MODELS can be built and trained without errors.
@pytest.mark.parametrize("run_name", list(MODELS))
def test_every_model_trains(run_name, data):
    X_train, X_test, y_train, _ = data
    model_class, params = MODELS[run_name]
    pipeline = build_pipeline(model_class(**params)).fit(X_train, y_train)
    assert set(pipeline.predict(X_test)) <= SPECIES


#The best model is the one with the lowest CV log loss, even if another has a higher F1.
def test_select_best_picks_lowest_log_loss():
    results = {
        "model_a": {"metrics": {"cv_log_loss_mean": 0.20, "cv_f1_macro_mean": 0.99}},
        "model_b": {"metrics": {"cv_log_loss_mean": 0.05, "cv_f1_macro_mean": 0.95}},
        "model_c": {"metrics": {"cv_log_loss_mean": 0.10, "cv_f1_macro_mean": 0.97}},
    }
    assert select_best(results) == "model_b"