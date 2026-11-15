#Tests for the served champion model; they need `mlflow models serve` running on port 5002.

import pytest
import requests

from src.data import FEATURES, clean_data, load_data, split_data
from src.predict import SAMPLE_PENGUINS, SERVING_URL, predict

PING_URL = SERVING_URL.replace("/invocations", "/ping")


#Check whether the model server is reachable.
def server_is_up():
    try:
        return requests.get(PING_URL, timeout=2).status_code == 200
    except requests.RequestException:
        return False


#Skip every test in this file when the server is not running, instead of failing.
pytestmark = pytest.mark.skipif(not server_is_up(), reason="model server is not running on port 5002")


#The server answers its health check.
def test_server_is_healthy():
    assert requests.get(PING_URL, timeout=2).status_code == 200


#One penguin of each species comes back with the right label, including the one missing sex.
def test_api_predicts_sample_penguins():
    assert predict(SAMPLE_PENGUINS) == ["Adelie", "Chinstrap", "Gentoo"]


#The served model is as accurate on the whole test set as it was during training.
def test_api_accuracy_on_test_set():
    _, X_test, _, y_test = split_data(clean_data(load_data()))
    #JSON has no NaN, so turn missing values into None (null).
    rows = X_test.astype(object).where(X_test.notna(), None).values.tolist()

    predictions = predict(rows)

    assert len(predictions) == len(y_test)
    accuracy = sum(p == t for p, t in zip(predictions, y_test)) / len(y_test)
    assert accuracy >= 0.95


#A request missing a required column is rejected with a client error, not a crash.
def test_api_rejects_missing_column():
    columns = [c for c in FEATURES if c != "body_mass_g"]
    payload = {"dataframe_split": {"columns": columns, "data": [[39.1, 18.7, 181, "Torgersen", "MALE"]]}}
    response = requests.post(SERVING_URL, json=payload, timeout=10)
    assert 400 <= response.status_code < 500