#Send raw penguin measurements to the served champion model and print its predictions.

import requests

from src.data import FEATURES

SERVING_URL = "http://127.0.0.1:5002/invocations"

#One penguin of each species; the last one is missing sex to show the pipeline fills it in.
SAMPLE_PENGUINS = [
    [39.1, 18.7, 181, 3750, "Torgersen", "MALE"],
    [46.5, 17.9, 192, 3500, "Dream", "FEMALE"],
    [50.0, 15.2, 218, 5700, "Biscoe", None],
]


#Wrap rows in MLflow's dataframe_split format and return the predicted species.
def predict(rows, url=SERVING_URL):
    payload = {"dataframe_split": {"columns": FEATURES, "data": rows}}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["predictions"]


if __name__ == "__main__":
    for row, species in zip(SAMPLE_PENGUINS, predict(SAMPLE_PENGUINS)):
        print(f"{row} -> {species}")