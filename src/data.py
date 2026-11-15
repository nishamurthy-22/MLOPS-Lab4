#Load, clean, and split the Palmer Penguins dataset.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "penguins.csv"

TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    #Read the penguins CSV into a DataFrame.
    return pd.read_csv(path)


def clean_data(df):
    #Drop rows with missing body measurements.
    #Rows missing only `sex` are kept, because the model pipeline fills those in.

    return df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)


def split_data(df, test_size=0.2, random_state=42):
    #Stratified train/test split, so each species keeps its proportion.
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )