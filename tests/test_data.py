#Tests for loading, cleaning, and splitting the penguins dataset.

import pandas as pd

from src.data import FEATURES, NUMERIC_FEATURES, TARGET, clean_data, load_data, split_data


#The raw CSV has the expected size and columns.
def test_load_data_shape_and_columns():
    df = load_data()
    assert df.shape == (344, 7)
    assert set(df.columns) == set(FEATURES + [TARGET])


#Rows with no measurements are dropped, but rows missing only sex are kept.
def test_clean_data_drops_only_rows_without_measurements():
    df = clean_data(load_data())
    assert len(df) == 342
    assert df[NUMERIC_FEATURES].isna().sum().sum() == 0
    assert df["sex"].isna().sum() == 9


#clean_data works on a small hand-made table, not just the real file.
def test_clean_data_on_small_example():
    df = pd.DataFrame({
        "species": ["Adelie", "Gentoo", "Chinstrap"],
        "island": ["Torgersen", "Biscoe", "Dream"],
        "bill_length_mm": [39.1, None, 46.5],
        "bill_depth_mm": [18.7, None, 17.9],
        "flipper_length_mm": [181, None, 192],
        "body_mass_g": [3750, None, 3500],
        "sex": ["MALE", None, None],
    })
    cleaned = clean_data(df)
    assert list(cleaned["species"]) == ["Adelie", "Chinstrap"]


#The split is 80/20 and each species keeps about the same share in train and test.
def test_split_is_stratified():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 273
    assert len(X_test) == 69
    assert list(X_train.columns) == FEATURES

    train_share = y_train.value_counts(normalize=True)
    test_share = y_test.value_counts(normalize=True)
    for species in train_share.index:
        assert abs(train_share[species] - test_share[species]) < 0.02


#The same random_state always gives the same split.
def test_split_is_reproducible():
    df = clean_data(load_data())
    first_test = split_data(df)[1]
    second_test = split_data(df)[1]
    assert first_test.index.equals(second_test.index)