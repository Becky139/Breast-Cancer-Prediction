import pandas as pd


def load_data():
    data = pd.read_csv("inputs/datasets/raw/data.csv")
    return data


def binary_data():
    data = pd.read_csv("outputs/datasets/cleaned/binary_data.csv", index_col=False)
    data.drop("Unnamed: 0", axis=1, inplace=True)
    return data


def raw_data():
    data = pd.read_csv("inputs/datasets/raw/data.csv", index_col=False)
    return data
