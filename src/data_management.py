import streamlit as st
import pandas as pd
import numpy as np


def load_data():
    data = pd.read_csv("inputs/datasets/raw/data.csv")
    return data

