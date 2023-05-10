import plotly.express as px
import numpy as np
import joblib
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_data


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_data_anaylsis():

    # load data
    df = load_data()

    st.write("### Cancer Predition Features")
    st.info(
        f"* Observation: We are going to ignore id column."
        f" As you can see on the above chart, even though some features left skewed, all of them follows the normal distribution.\n")

    # inspect data
    if st.checkbox("Inspect the data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

 
    # Chart
    model_from_joblib =joblib.load("outputs/datasets/charts/diagnosis.pkl")
