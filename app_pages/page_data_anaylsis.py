import plotly.express as px
import numpy as np
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

    st.pyplot()
    st.write(
        f"* Observation: We can relatively say that the dataset is balanced.")

    # Correlation Study Summary
    st.plotly_chart()
    st.write(
        f"Observation: There are a lot of features that have strong correlation to each other."
        f"We might need to a type of feature selection method. "
        f" First we need to check the baseline then we should decide if we need or not. "
        f"The most correlated variable are: "
    )

    # Text based on "02 - " notebook - "Conclusions and Next steps" section
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* A churned customer typically has a month-to-month contract \n"
        f"* A churned customer typically has fibre optic. \n"
        f"* A churned customer typically doesn't have tech support. \n"
        f"* A churned customer doesn't have online security. \n"
        f"* A churned customer typically has low tenure levels. \n"
    )

