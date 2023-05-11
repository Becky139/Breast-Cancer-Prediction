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
    st.write(
        f"One of the main goals of visualizing the data here is to observe which features are most helpful in predicting malignant or benign cancer."
        f"The other is to see general trends that may aid us in model selection and hyper parameter selection."

        f"We will apply 3 techniques that you can use to understand each attribute of the dataset independently."

        f"•	Histograms."

        f"•	Density Plots."

        f"•	Box and Whisker Plots.")
    
    st.image('outputs/datasets/charts/diagnosis.jpeg')
    
    st.write(
        f"Histograms are commonly used to visualize numerical variables. A histogram is similar to a bar graph after the values of the variable are grouped (binned) into a finite number of intervals (bins)."

        f"Histograms group data into bins and provide you a count of the number of observations in each bin. From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. It can also help you see possible outliers."
    )
    
    st.image('outputs/datasets/charts/histogram/_mean_suffix.jpg')
    
    st.image('outputs/datasets/charts/histogram/_se_suffix.jpg')
    
    st.image('outputs/datasets/charts/histogram/_worst_suffix.jpg')
    
    st.info(
        "# Observation"

        f"We can see that perhaps the attributes concavity,and concavity_point may have an exponential distribution ( )."
        f"We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution." 
        f"This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables."
    )
    
    st.image('outputs/datasets/charts/density/_mean_suffix.npy')
    
