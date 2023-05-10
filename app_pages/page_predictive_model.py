import joblib
import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_data


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_predictive_model():

   # PCA Plot
    st.write("### Feature decomposition using Principal Component Analysis (PCA)")
    st.info(
        f"* From the pair plot in **Part_2**, lot of feature pairs divide nicely the data to a similar extent, therefore, it makes sense to use one of the dimensionality reduction methods to try to use as many features as possible and maintian as much information as possible when working with only 2 dimensions. I will use PCA.."
    )
    model_from_joblib =joblib.load('outputs/datasets/feature/pca.pkl')
    
    

    # Scree Plot
    st.write(
            f"* ## Deciding How Many Principal Components to Retain"
            f"In order to decide how many principal components should be retained, it is common to summarise the results of a principal components analysis by making a scree plot. More about scree plot can be found **[here](http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html)**, and **[hear](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/)**."
            )




 
  