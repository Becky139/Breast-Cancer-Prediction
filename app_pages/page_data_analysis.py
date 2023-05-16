# Load libraries for data processing
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import norm
import seaborn as sns  # visualization


def page_data_analyis_body():

    @ st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def load_data():
        data = pd.read_csv("outputs/data/clean_data.csv")
        return data

    st.markdown(" ## 2.1 Objectives of Data Exploration")

    st.markdown("Exploratory data analysis (EDA) is a very important step which takes place after feature engineering and acquiring data and it should be done before any modeling. This is because it is very important for a data scientist to be able to understand the nature of the data without making assumptions. The results of data exploration can be extremely useful in grasping the structure of the data, the distribution of the values, and the presence of extreme values and interrelationships within the data set.")

    st.markdown("**The purpose of EDA is:**")

    st.markdown(
    "* to use summary statistics and visualizations to better understand data,")

    st.markdown("*find clues about the tendencies of the data, its quality and to formulate assumptions and the hypothesis of our analysis")

    st.markdown("* For data preprocessing to be successful, it is essential to have an overall picture of your data")

    st.markdown("Basic statistical descriptions can be used to identify properties of the data and highlight which data values should  be treated as noise or outliers.*")

    st.markdown(
    "Next step is to explore the data. There are two approached used to examine the data using:")

    st.markdown("1. ***Descriptive statistics*** is the process of condensing key characteristics of the data set into simple numeric metrics. Some of the common metrics used are mean, standard deviation, and correlation.")

    st.markdown("2. ***Visualization*** is the process of projecting the data, or parts of it, into Cartesian space or into abstract images. In the data mining process, data exploration is leveraged in many different steps including preprocessing, modeling, and interpretation of results.")

    st.markdown("# 2.2 Descriptive statistics")
    st.markdown("Summary statistics are measurements meant to describe data. In the field of descriptive statistics, there are many [summary measurements](http://www.saedsayad.com/numerical_variables.htm)")
    