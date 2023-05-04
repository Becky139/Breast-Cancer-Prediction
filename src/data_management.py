import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.cache(suppress_st_warning=True, allow_output_mutation=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
def load_data():
    df = pd.read_csv("outputs/datasets/collection/data.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)