import streamlit as st
import pandas as pd
import numpy as np

st.cache(suppress_st_warning=True, allow_output_mutation=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
def load_data():
    df = pd.read_csv("outputs/datasets/cleaned/data.csv")
    return df
