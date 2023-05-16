# Contents of ~/my_app/streamlit_app.py
import streamlit as st


def page_summary_body():
    st.markdown("# Project Summary 🎈")
    st.sidebar.markdown("# Project Summary 🎈")


def page_data_analyis_body():
    st.markdown("# Data Analysis ❄️")
    st.sidebar.markdown("# Data Analysis❄️")


def page_data_preprocesing_body():
    st.markdown("# Data Preprocessing 🎉")
    st.sidebar.markdown("# Data Preprocessing 🎉")


def page_predictive_model_body():
    st.markdown("# Predictive Model 🎉")
    st.sidebar.markdown("# Predictive Model 🎉")


def page_project_hypothesis_body():
    st.markdown("# Project Hypothesis 🎉")
    st.sidebar.markdown("# Project Hypothesis 🎉")


def page_model_compairson_body():
    st.markdown("# Final Model 🎉")
    st.sidebar.markdown("# Final Model 🎉"


page_names_to_funcs = {
    "Project Summary": page_summary_body,
    "Data Analysis": page_data_anaylsis_body,
    "Data Preprocessing": page_data_preprocesing_body,
    "Predictive Model": page_predictive_model_body,
    "Project Hypothesis": page_project_hypothesis_body,
    "Final Model": page_model_compairson_body,
}


selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

