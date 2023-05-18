# Contents of ~/my_app/streamlit_app.py
import streamlit as st

from app_pages.page_summary import page_summary_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_data_anaylsis import page_data_anaylsis_body


def page_data_preprocesing_body():
    st.markdown("# Data Preprocessing 🎉")
    st.sidebar.markdown("# Data Preprocessing 🎉")





def page_model_compairson_body():
    st.markdown("# Final Model 🎉")
    st.sidebar.markdown("# Final Model 🎉")


page_names_to_funcs = {
    "Project Summary": page_summary_body,
    "Data Anaylsis": page_data_anaylsis_body,
    "Project Hypothesis": page_project_hypothesis_body,
    "Predictive Model":page_predictive_model_body,
}


selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()