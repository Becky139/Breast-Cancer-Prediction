import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_data_analysis import page_data_analysis_body
from app_pages.page_predictive_model import page_predictive_model_body
from app_pages.page_model_comparison import page_model_comparison_body


# Create an instance of the app
app = MultiPage(app_name="Breast Cancer Prediction")


# load pages scripts
app.add_page("Project Summary", page_summary_body)
app.add_page("Data Analysis", page_data_analysis_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("Predictive Model", page_predictive_model_body)
app.add_page("Model Comparison", page_model_comparison_body)

app.run()  # Run the  app
