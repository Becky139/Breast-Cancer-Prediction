import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_data_anaylsis import page_data_anaylsis
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predictive_model import page_predictive_model

app = MultiPage(app_name= "Breast Cancer Prediction") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body())
app.add_page("Data Anaysis", page_data_anaylsis())
app.add_page("Project Hypothesis", page_project_hypothesis_body())
app.add_page("Predictive Model", page_predictive_model())

app.run() # Run the  app