import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_data_anaylsis import page_data_anaylsis
from app_pages.page_project_hypothesis import page_project_hypothesis_body


app = MultiPage(app_name= "Breast Cancer Prediction") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Data Anaysis", page_data_anaylsis())
app.add_page("Project Hypothesis", page_project_hypothesis_body())


app.run() # Run the  app