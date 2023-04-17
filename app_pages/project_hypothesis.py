import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "" notebook
    st.success(
        f"*We beliveve that by looking at differant features we can differentiate between benign or malignant tumours. "
        f"\n\n"

        f"*This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection and hyper  parameter selection.\n "
        f" The goal is to classify whether the breast cancer is benign or malignant\n. "
        f"To achieve this i have used machine learning classification methods to fit a function that can predict the discrete class of new input.\n"
    )