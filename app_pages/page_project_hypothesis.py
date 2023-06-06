import streamlit as st


def page_project_hypothesis_body():
    st.markdown("# Project Hypothesis 💡")
    st.sidebar.markdown("# Project Hypothesis 💡")

    st.write("### Project Hypothesis and Validation")

    # Hypothesis taen from readme file
    st.success(
        f"* We believe that by looking at different features we can with a good degree of accuracy differentiate between benign or malignant tumours. For this model, we would need the accuracy to be above 90% as stated in the business requirements. "
        f" \n\n"
        f"* This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection and hyperparameter selection. "
        f"The goal is to classify whether the breast cancer is benign or malignant. To achieve this, I have used machine learning classification methods to fit a function that can predict the discrete class of new input. "
    )
