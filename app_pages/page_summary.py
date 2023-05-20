import streamlit as st


def page_summary_body():
    st.markdown("# Project Summary 🖥️")
    st.sidebar.markdown("# Project Summary 🖥️")

    st.title('# Using Predictive Analysis To Predict Diagnosis of a Breast Tumor')

    st.info('##  Identify the problem')
    st.write('Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a results of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.')

    st.info('## Expected outcome')
    st.write('Given breast cancer results from breast fine needle aspiration (FNA) test (is a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore or swelling) with a fine needle similar to a blood sample needle). Since this build a model that can classify a breast cancer tumor using two training classification:')

    st.info('* 1= Malignant (Cancerous) - Present')
    st.info('* 0= Benign (Not Cancerous) -Absent')

    st.info('### 1.2 Objective')
    st.write('Since the labels in the data are discrete, the predication falls into two categories, (i.e. Malignant or benign). In machine learning this is a classification problem.')

    st.write('> *Thus, the goal is to classify whether the breast cancer is benign or malignant and predict the recurrence and non-recurrence of malignant cases after a certain period.  To achieve this we have used machine learning classification methods to fit a function that can predict the discrete class of new input.*')

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Becky139/Breast-Cancer-Prediction/README.md).")

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested conducting a study to differentiate weather a tumor is benign or malignant. "

        f"* 2 - The client is interested in predicting whether a given tumour is malignant or benign based on the given features, with a high degree of accuracy. "
    )
