import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"***Problem Statement**"
        f"*Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women.\n"
        f"*Breast Cancer occurs as a result of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor.\n"
        f"*A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous).\n "
        f"*Tests such as MRI, mammogram, ultrasound, and biopsy are commonly used to diagnose breast cancer performed.\n"
         
        f"Given breast cancer results from breast fine-needle aspiration (FNA) test (is a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore, or swelling) with a fine needle similar to a blood sample needle). Since this build a model that can classify a breast cancer tumor using two training classification:\n"
        f"* 1 = Malignant (Cancerous) - Present\n"
        f"* 0 = Benign (Not Cancerous) -Absent\n")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/breast-cancer-prediction/README.md).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested conducting a study to differentiate weather a tumor is benign or malignant.\n"
        
        f"* 2 - The client is interested in predicting whether a given tumour is malignant or benign based on the given features, with a high degree of accuracy.\n "
        )