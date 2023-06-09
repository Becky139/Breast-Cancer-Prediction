import streamlit as st
from src.data_management import raw_data


def page_summary_body():
    st.markdown("# Project Summary 🖥️")
    st.sidebar.markdown("# Project Summary 🖥️")

    st.title(
        "Using Predictive Analysis To Predict Diagnosis of a Breast Tumour"
        )

    st.info("##  Identify the problem")
    st.write(
        "Breast cancer is the most common malignancy among women, accounting \
        for nearly 1 in 3 cancers diagnosed among women in the United States, \
        and it is the second leading cause of cancer death among women. Breast\
        Cancer occurs as a result of abnormal growth of cells in the breast \
        tissue, commonly referred to as a tumour. A tumour does not mean \
        cancer - tumours can be benign (not cancerous), pre-malignant \
        (pre-cancerous)or malignant (cancerous). Tests such as MRI, mammogram,\
        ultrasound and biopsies are commonly used to diagnose breast cancer \
        performed."
    )

    st.info("## Expected outcome")
    st.write(
        "Given breast cancer results from breast fine needle aspiration (FNA)\
        test (is a quick and simple procedure to perform, which removes some\
        fluid or cells from a breast lesion or cyst (a lump, sore or swelling)\
        with a fine needle similar to a blood sample needle). Since this build\
        a model that can classify a breast cancer tumour using two training\
        classification:"
    )

    st.info("* 1= Malignant (Cancerous) - Present")
    st.info("* 0= Benign (Not Cancerous) -Absent")

    st.info("### Objective")
    st.write(
        "Since the labels in the data are discrete, the predication falls into\
        two categories, (i.e. Malignant or benign). In machine learning this\
        is a classification problem."
    )

    st.write(
        "> *Thus, the goal is to classify whether the breast cancer is benign\
        or malignant and predict the recurrence and non-recurrence of\
        malignant cases after a certain period.  To achieve this, we have used\
        machine learning classification methods to fit a function that can\
        predict the discrete class of new input.*"
    )
    
    st.write("Early-stage breast cancer often has a positive outlook with many\
        people surviving due to treatment. Early-stage breast cancer can include\
        several stages of breast cancer when cancer has not spread to distant \
        areas of the body. Treatments can often cure or cause cancer to go into \
        complete remission means that cancer cells are undetectable during \
        testing. This project will help people get a quicker diagnosis and \
        faster treatment helping to increase survival rates.")

    # Link to README file, so the users can have access to full project
    # documentation
    st.write(
        f"* For additional information (particularly regarding the dataset "
        f"and data preparation), please visit the [Project README file]"
        f"(https://github.com/Becky139/Breast-Cancer-Prediction/blob/main/"
        f"/README.md)."
    )

    # copied from README file - "Business Requirements" section
    st.success(
        "The project has 2 business requirements:\n"
        "* 1 - The client is interested conducting a study to differentiate\
        weather a tumour is benign or malignant.\n"
        "* As a client, I can navigate easily around an interactive dashboard\
        so that I can view and understand the data presented and I can view\
        and toggle visual graphs, charts of the data.\n\n"
        "* 2 - As a client, I can access and see the process used in\
        developing the machine learning model with access given to the\
        finalized model.\n"
        "* The client is interested in predicting whether a given tumour is\
        malignant or benign based on the given features, with a high degree\
        of accuracy.\n"
    )

    st.info("# Dataset Content")

    st.markdown(
        "* The dataset is on Kaggle it is publicly available, so it does not\
        require a license to use. Each row represents a person and the columns\
        have different signs or textures of tumours. The dataset contains **\
        569 samples of malignant and benign tumour cells**.\n"
        "* The first two columns in the dataset store the unique ID numbers of\
        the samples and the corresponding diagnosis(M=malignant, B=benign),\
        respectively.\n"
        "* The columns 3-32 contain 30 real-value features that have been\
        computed from digitized images of the cell nuclei, which can be used\
        to build a model to predict whether a tumour is benign or malignant.\n"
    )

    # load data
    df = raw_data()

    st.write("### Raw Cancer Dataset from Kaggle")

    # inspect data
    if st.checkbox("Inspect Cancer Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows."
        )

        st.write(df.head(10))

    st.write("---")

    st.markdown("<a href='#linkto_top'><button>Back to top</button></a>",
                unsafe_allow_html=True)
