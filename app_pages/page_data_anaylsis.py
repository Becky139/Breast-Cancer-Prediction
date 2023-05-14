import streamlit as st
from src.data_management import load_data


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_data_anaylsis():

    # load data
    df = load_data()

    st.write("### Cancer Predition Features")
    st.info(
        f"* Observation: We are going to ignore id column."
        f" As you can see on the above chart, even though some features left skewed, all of them follows the normal distribution.\n")

    # inspect data
    if st.checkbox("Inspect the data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

 
    # Chart
    st.write(
        f"One of the main goals of visualizing the data here is to observe which features are most helpful in predicting malignant or benign cancer."
        f"The other is to see general trends that may aid us in model selection and hyper parameter selection."
        )
    
    st.image('outputs/datasets/charts/diagnosis.jpeg')
    
    st.write(
        f"Histograms are commonly used to visualize numerical variables. A histogram is similar to a bar graph after the values of the variable are grouped (binned) into a finite number of intervals (bins)."

        f"Histograms group data into bins and provide you a count of the number of observations in each bin. From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. It can also help you see possible outliers."
    )
    
    st.image('outputs/datasets/charts/scatter_plot.jpg')

    st.info(
        f"From the above visual representation, we can analyze that the instances with area mean approximately in around the range of 800 to 2500 and smoothness mean from 0.08 to 0.14 is estimated to be of class 1 and the remaininng first half that is 0 to 500 of area mean and 0.07 to 0.12 smoothness mean is computed to be of class 0."
    )
    
    st.image('outputs/datasets/charts/pairplot_full.jpg')

    st.info(
        f"The above pairplot show the correlations between all the available features present in a dataset with the help of the seaborn library."
    )
    
    st.image('outputs/datasets/charts/pairplot.jpg')

    st.info(
        f"Likewise , the above representation demonstrates the correlation between the above provided features along with a distinct separation in our target classes."
    )
    
    
    st.image('outputs/datasets/charts/correlation.jpeg')

    st.info(
        f"Very much similar to the above done approaches, the above heatmap is a more visual and numerically informative representation of the correlationbetween the features. The principle is that the closer is the correlation value to 1 between the features , the lighter is the shade refering to higher correlation."
    )

    st.image("outputs/datasets/charts/perimeter_se.jpeg")

    st.info(
        f"The above graph displays the range of the perimeter_se present in our dataset along with the mode of respective perimeter_se values."
    )

    st.image("outputs/datasets/charts/perimeter_mean.jpeg")

    st.info(
        f"From the above representation, we know that the feature perimeter_mean has most of the instances ranging from around 75 to 105."
    )

    st.image("outputs/datasets/charts/texture_smoothness.jpeg")

    st.info(
        f"Given our primary understanding that the texture refers to the consistency in the surface is something that is somewhat associated to the smoothness factor, the graphical relationship was mapped and it was found that most of the instances with higher texture_mean had higher smoothness_mean value in general."
    )
    
