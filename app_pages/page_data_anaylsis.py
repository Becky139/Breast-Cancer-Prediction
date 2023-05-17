# Load libraries for data processing
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import joblib
from scipy.stats import norm
import seaborn as sns  # visualization
from src.data_management import load_data, load_pkl_file


def page_data_anaylsis_body():
    st.markdown("# Data Analysis 📊")
    st.sidebar.markdown("# Data Analysis📊")

    st.markdown(" ## 2.1 Objectives of Data Exploration")

    st.markdown("Exploratory data analysis (EDA) is a very important step which takes place after feature engineering and acquiring data and it should be done before any modeling. This is because it is very important for a data scientist to be able to understand the nature of the data without making assumptions. The results of data exploration can be extremely useful in grasping the structure of the data, the distribution of the values, and the presence of extreme values and interrelationships within the data set.")

    st.markdown("**The purpose of EDA is:**")

    st.markdown(
        "* to use summary statistics and visualizations to better understand data,")

    st.markdown("*find clues about the tendencies of the data, its quality and to formulate assumptions and the hypothesis of our analysis")

    st.markdown(
        "* For data preprocessing to be successful, it is essential to have an overall picture of your data")

    st.markdown("Basic statistical descriptions can be used to identify properties of the data and highlight which data values should  be treated as noise or outliers.*")

    st.markdown(
        "Next step is to explore the data. There are two approached used to examine the data using:")

    st.markdown("1. ***Descriptive statistics*** is the process of condensing key characteristics of the data set into simple numeric metrics. Some of the common metrics used are mean, standard deviation, and correlation.")

    st.markdown("2. ***Visualization*** is the process of projecting the data, or parts of it, into Cartesian space or into abstract images. In the data mining process, data exploration is leveraged in many different steps including preprocessing, modeling, and interpretation of results.")

    st.markdown("# 2.2 Descriptive statistics")
    st.markdown(
        "Summary statistics are measurements meant to describe data. In the field of descriptive statistics, there are many [summary measurements](http://www.saedsayad.com/numerical_variables.htm)")

    # basic descriptive statistics

    # descriptive_statistics = load_pkl_file(
    #   f'src/nb2/describe.pkl')
    # st.write(descriptive_statistics)

    # data["diagnosis"].replace({"B": 0, "M": 1}, inplace=True)

    # data.skew()

    st.markdown(
        ">The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.")
    st.markdown("From the graphs, we can see that **radius_mean**, **perimeter_mean**, **area_mean**, **concavity_mean** and **concave_points_mean** are useful in predicting cancer type due to the distinct grouping between malignant and benign cancer types in these features. We can also see that area_worst and perimeter_worst are also quite useful.")

    # data.diagnosis.unique()

    # M = 1 B = 0

    # Group by diagnosis and review the output.
    # diag_gr=data.groupby('diagnosis', axis=0)
    # pd.DataFrame(diag_gr.size(), columns=['# of observations'])

    # Check binary encoding from NB1 to confirm the coversion of the diagnosis categorical data into numeric, where
    # * Malignant = 1 (indicates prescence of cancer cells)
    # * Benign = 0 (indicates abscence)
    #
    # ##### **Observation**
    # > *357 observations indicating the absence of cancer cells and 212 show absence of cancer cell*
    #
    # Lets confirm this, by ploting the histogram

    st.markdown(" # 2.3 Unimodal Data Visualizations")

    st.markdown(" One of the main goals of visualizing the data here is to observe which features are most helpful in predicting malignant or benign cancer. The other is to see general trends that may aid us in model selection and hyper parameter selection.")

    st.markdown(
        "Apply 3 techniques that you can use to understand each attribute of your dataset independently.")
    st.markdown("* Histograms.")
    st.markdown("* Density Plots.")
    st.markdown("* Box and Whisker Plots.")

    # lets get the frequency of cancer diagnosis
    st.image('src/nb2/diagnosis.jpeg')

    st.markdown(" ## 2.3.1 Visualise distribution of data via histograms")
    st.markdown(" Histograms are commonly used to visualize numerical variables. A histogram is similar to a bar graph after the values of the variable are grouped (binned) into a finite number of intervals (bins).")

    st.markdown(" Histograms group data into bins and provide you a count of the number of observations in each bin. From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. It can also help you see possible outliers.")

    # ### Separate columns into smaller dataframes to perform visualization

    # Break up columns into groups, according to their suffix designation
    # (_mean, _se,
    # and __worst) to perform visualisation plots off.
    # Join the 'ID' and 'Diagnosis' back on
    # data_id_diag=data.loc[:,["id","diagnosis"]]
    # data_diag=data.loc[:, ["diagnosis"]]

    # For a merge + slice:
    # data_mean=data.iloc[:, 1:11]
    # data_se=data.iloc[:, 11:22]
    # data_worst=data.iloc[:, 23:]

    # print(data_id_diag.columns)
    # print(data_mean.columns)
   # print(data_se.columns)
   # print(data_worst.columns)

    # ### Histogram the "_mean" suffix designition
    st.image('src/nb2/hist_mean.jpeg')

    # Plot histograms of CUT1 variables

    # Any individual histograms, use this:
    # df_cut['radius_worst'].hist(bins=100)

    # ### __Histogram for  the "_se" suffix designition__

    # Plot histograms of _se variables
    st.image('src/nb2/hist_se.jpeg')

    # ### __Histogram "_worst" suffix designition__

    # Plot histograms of _worst variables
    st.image('src/nb2/hist_worst.jpeg')

    st.markdown(" ### __Observation__")

    st.markdown(" >We can see that perhaps the attributes  **concavity**,and **concavity_point ** may have an exponential distribution ( ). We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.")

    st.markdown(" ## 2.3.2 Visualize distribution of data via density plots")

    st.info('### Density plots "_mean" suffix designition')

    # Density Plots
    st.image('src/nb2/density_mean.jpeg')
  

    
    st.info('### Density plots "_se" suffix designition') 
    

    # Density Plots
    st.image('src/nb2/density_se.jpeg')

    st.info('### Density plot "_worst" suffix designition') 

    # Density Plots
    st.image('src/nb2/density_worst.jpeg')

    st.markdown(" ### Observation")
    st.markdown(" >We can see that perhaps the attributes perimeter,radius, area, concavity,ompactness may have an exponential distribution ( ). We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.")

    st.title('##  Visualise distribution of data via box plots')

    st.info('### Box plot "_mean" suffix designition')

    # box and whisker plots
    st.image('src/nb2/boxplot_mean.jpeg')

    st.info('### Box plot "_se" suffix designition')

    # box and whisker plots
    st.image('src/nb2/boxplot_se.jpeg')

    st.info('### Box plot "_worst" suffix designition')

    # box and whisker plots
    st.image('src/nb2/boxplot_worst.jpeg')

    st.markdown(" ### Observation")
    st.markdown(" >We can see that perhaps the attributes perimeter,radius, area, concavity,ompactness may have an exponential distribution ( ). We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.")

    st.markdown("# 2.4 Multimodal Data Visualizations")
    st.markdown(" * Scatter plots")
    st.markdown(" * Correlation matrix")

    st.markdown(" ### Correlation matrix")

    # plot correlation matrix
    st.image('src/nb2/correlation.jpeg')

    st.markdown(" ### Observation:")
    st.markdown(
        " We can see strong positive relationship exists with mean values paramaters between 1-0.75;.")
    st.markdown(
        " * The mean area of the tissue nucleus has a strong positive correlation with mean values of radius and parameter;")
    st.markdown(" * Some paramters are moderately positive corrlated (r between 0.5-0.75)are concavity and area, concavity and perimeter etc")
    st.markdown(" * Likewise, we see some strong negative correlation between fractal_dimension with radius, texture, parameter mean values.")

    st.image('src/nb2/scatter.jpeg')

    st.markdown(" ### Summary")

    st.markdown(" * Mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.")

    st.markdown(" * mean values of texture, smoothness, symmetry or fractual dimension does not show a particular preference of one diagnosis over the other.")

    st.markdown(
        " * In any of the histograms there are no noticeable large outliers that warrants further cleanup.")
