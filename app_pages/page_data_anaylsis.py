# Load libraries for data processing
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import norm
import seaborn as sns  # visualization


def page_data_anaylsis_body():
    st.markdown("# Data Analysis ❄️")
    st.sidebar.markdown("# Data Analysis❄️")
    @ st.cache(suppress_st_warning=True, allow_output_mutation=True)


    def load_data():
        data = pd.read_csv("outputs/datasets/cleaned/data.csv")
        return data

    st.markdown(" ## 2.1 Objectives of Data Exploration")

    st.markdown("Exploratory data analysis (EDA) is a very important step which takes place after feature engineering and acquiring data and it should be done before any modeling. This is because it is very important for a data scientist to be able to understand the nature of the data without making assumptions. The results of data exploration can be extremely useful in grasping the structure of the data, the distribution of the values, and the presence of extreme values and interrelationships within the data set.")

    st.markdown("**The purpose of EDA is:**")

    st.markdown(
    "* to use summary statistics and visualizations to better understand data,")

    st.markdown("*find clues about the tendencies of the data, its quality and to formulate assumptions and the hypothesis of our analysis")

    st.markdown("* For data preprocessing to be successful, it is essential to have an overall picture of your data")

    st.markdown("Basic statistical descriptions can be used to identify properties of the data and highlight which data values should  be treated as noise or outliers.*")

    st.markdown(
    "Next step is to explore the data. There are two approached used to examine the data using:")

    st.markdown("1. ***Descriptive statistics*** is the process of condensing key characteristics of the data set into simple numeric metrics. Some of the common metrics used are mean, standard deviation, and correlation.")

    st.markdown("2. ***Visualization*** is the process of projecting the data, or parts of it, into Cartesian space or into abstract images. In the data mining process, data exploration is leveraged in many different steps including preprocessing, modeling, and interpretation of results.")

    st.markdown("# 2.2 Descriptive statistics")
    st.markdown("Summary statistics are measurements meant to describe data. In the field of descriptive statistics, there are many [summary measurements](http://www.saedsayad.com/numerical_variables.htm)")
    

    # basic descriptive statistics
            data.describe()





            data["diagnosis"].replace({"B": 0, "M": 1}, inplace=True)





            data.skew()


    st.markdown(">The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.")
    st.markdown("From the graphs, we can see that **radius_mean**, **perimeter_mean**, **area_mean**, **concavity_mean** and **concave_points_mean** are useful in predicting cancer type due to the distinct grouping between malignant and benign cancer types in these features. We can also see that area_worst and perimeter_worst are also quite useful.")



            data.diagnosis.unique()


            # M = 1 B = 0

        


    # Group by diagnosis and review the output.
            diag_gr=data.groupby('diagnosis', axis=0)
            pd.DataFrame(diag_gr.size(), columns=['# of observations'])


            # Check binary encoding from NB1 to confirm the coversion of the diagnosis categorical data into numeric, where
            # * Malignant = 1 (indicates prescence of cancer cells)
            # * Benign = 0 (indicates abscence)
            #
            # ##### **Observation**
            # > *357 observations indicating the absence of cancer cells and 212 show absence of cancer cell*
            #
            # Lets confirm this, by ploting the histogram

        

            vars_to_study=['diagnosis']


            # # 2.3 Unimodal Data Visualizations
            #
            # One of the main goals of visualizing the data here is to observe which features are most helpful in predicting malignant or benign cancer. The other is to see general trends that may aid us in model selection and hyper parameter selection.
            #
            # Apply 3 techniques that you can use to understand each attribute of your dataset independently.
            # * Histograms.
            # * Density Plots.
            # * Box and Whisker Plots.

            # In[43]:


            # lets get the frequency of cancer diagnosis
            sns.set_style("white")
            sns.set_context({"figure.figsize": (10, 8)})
            sns.countplot(x=data['diagnosis'])
            plt.show()

            target_var='diagnosis'


            # ## 2.3.1 Visualise distribution of data via histograms
            # Histograms are commonly used to visualize numerical variables. A histogram is similar to a bar graph after the values of the variable are grouped (binned) into a finite number of intervals (bins).
            #
            # Histograms group data into bins and provide you a count of the number of observations in each bin. From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. It can also help you see possible outliers.

            # ### Separate columns into smaller dataframes to perform visualization

            # In[29]:


            # Break up columns into groups, according to their suffix designation
            # (_mean, _se,
            # and __worst) to perform visualisation plots off.
            # Join the 'ID' and 'Diagnosis' back on
            # data_id_diag=data.loc[:,["id","diagnosis"]]
            data_diag=data.loc[:, ["diagnosis"]]

            # For a merge + slice:
            data_mean=data.iloc[:, 1:11]
            data_se=data.iloc[:, 11:22]
            data_worst=data.iloc[:, 23:]

            # print(data_id_diag.columns)
            print(data_mean.columns)
            print(data_se.columns)
            print(data_worst.columns)



            # ### Histogram the "_mean" suffix designition

            # In[30]:


            # Plot histograms of CUT1 variables
            hist_mean=data_mean.hist(bins=10, figsize=(15, 10), grid=False,)

            # Any individual histograms, use this:
            # df_cut['radius_worst'].hist(bins=100)


            # ### __Histogram for  the "_se" suffix designition__

            # In[31]:


            # Plot histograms of _se variables
            hist_se=data_se.hist(bins=10, figsize=(15, 10), grid=False,)


            # ### __Histogram "_worst" suffix designition__

            # In[32]:


            # Plot histograms of _worst variables
            hist_worst=data_worst.hist(bins=10, figsize=(15, 10), grid=False,)


            # ### __Observation__
            #
            # >We can see that perhaps the attributes  **concavity**,and **concavity_point ** may have an exponential distribution ( ). We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.
            #

            # ## 2.3.2 Visualize distribution of data via density plots

            # ### Density plots "_mean" suffix designition

            # In[33]:


            # Density Plots
            plt=data_mean.plot(kind='density', subplots=True, layout=(4, 3), sharex=False,
                               sharey=False, fontsize=12, figsize=(15, 10))


            # ### Density plots "_se" suffix designition

            # In[34]:


            # Density Plots
            plt=data_se.plot(kind='density', subplots=True, layout=(4, 3), sharex=False,
                             sharey=False, fontsize=12, figsize=(15, 10))


            # ### Density plot "_worst" suffix designition

            # In[35]:


            # Density Plots
            plt=data_worst.plot(kind='kde', subplots=True, layout=(4, 3), sharex=False, sharey=False, fontsize=5,
                                figsize=(15, 10))


            # ### Observation
            # >We can see that perhaps the attributes perimeter,radius, area, concavity,ompactness may have an exponential distribution ( ). We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.

            # ## 2.3.3 Visualise distribution of data via box plots

            # ### Box plot "_mean" suffix designition

            # In[36]:


            # box and whisker plots
            plt=data_mean.plot(kind='box', subplots=True, layout=(
                4, 4), sharex=False, sharey=False, fontsize=12)


            # ### Box plot "_se" suffix designition

            # In[37]:


            # box and whisker plots
            plt=data_se.plot(kind='box', subplots=True, layout=(
                4, 4), sharex=False, sharey=False, fontsize=12)


            # ### Box plot "_worst" suffix designition

            # In[38]:


            # box and whisker plots
            plt=data_worst.plot(kind='box', subplots=True, layout=(
                4, 4), sharex=False, sharey=False, fontsize=12)


            # ### Observation
            # >We can see that perhaps the attributes perimeter,radius, area, concavity,ompactness may have an exponential distribution ( ). We can also see that perhaps the texture and smooth and symmetry attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.

            # # 2.4 Multimodal Data Visualizations
            # * Scatter plots
            # * Correlation matrix

            # ### Correlation matrix

            # In[39]:


            # plot correlation matrix
            plt.figure(figsize=(25, 12))
            corr=data.corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu",)


            # ### Observation:
            # We can see strong positive relationship exists with mean values paramaters between 1-0.75;.
            # * The mean area of the tissue nucleus has a strong positive correlation with mean values of radius and parameter;
            # * Some paramters are moderately positive corrlated (r between 0.5-0.75)are concavity and area, concavity and perimeter etc
            # * Likewise, we see some strong negative correlation between fractal_dimension with radius, texture, parameter mean values.
            #

            # In[40]:


            plt.style.use('fivethirtyeight')
            sns.set_style("white")

            g=sns.PairGrid(data[[data.columns[1], data.columns[2], data.columns[3],
                                 data.columns[4], data.columns[5], data.columns[6]]], hue='diagnosis')
            g=g.map_diag(plt.hist)
            g=g.map_offdiag(plt.scatter, s=3)


            # ### Summary
            #
            # * Mean values of cell radius, perimeter, area, compactness, concavity
            #     and concave points can be used in classification of the cancer. Larger
            #     values of these parameters tends to show a correlation with malignant
            #     tumors.
            # * mean values of texture, smoothness, symmetry or fractual dimension
            #     does not show a particular preference of one diagnosis over the other.
            #
            # * In any of the histograms there are no noticeable large outliers that warrants further cleanup.