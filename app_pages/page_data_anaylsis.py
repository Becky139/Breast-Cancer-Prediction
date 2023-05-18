# Load libraries for data processing
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import joblib
from scipy.stats import norm
import seaborn as sns  # visualization


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


    st.markdown('# Data Pre-Processing the data')

    st.markdown('[Data preprocessing](http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html) is a crucial step for any data analysis problem.  It is often a very good idea to prepare your data in such way to best expose the structure of the problem to the machine learning algorithms that you intend to use.This involves a number of activities such as:')
    st.markdown('* Assigning numerical values to categorical data;')
    st.markdown('* Handling missing values; and')
    st.markdown('* Normalizing the features (so that features on small scales do not dominate when fitting a model to the data).')

    st.markdown('### Goal:')

    st.markdown('Find the most predictive features of the data and filter it so it will enhance the predictive power of the analytics model.') 

    st.markdown('#### Load data and essential libraries')

    st.markdown('#### Feature Standardization')
 
    st.markdown('* Standardization is a useful technique to transform attributes with a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.') 

    st.markdown('* As seen in [NB2_Exploratory data analysis](https://github.com/ShiroJean/Breast-cancer-risk-prediction/blob/master/NB2_ExploratoryDataAnalysis.ipynb) the raw data has differing distributions which may have an impact on the most ML algorithms. Most machine learning and optimization algorithms behave much better if features are on the same scale.')


    st.markdown('Let’s evaluate the same algorithms with a standardized copy of the dataset. Here, I use sklearn to scale and transform the data such that each attribute has a mean value of zero and a standard deviation of one')
# 


    st.image('src/nb3/pca.jpeg')

    st.markdown('Now, what we got after applying the linear PCA transformation is a lower dimensional subspace (from 3D to 2D in this case), where the samples are “most spread” along the new feature axes.')

    st.markdown('## Deciding How Many Principal Components to Retain')

    st.markdown('In order to decide how many principal components should be retained, it is common to summarise the results of a principal components analysis by making a scree plot. More about scree plot can be found [here](http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html), and [hear](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/)')

    st.image('src/nb3/scree.jpeg')

    st.markdown('#### Observation')
    st.markdown('The most obvious change in slope in the scree plot occurs at component 2, which is the “elbow” of the scree plot. Therefore, it cound be argued based on the basis of the scree plot that the first three components should be retained.')

    st.markdown('### A Summary of the Data Preprocing Approach used here:')
    
    st.markdown('1. assign features to a NumPy array X, and transform the class labels from their original string representation (M and B) into integers')
    st.markdown('2. Split data into training and test sets')
    st.markdown('3. Standardize the data.')
    st.markdown('4. Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix')
    st.markdown('5. Sort eigenvalues in descending order and choose the kk eigenvectors that correspond to the kk largest eigenvalues where k is the number of dimensions of the new feature subspace (k≤dk≤d).')
    st.markdown('6. Construct the projection matrix W from the selected k eigenvectors.')
    st.markdown('7. Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.')

    st.markdown('It is common to select a subset of features that have the largest correlation with the class labels. The effect of feature selection must be assessed within a complete modeling pipeline in order to give you an unbiased estimated of your models true performance. Hence, in the next section you will first be introduced to cross-validation, before applying the PCA-based feature selection strategy in the model building pipeline.')

