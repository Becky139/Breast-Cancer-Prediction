# Load libraries for data processing
import streamlit as st
from src.data_management import load_data, binary_data


def page_data_anaylsis_body():
    st.markdown("## Data Analysis 📊")
    st.sidebar.markdown("# Data Analysis📊")

    # load data
    df = load_data()

    st.write("### Cancer Dataset")

    # inspect data
    if st.checkbox("Inspect Cancer Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns,"
            f"find below the first 10 rows."
        )

        st.write(df.head(10))

    st.write("---")

    st.title("Objectives of Data Exploration")

    st.markdown(
        "Exploratory data analysis (EDA) is a very important step which takes\
        place after feature engineering and acquiring data and it should be \
        done before any modeling. This is because it is very important for a \
        data scientist to be able to understand the nature of the data without\
        making assumptions. The results of data exploration can be extremely \
        useful in grasping the structure of the data, the distribution of the \
        values, and the presence of extreme values and interrelationships \
        within the data set."
    )

    st.markdown("**The purpose of EDA is:**")

    st.markdown(
        "* to use summary statistics and visualizations to better understand \
        data,"
    )

    st.markdown(
        "*find clues about the tendencies of the data, its quality and to \
        formulate assumptions and the hypothesis of our analysis"
    )

    st.markdown(
        "* For data preprocessing to be successful, it is essential to have an\
        overall picture of your data"
    )

    st.markdown(
        "Basic statistical descriptions can be used to identify properties of\
        the data and highlight which data values should  be treated as noise \
        or outliers.*"
    )

    st.markdown(
        "Next step is to explore the data. There are two approached used to \
        examine the data using:"
    )

    st.markdown(
        "1. ***Descriptive statistics*** is the process of condensing key \
        characteristics of the data set into simple numeric metrics. Some of \
        the common metrics used are mean, standard deviation, and correlation."
    )

    st.markdown(
        "2. ***Visualization*** is the process of projecting the data, or \
        parts of it, into Cartesian space or into abstract images. In the data\
        mining process, data exploration is leveraged in many different steps \
        including preprocessing, modeling, and interpretation of results."
    )

    st.info("### Descriptive statistics")
    st.markdown(
        "Summary statistics are measurements meant to describe data. In the \
        field of descriptive statistics, there are many \
        [summary measurements](http://www.saedsayad.com/numerical_variables.htm)"
    )

    st.markdown("skew")

    st.markdown(
        ">The skew result show a positive (right) or negative (left) skew. \
        Values closer to zero show less skew."
    )
    st.markdown(
        "From the graphs, we can see that **radius_mean**, **perimeter_mean**,\
        **area_mean**, **concavity_mean** and **concave_points_mean** are \
        useful in predicting cancer type due to the distinct grouping between \
        malignant and benign cancer types in these features. We can also see \
        that area_worst and perimeter_worst are also quite useful."
    )

    st.info("### Unimodal Data Visualizations")

    # lets get the frequency of cancer diagnosis
    st.image("outputs/nb2/diagnosis.jpeg")

    st.markdown(
        "Check binary encoding from NB1 to confirm the coversion of the \
        diagnosis categorical data into numeric, where"
    )
    st.markdown("* Malignant = 1 (indicates prescence of cancer cells)")
    st.markdown("* Benign = 0 (indicates abscence)")

    df = binary_data()

    st.write("### Cancer Dataset With Binary encoding")
    # inspect data
    if st.checkbox("Inspect Binary Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the last 10 rows."
        )

        st.write(df.tail(10))

    st.write("---")
    st.success("**Observation**")
    st.markdown(
        "*357 observations indicating the absence of cancer cells and 212 show\
        absence of cancer cell *"
    )

    st.markdown("Lets confirm this, by ploting the histogram")

    st.markdown(
        " One of the main goals of visualizing the data here is to observe \
        which features are most helpful in predicting malignant or benign \
        cancer. The other is to see general trends that may aid us in model \
        selection and hyper parameter selection."
    )

    st.markdown(
        "Apply 3 techniques that you can use to understand each attribute of \
        your dataset independently."
    )
    st.markdown("* Histograms.")
    st.markdown("* Density Plots.")
    st.markdown("* Box and Whisker Plots.")

    st.title("Visualise distribution of data via histograms")
    st.markdown(
        " Histograms are commonly used to visualize numerical variables. A \
        histogram is similar to a bar graph after the values of the variable \
        are grouped (binned) into a finite number of intervals (bins)."
    )

    st.markdown(
        " Histograms group data into bins and provide you a count of the \
        number of observations in each bin. From the shape of the bins you can\
        quickly get a feeling for whether an attribute is Gaussian, skewed or \
        even has an exponential distribution. It can also help you see \
        possible outliers."
    )

    st.markdown(
        "Separate columns into smaller dataframes to perform visualization. \
        Break up columns into groups, according to their suffix designation"
    )

    st.markdown("1. data_mean.columns")
    st.markdown("2. data_se.columns")
    st.markdown("3. data_worst.columns")

    st.info("### Histogram the _mean suffix designition")

    # Plot histograms of _mean variables
    st.image("outputs/nb2/hist_mean.jpeg")

    st.info("### Histogram for  the _se suffix designition")

    # Plot histograms of _se variables
    st.image("outputs/nb2/hist_se.jpeg")

    st.info("### Histogram _worst suffix designition")

    # Plot histograms of _worst variables
    st.image("outputs/nb2/hist_worst.jpeg")

    st.success("Observation")
    st.markdown(
        "We can see that perhaps the attributes  **concavity**,and \
        **concavity_point ** may have an exponential distribution ( ). We \
        can also see that perhaps the texture and smooth and symmetry \
        attributes may have a Gaussian or nearly Gaussian distribution. This \
        is interesting because many machine learning techniques assume a \
        Gaussian univariate distribution on the input variables"
    )

    st.title("Visualize distribution of data via density plots")

    st.info('### Density plots "_mean" suffix designition')

    # Density Plots
    st.image("outputs/nb2/density_mean.jpeg")

    st.info('### Density plots "_se" suffix designition')

    # Density Plots
    st.image("outputs/nb2/density_se.jpeg")

    st.info('### Density plot "_worst" suffix designition')

    # Density Plots
    st.image("outputs/nb2/density_worst.jpeg")

    st.success("### Observation")
    st.markdown(
        "We can see that perhaps the attributes perimeter,radius, area, \
        concavity,ompactness may have an exponential distribution ( ). We can\
        also see that perhaps the texture and smooth and symmetry attributes \
        may have a Gaussian or nearly Gaussian distribution. This is \
        interesting because many machine learning techniques assume a Gaussian\
        univariate distribution on the input variables."
    )

    st.title("Visualise distribution of data via box plots")

    st.info('### Box plot "_mean" suffix designition')

    # box and whisker plots
    st.image("outputs/nb2/boxplot_mean.jpeg")

    st.info('### Box plot "_se" suffix designition')

    # box and whisker plots
    st.image("outputs/nb2/boxplot_se.jpeg")

    st.info('### Box plot "_worst" suffix designition')

    # box and whisker plots
    st.image("outputs/nb2/boxplot_worst.jpeg")

    st.success("### Observation")
    st.markdown(
        "We can see that perhaps the attributes perimeter,radius, area, \
        concavity,ompactness may have an exponential distribution ( ). We can\
        also see that perhaps the texture and smooth and symmetry attributes \
        may have a Gaussian or nearly Gaussian distribution. This is \
        interesting because many machine learning techniques assume a Gaussian\
        univariate distribution on the input variables."
    )

    st.info("## Multimodal Data Visualizations")
    st.markdown(" * Correlation matrix")
    st.markdown(" * Scatter plots")

    st.info(" ### Correlation matrix")

    # plot correlation matrix
    st.image("outputs/nb2/correlation.jpeg")

    st.success(" ### Observation:")
    st.markdown(
        " We can see strong positive relationship exists with mean values \
        paramaters between 1-0.75;."
    )
    st.markdown(
        " * The mean area of the tissue nucleus has a strong positive \
        correlation with mean values of radius and parameter;"
    )
    st.markdown(
        " * Some paramters are moderately positive corrlated (r between \
        0.5-0.75)are concavity and area, concavity and perimeter etc"
    )
    st.markdown(
        " * Likewise, we see some strong negative correlation between \
        fractal_dimension with radius, texture, parameter mean values."
    )

    st.info("### Scatter Plots")
    st.image("outputs/nb2/scatter.jpeg")

    st.success(" ### Summary")

    st.markdown(
        " * Mean values of cell radius, perimeter, area, compactness, \
        concavity and concave points can be used in classification of the \
        cancer. Larger values of these parameters tends to show a correlation\
        with malignant tumors."
    )

    st.markdown(
        " * mean values of texture, smoothness, symmetry or fractual dimension\
        does not show a particular preference of one diagnosis over the other."
    )

    st.markdown(
        " * In any of the histograms there are no noticeable large outliers \
        that warrants further cleanup."
    )

    st.title("Data Pre-Processing the data")

    st.markdown(
        "[Data preprocessing](http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html)\
        is a crucial step for any data analysis problem.\
        It is often a very good idea to prepare your data in such way to best \
        expose the structure of the problem to the machine learning algorithms\
        that you intend to use.This involves a number of activities such as:"
    )
    st.markdown("* Assigning numerical values to categorical data;")
    st.markdown("* Handling missing values; and")
    st.markdown(
        "* Normalizing the features (so that features on small scales do not \
        dominate when fitting a model to the data)."
    )

    st.info("### Goal:")

    st.markdown(
        "Find the most predictive features of the data and filter it so it \
        will enhance the predictive power of the analytics model."
    )

    st.markdown("#### Feature Standardization")

    st.markdown(
        "* Standardization is a useful technique to transform attributes with \
        a Gaussian distribution and differing means and standard deviations to\
        a standard Gaussian distribution with a mean of 0 and a standard \
        deviation of 1."
    )

    st.markdown(
        "* As seen in NB2_Exploratory data analysis the raw data has differing\
        distributions which may have an impact on the most ML algorithms. \
        Most machine learning and optimization algorithms behave much better \
        if features are on the same scale."
    )

    st.markdown(
        "Let’s evaluate the same algorithms with a standardized copy of the \
        dataset. Here, I use sklearn to scale and transform the data such that\
        each attribute has a mean value of zero and a standard deviation of\
        one"
    )

    st.image("outputs/nb3/pca.jpeg")

    st.success(
        "Now, what we got after applying the linear PCA transformation is a \
        lower dimensional subspace (from 3D to 2D in this case), where the \
        samples are “most spread” along the new feature axes."
    )

    st.info("## Deciding How Many Principal Components to Retain")

    st.markdown(
        "In order to decide how many principal components should be retained, \
        it is common to summarise the results of a principal components \
        analysis by making a scree plot. More about scree plot can be found \
        [here](http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html)\
        , and\
        [hear](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/)"
    )

    st.image("outputs/nb3/scree.jpeg")

    st.success("#### Observation")
    st.markdown(
        "The most obvious change in slope in the scree plot occurs at \
        component 2, which is the “elbow” of the scree plot. Therefore, it \
        cound be argued based on the basis of the scree plot that the first \
        three components should be retained."
    )

    st.success("### A Summary of the Data Preprocing Approach used here:")

    st.markdown(
        "1. assign features to a NumPy array X, and transform the class labels\
        from their original string representation (M and B) into integers"
    )
    st.markdown("2. Split data into training and test sets")
    st.markdown("3. Standardize the data.")
    st.markdown(
        "4. Obtain the Eigenvectors and Eigenvalues from the covariance matrix\
        or correlation matrix"
    )
    st.markdown(
        "5. Sort eigenvalues in descending order and choose the kk \
        eigenvectors that correspond to the kk largest eigenvalues where k is\
        the number of dimensions of the new feature subspace (k≤dk≤d)."
    )
    st.markdown(
        "6. Construct the projection matrix W from the selected k eigenvectors"
        )

    st.markdown(
        "7. Transform the original dataset X via W to obtain a k-dimensional \
        feature subspace Y."
    )

    st.markdown(
        "It is common to select a subset of features that have the largest \
        correlation with the class labels. The effect of feature selection \
        must be assessed within a complete modeling pipeline in order to give \
        you an unbiased estimated of your models true performance. Hence, in \
        the next section you will first be introduced to cross-validation, \
        before applying the PCA-based feature selection strategy in the model \
        building pipeline."
    )

    st.markdown("<a href='#linkto_top'><button>Back to top</button></a>",
                unsafe_allow_html=True)
