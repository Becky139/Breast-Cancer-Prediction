import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def page_optimizing_classifier_body():
    st.markdown("# Optimizing Classifier 🚀")
    st.sidebar.markdown("# Optimizing Classifier 🚀")

    st.markdown('# Optimizing the SVM Classifier')

    st.markdown('Machine learning models are parameterized so that their behavior can be tuned for a given problem. Models can have many parameters and finding the best combination of parameters can be treated as a search problem. In this notebook, I aim to tune parameters of the SVM Classification model using scikit-learn.')

    st.markdown('#### Build a predictive model and evaluate with 5-cross validation using support vector classifies (ref NB4) for details')

    st.image('outputs/nb5/classification_report.jpeg')

    st.markdown('## Importance of optimizing a classifier')

    st.markdown('We can tune two key parameters of the SVM algorithm:')
    st.markdown('* the value of C (how much to relax the margin)')
    st.markdown('* and the type of kernel.')

    st.markdown('The default for SVM (the SVC class) is to use the Radial Basis Function (RBF) kernel with a C value set to 1.0. Like with KNN, we will perform a grid search using 10-fold cross validation with a standardized copy of the training dataset. We will try a number of simpler kernel types and C values with less bias and more bias (less than and more than 1.0 respectively).')

    st.markdown('Python scikit-learn provides two simple methods for algorithm parameter tuning:')
    st.markdown('* Grid Search Parameter Tuning.')
    st.markdown('* Random Search Parameter Tuning.')

    st.image('outputs/nb5/grid_svc_classification.jpeg')

    st.markdown('### Decision boundaries of different classifiers')
    st.markdown('Lets see the decision boundaries produced by the linear, Gaussian and polynomial classifiers.')

    st.image('outputs/nb5/svc_charts.jpeg')

    st.markdown('## Conclusion')

    st.markdown('This work demonstrates the modelling of breast cancer as classification task using Support Vector Machine')

    st.markdown('The SVM performs better when the dataset is standardized so that all attributes have a mean value of zero and a standard deviation of one. We can calculate this from the entire training dataset and apply the same transform to the input attributes from the validation dataset.')

    st.markdown('Next Task:')
    st.markdown('1. Summary and conclusion of findings')
    st.markdown('2. Compare with other classification methods')
    st.markdown('    * Decision trees with tree.DecisionTreeClassifier();')
    st.markdown('    * K-nearest neighbors with neighbors.KNeighborsClassifier();')
    st.markdown('    * Random forests with ensemble.RandomForestClassifier();')
    st.markdown('    * Perceptron (both gradient and stochastic gradient) with mlxtend.classifier.Perceptron; and')
    st.markdown('    * Multilayer perceptron network (both gradient and stochastic gradient) with mlxtend.classifier.MultiLayerPerceptron.')
