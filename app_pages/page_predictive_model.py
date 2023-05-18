import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def page_predictive_model_body():
    st.markdown("# Predictive Model 🎉")
    st.sidebar.markdown("# Predictive Model 🎉")

    st.markdown('# NB4 Predictive model using Support Vector Machine (SVM)')


    st.markdown('Support vector machines (SVMs) learning algorithm will be used to build the predictive model.  SVMs are one of the most popular classification algorithms, and have an elegant way of transforming nonlinear data so that one can use a linear algorithm to fit a linear model to the data (Cortes and Vapnik 1995)')
 
    st.markdown('Kernelized support vector machines are powerful models and perform well on a variety of datasets.')
    st.markdown('1. SVMs allow for complex decision boundaries, even if the data has only a few features.') 
    st.markdown('2. They work well on low-dimensional and high-dimensional data (i.e., few and many features), but don’t scale very well with the number of samples.')
    st.markdown('**Running an SVM on data with up to 10,000 samples might work well, but working with datasets of size 100,000 or more can become challenging in terms of runtime and memory usage.**')
 
    st.markdown('3. SVMs requires careful preprocessing of the data and tuning of the parameters. This is why, these days, most people instead use tree-based models such as random forests or gradient boosting (which require little or no preprocessing) in many applications.') 
    st.markdown('4.  SVM models are hard to inspect; it can be difficult to understand why a particular prediction was made, and it might be tricky to explain the model to a nonexpert.')

    st.markdown('### Important Parameters')
    st.markdown('The important parameters in kernel SVMs are the')
    st.markdown('* Regularization parameter C,') 
    st.markdown('* The choice of the kernel,(linear, radial basis function(RBF) or polynomial)')
    st.markdown('* Kernel-specific parameters.')
 
    st.markdown('gamma and C both control the complexity of the model, with large values in either resulting in a more complex model. Therefore, good settings for the two parameters are usually strongly correlated, and C and gamma should be adjusted together.')

    st.markdown('#### Classification with cross-validation')

    st.markdown('As discussed in notebook [NB3](https://github.com/ShiroJean/Breast-cancer-risk-prediction/blob/master/NB3_DataPreprocesing.ipynb) splitting the data into test and training sets is crucial to avoid overfitting. This allows generalization of real, previously-unseen data. Cross-validation extends this idea further. Instead of having a single train/test split, we specify **so-called folds** so that the data is divided into similarly-sized folds.') 
 
    st.markdown('* Training occurs by taking all folds except one – referred to as the holdout sample.') 
    st.markdown('* On the completion of the training, you test the performance of your fitted model using the holdout sample.') 

    st.markdown('* The holdout sample is then thrown back with the rest of the other folds, and a different fold is pulled out as the new holdout sample.') 
 
    st.markdown('* Training is repeated again with the remaining folds and we measure performance using the holdout sample. This process is repeated until each fold has had a chance to be a test or holdout sample.') 
 
    st.markdown('* The expected performance of the classifier, called cross-validation error, is then simply an average of error rates computed on each holdout sample.') 
 
    st.markdown('This process is demonstrated by first performing a standard train/test split, and then computing cross-validation error.')

    st.image('src/nb4/classifier_accuracy.png')


    st.markdown('To get a better measure of prediction accuracy (which you can use as a proxy for “goodness of fit” of the model), you can successively split the data into folds that you will use for training and testing:')

    st.image('src/nb4/cross_validation_classifier.png')


    st.markdown('The above evaluations were based on using the entire set of features. You will now employ the correlation-based feature selection strategy to assess the effect of using 3 features which have the best correlation with the class labels.')

    st.image('src/nb4/cross_validation_feature.png')

    st.image('src/nb4/score_uncertainty.png')


    st.markdown('From the above results, you can see that only a fraction of the features are required to build a model that performs similarly to models based on using the entire set of features.')
    st.markdown('Feature selection is an important part of the model-building process that you must always pay particular attention to. The details are beyond the scope of this notebook. In the rest of the analysis, you will continue using the entire set of features.')


    st.markdown('### Model Accuracy: Receiver Operating Characteristic (ROC) curve')
 
    st.markdown('In statistical modeling and machine learning, a commonly-reported performance measure of model accuracy for binary classification problems is Area Under the Curve (AUC).')
 
    st.markdown('To understand what information the ROC curve conveys, consider the so-called confusion matrix that essentially is a two-dimensional table where the classifier model is on one axis (vertical), and ground truth is on the other (horizontal) axis, as shown below. Either of these axes can take two values (as depicted)')
 

    st.markdown('Model says "+" |Model says  "-" --- | --- | ---True positive` | `False negative` | ** Actual: "+" **`False positive`  | `True negative` |  Actual: "-"')
    st.markdown('In an ROC curve, you plot “True Positive Rate” on the Y-axis and “False Positive Rate” on the X-axis, where the values “true positive”, “false negative”, “false positive”, and “true negative” are events (or their probabilities) as described above. The rates are defined according to the following:')
    st.markdown('> * True positive rate (or sensitivity)}: tpr = tp / (tp + fn)')
    st.markdown('> * False positive rate:       fpr = fp / (fp + tn)')
    st.markdown('> * True negative rate (or specificity): tnr = tn / (fp + tn)')

    st.markdown('In all definitions, the denominator is a row margin in the above confusion matrix. Thus,one can  expres * the true positive rate (tpr) as the probability that the model says "+" when the real value is indeed "+" (i.e., a conditional probability). However, this does not tell you how likely you are to be correct when calling "+" (i.e., the probability of a true positive, conditioned on the test result being "+").')
           
    st.image('src/nb4/classification_report.jpeg')


    st.markdown('#### Observation') 
    st.markdown('There are two possible predicted classes: "1" and "0". Malignant = 1 (indicates prescence of cancer cells) and Benign')
    st.markdown('= 0 (indicates abscence).')
 
    st.markdown('* The classifier made a total of 174 predictions (i.e 174 patients were being tested for the presence breast cancer).')
    st.markdown('* Out of those 174 cases, the classifier predicted "yes" 58 times, and "no" 113 times.')
    st.markdown('* In reality, 64 patients in the sample have the disease, and 107 patients do not.')
     
    st.markdown('#### Rates as computed from the confusion matrix')
    st.markdown('1. **Accuracy**: Overall, how often is the classifier correct?')
    st.markdown('     * (TP+TN)/total = (57+106)/171 = 0.95')
     
    st.markdown('2. **Misclassification Rate**: Overall, how often is it wrong?')
    st.markdown('     * (FP+FN)/total = (1+7)/171 = 0.05 equivalent to 1 minus Accuracy also known as ***"Error Rate"***')
     
    st.markdown('3. **True Positive Rate:** When its actually yes, how often does it predict 1?')
    st.markdown('    * TP/actual yes = 57/64 = 0.89 also known as "Sensitivity" or ***"Recall"***')
     
    st.markdown('4. **False Positive Rate**: When its actually 0, how often does it predict 1?')
    st.markdown('    * FP/actual no = 1/107 = 0.01')
     
    st.markdown(' 5. **Specificity**: When its actually 0, how often does it predict 0? also know as **true positive rate**')
    st.markdown('    * TN/actual no = 106/107 = 0.99 equivalent to 1 minus False Positive Rate')
     
    st.markdown(' 6. **Precision**: When it predicts 1, how often is it correct?')
    st.markdown('    * TP/predicted yes = 57/58 = 0.98')
     
    st.markdown(' 7. **Prevalence**: How often does the yes condition actually occur in our sample?')
    st.markdown('    * actual yes/total = 64/171 = 0.34')


    st.image(src/nb4/roc_curve.jpeg)


    st.markdown('* To interpret the ROC correctly, consider what the points that lie along the diagonal represent. For these situations, there is an equal chance of "+" and "-" happening. Therefore, this is not that different from making a prediction by tossing of an unbiased coin. Put simply, the classification model is random.')
     
    st.markdown('* For the points above the diagonal, tpr > fpr, and the model says that you are in a zone where you are performing better than random. For example, assume tpr = 0.99 and fpr = 0.01, Then, the probability of being in the true positive group is $(0.99 / (0.99 + 0.01)) = 99\%$. Furthermore, holding fpr constant, it is easy to see that the more vertically above the diagonal you are positioned, the better the classification model.')

    st.markdown('## Next I will look into optimizing the class')
