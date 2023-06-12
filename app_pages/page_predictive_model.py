import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def page_predictive_model_body():
    st.markdown("# Predictive Model 🎉")
    st.sidebar.markdown("# Predictive Model 🎉")

    st.title(" Predictive model using Support Vector Machine (SVM)")

    st.markdown(
        "Support vector machines (SVMs) learning algorithm will be used to \
        build the predictive model.  SVMs are one of the most popular \
        classification algorithms, and have an elegant way of transforming \
        nonlinear data so that one can use a linear algorithm to fit a linear \
        model to the data (Cortes and Vapnik 1995)"
    )
    st.markdown(
        "Support vector machines (SVMs) learning algorithm will be used to \
        build the predictive model.  SVMs are one of the most popular \
        classification algorithms, and have an elegant way of transforming \
        nonlinear data so that one can use a linear algorithm to fit a linear \
        model to the data (Cortes and Vapnik 1995)"
    )
    st.markdown(
        "Kernelized support vector machines are powerful models and perform \
        well on a variety of datasets."
    )
    st.markdown(
        "1. SVMs allow for complex decision boundaries, even if the data has \
        only a few features."
    )
    st.markdown(
        "1. SVMs allow for complex decision boundaries, even if the data has \
        only a few features."
    )

    st.markdown(
        "3. SVMs requires careful preprocessing of the data and tuning of the \
        parameters. This is why, these days, most people instead use \
        tree-based models such as random forests or gradient boosting (which \
        require little or no preprocessing) in many applications."
    )
    st.markdown(
        "3. SVMs requires careful preprocessing of the data and tuning of the \
        parameters. This is why, these days, most people instead use \
        tree-based models such as random forests or gradient boosting (which \
        require little or no preprocessing) in many applications."
    )
    st.markdown(
        "3. SVMs requires careful preprocessing of the data and tuning of the \
        parameters. This is why, these days, most people instead use \
        tree-based models such as random forests or gradient boosting (which \
        require little or no preprocessing) in many applications."
    )
    st.markdown(
        "4.  SVM models are hard to inspect; it can be difficult to understand\
        why a particular prediction was made, and it might be tricky to \
        explain the model to a nonexpert."
    )

    st.markdown("* Regularization parameter C,")
    st.markdown(
        "* The choice of the kernel,(linear, radial basis function(RBF) or \
        polynomial)"
    )

    st.markdown(
        "* The choice of the kernel,(linear, radial basis function(RBF) or \
        polynomial)"
    )

    st.markdown(
        "gamma and C both control the complexity of the model, with large \
        values in either resulting in a more complex model. Therefore, good \
        settings for the two parameters are usually strongly correlated, and C\
        and gamma should be adjusted together."
    )

    st.markdown(
        "* Training occurs by taking all folds except one – referred to as the\
        holdout sample."
    )
    st.markdown(
        "* On the completion of the training, you test the performance of your\
        fitted model using the holdout sample."
    )

    st.markdown(
        "* The holdout sample is then thrown back with the rest of the other \
        folds, and a different fold is pulled out as the new holdout sample."
    )

    st.markdown(
        "* Training is repeated again with the remaining folds and we measure \
        performance using the holdout sample. This process is repeated until \
        each fold has had a chance to be a test or holdout sample."
    )

    st.markdown(
        "* The expected performance of the classifier, called cross-validation\
        error, is then simply an average of error rates computed on each \
        holdout sample."
    )

    st.markdown(
        "* Training is repeated again with the remaining folds and we measure \
        performance using the holdout sample. This process is repeated until \
        each fold has had a chance to be a test or holdout sample."
    )

    st.markdown(
        "* The expected performance of the classifier, called cross-validation\
         error, is then simply an average of error rates computed on each \
        holdout sample."
    )

    st.markdown(
        "This process is demonstrated by first performing a standard \
        train/test split, and then computing cross-validation error."
    )

    st.image("outputs/nb4/classifier_accuracy.png")
    st.markdown(
        "To get a better measure of prediction accuracy (which you can use as \
        a proxy for “goodness of fit” of the model), you can successively \
        split the data into folds that you will use for training and testing:"
    )

    st.markdown(
        "The above evaluations were based on using the entire set of features.\
         You will now employ the correlation-based feature selection strategy \
        to assess the effect of using 3 features which have the best \
        correlation with the class labels."
    )

    st.image("outputs/nb4/cross_validation_feature.png")
    st.image("outputs/nb4/score_uncertainty.png")

    st.markdown(
        "From the above results, you can see that only a fraction of the \
        features are required to build a model that performs similarly to \
        models based on using the entire set of features."
    )

    st.info("### Model Accuracy Receiver Operating Characteristic (ROC) curve")

    st.markdown(
        "Feature selection is an important part of the model-building process \
        that you must always pay particular attention to. The details are \
        beyond the scope of this notebook. In the rest of the analysis, you \
        will continue using the entire set of features."
    )

    st.markdown(
        'Model says "+" |Model says  "-" --- | --- | ---True positive` | \
        `False negative` | ** Actual: "+" **`False positive`  | \
        `True negative` |  Actual: "-"'
    )
    st.markdown(
        "In an ROC curve, you plot “True Positive Rate” on the Y-axis and \
        “False Positive Rate” on the X-axis, where the values “true positive”,\
        “false negative”, “false positive”, and “true negative” are events \
        (or their probabilities) as described above. The rates are defined \
        according to the following:"
    )
    st.markdown(
        "> * True positive rate (or sensitivity)}: tpr = tp / \
    (tp + fn)"
    )
    st.markdown("> * False positive rate:       fpr = fp / (fp + tn)")
    st.markdown("> * True negative rate (or specificity) tnr = tn / (fp + tn)")

    st.image("outputs/nb4/classification_report.jpeg")

    st.success("#### Observation")
    st.markdown(
        'There are two possible predicted classes: "1" and "0". Malignant = 1\
        (indicates prescence of cancer cells) and Benign'
    )

    st.markdown(
        '* Out of those 174 cases, the classifier predicted "yes" 58 times, \
        and "no" 113 times.'
    )
    st.markdown(
        "* In reality, 64 patients in the sample have the disease, and 107 \
        patients do not."
    )

    st.info("### Rates as computed from the confusion matrix")

    st.markdown(
        '     * (FP+FN)/total = (1+7)/171 = 0.05 equivalent to 1 minus \
        Accuracy also known as ***"Error Rate"***'
    )

    st.markdown(
        "3. **True Positive Rate:** When its actually yes, how often does it \
        predict 1?"
    )
    st.markdown(
        '    * TP/actual yes = 57/64 = 0.89 also known as "Sensitivity" or \
        ***"Recall"***'
    )

    st.markdown(
        "4. **False Positive Rate**: When its actually 0, how often does it \
        predict 1?"
    )

    st.markdown(
        " 5. **Specificity**: When its actually 0, how often does it \
        predict 0? also know as **true positive rate**"
    )
    st.markdown(
        "    * TN/actual no = 106/107 = 0.99 equivalent to 1 minus False \
        Positive Rate"
    )

    st.markdown(
        " 6. **Precision**: When it predicts 1, how often is it \
    correct?"
    )

    st.markdown(
        " 7. **Prevalence**: How often does the yes condition actually occur \
        in our sample?"
    )

    st.markdown(
        " 6. **Precision**: When it predicts 1, how often is it \
    correct?"
    )
    st.markdown(
        '* To interpret the ROC correctly, consider what the points that lie \
        along the diagonal represent. For these situations, there is an equal \
        chance of "+" and "-" happening. Therefore, this is not that different\
        from making a prediction by tossing of an unbiased coin. Put simply, \
        the classification model is random.'
    )
    st.markdown(
        " 6. **Precision**: When it predicts 1, how often is it \
        correct?"
    )
    st.markdown(
        " 7. **Prevalence**: How often does the yes condition actually occur \
        in our sample?"
    )
    st.markdown(
        " 7. **Prevalence**: How often does the yes condition actually occur \
        in our sample?"
    )

    st.image("outputs/nb4/roc_curve.jpeg")

    st.markdown(
        '* To interpret the ROC correctly, consider what the points that lie \
        along the diagonal represent. For these situations, there is an equal \
        chance of "+" and "-" happening. Therefore, this is not that different\
        from making a prediction by tossing of an unbiased coin. Put simply, \
        the classification model is random.'
    )

    st.markdown(
        "For the points above the diagonal, tpr > fpr, and the model says that\
        you are in a zone where you are performing better than random. For \
        example, assume tpr = 0.99 and fpr = 0.01, Then, the probability of \
        being in the true positive group is $(0.99 / (0.99 + 0.01)) = 99%. \
        Furthermore, holding fpr constant, it is easy to see that the more \
        vertically above the diagonal you are positioned, the better the \
        classification model."
    )

    st.markdown("## Next I will look into optimizing the class")

    st.title("Optimizing the SVM Classifier")

    st.markdown(
        "Machine learning models are parameterized so that their behavior can \
        be tuned for a given problem. Models can have many parameters and \
        finding the best combination of parameters can be treated as a search \
        problem. In this notebook, I aim to tune parameters of the SVM \
        Classification model using scikit-learn."
    )

    st.info(
        "### Build a predictive model and evaluate with 5-cross validation \
        using support vector classifies (ref NB4) for details"
    )

    st.image("outputs/nb5/accuracy_report.png")
    st.image("outputs/nb5/classification_report.jpeg")

    st.info("### Importance of Optimizing a Classifier")

    st.markdown("We can tune two key parameters of the SVM algorithm:")
    st.markdown("* the value of C (how much to relax the margin)")
    st.markdown("* and the type of kernel.")

    st.markdown(
        "The default for SVM (the SVC class) is to use the Radial Basis \
        Function (RBF) kernel with a C value set to 1.0. Like with KNN, we \
        will perform a grid search using 10-fold cross validation with a \
        standardized copy of the training dataset. We will try a number of \
        simpler kernel types and C values with less bias and more bias (less \
        than and more than 1.0 respectively)."
    )

    st.markdown(
        "Python scikit-learn provides two simple methods for algorithm \
        parameter tuning:"
    )
    st.markdown("* Grid Search Parameter Tuning.")
    st.markdown("* Random Search Parameter Tuning.")

    st.image("outputs/nb5/grid_svc_accuracy.png")
    st.image("outputs/nb5/grid_svc_classification.jpeg")

    st.info("### Decision Boundaries of Different Classifiers")
    st.markdown(
        "Lets see the decision boundaries produced by the linear, Gaussian \
        and polynomial classifiers."
    )

    st.image("outputs/nb5/svc_charts.jpeg")

    st.success("## Conclusion")

    st.markdown(
        "This work demonstrates the modelling of breast cancer as \
        classification task using Support Vector Machine"
    )

    st.markdown(
        "The SVM performs better when the dataset is standardized so that all\
        attributes have a mean value of zero and a standard deviation of one. \
        We can calculate this from the entire training dataset and apply the \
        same transform to the input attributes from the validation dataset."
    )

    st.markdown("Next Task:")
    st.markdown("1. Summary and conclusion of findings")
    st.markdown("2. Compare with other classification methods")
    st.markdown("    * Decision trees with tree.DecisionTreeClassifier();")
    st.markdown(
        "    * K-nearest neighbors with neighbors.KNeighborsClassifier\
    ();"
    )
    st.markdown("    * Random forests with ensemble.RandomForestClassifier();")
    st.markdown(
        "    * Perceptron (both gradient and stochastic gradient) with \
        mlxtend.classifier.Perceptron; and"
    )
    st.markdown(
        "    * Multilayer perceptron network (both gradient and stochastic \
        gradient) with mlxtend.classifier.MultiLayerPerceptron."
    )
