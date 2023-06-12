import streamlit as st


def page_model_comparison_body():
    st.markdown("# Model Comparison 💽")
    st.sidebar.markdown("# Model Comparison 💽")

    st.title("Comparison Between Different Classifiers")

    st.info("## Automate the ML process using pipelines")

    st.markdown(
        "There are standard workflows in a machine learning project that can \
        be automated. In Python scikit-learn, Pipelines help to clearly define\
        and automate these workflows."
    )
    st.markdown(
        "* Pipelines help overcome common problems like data leakage in your \
        test harness."
    )
    st.markdown(
        "* Python scikit-learn provides a Pipeline utility to help automate \
        machine learning workflows."
    )
    st.markdown(
        "* Pipelines work by allowing for a linear sequence of data transforms\
        to be chained together culminating in a modeling process that can be \
        evaluated."
    )

    st.info("### Data Preparation and Modeling Pipeline")

    st.markdown("### Evaluate Some Algorithms")
    st.markdown(
        "Now it is time to create some models of the data and estimate their \
        accuracy on unseen data. Here is what we are going to cover in this \
        step:"
    )
    st.markdown("1. Separate out a validation dataset.")
    st.markdown("2. Setup the test harness to use 10-fold cross validation.")
    st.markdown("3. Build 5 different models")
    st.markdown("4. Select the best model")

    st.info("## Evaluate Algorithms: Baseline")

    st.image("outputs/nb6/six_classifiers.png")

    st.success("### Observation")
    st.markdown(
        "The results suggest That both Logistic Regression and LDA may be \
        worth further study. These are just mean accuracy values. It is always\
        wise to look at the distribution of accuracy values calculated across \
        cross validation folds. We can do that graphically using box and \
        whisker plots."
    )

    st.image("outputs/nb6/accuracy_baseline.png")

    st.success("### Observation")
    st.markdown(
        "The results show a similar tight distribution for all classifiers \
        except SVM which is encouraging, suggesting low variance. The poor \
        results for SVM are surprising."
    )

    st.markdown(
        "It is possible the varied distribution of the attributes may have an \
        effect on the accuracy of algorithms such as SVM. In the next section \
        we will repeat this spot-check with a standardized copy of the \
        training dataset."
    )

    st.info("### Evaluate Algorithms: Standardize Data")

    st.image("outputs/nb6/standardize_data.png")

    st.image("outputs/nb6/scaled_algorithm.png")

    st.success("#### Observations")
    st.markdown(
        "The results show that standardization of the data has lifted the \
        skill of SVM to be the most accurate algorithm tested so far."
    )

    st.markdown(
        "The results suggest digging deeper into the SVM and LDA and LR \
        algorithms. It is very likely that configuration beyond the default \
        may yield even more accurate models."
    )

    st.title("Algorithm Tuning")
    st.markdown(
        "In this section we investigate tuning the parameters for three \
        algorithms that show promise from the spot-checking in the previous \
        section: LR, LDA and SVM."
    )

    st.info("### Tuning hyper-parameters - SVC estimator")

    st.image("outputs/nb6/model_training_accuracy_svc.png")

    st.info("### Tuning the hyper-parameters - k-NN hyperparameters")
    st.markdown(
        "For your standard k-NN implementation, there are two primary \
        hyperparameters that you’ll want to tune:"
    )

    st.markdown("* The number of neighbors k.")
    st.markdown("* The distance metric/similarity function.")

    st.markdown(
        "Both of these values can dramatically affect the accuracy of your \
        k-NN classifier. Grid object is ready to do 10-fold cross validation \
        on a KNN model using classification accuracy as the evaluation metric"
    )
    st.markdown(
        "In addition, there is a parameter grid to repeat the 10-fold cross \
        validation process 30 times"
    )
    st.markdown(
        "Each time, the n_neighbors parameter should be given a different \
        value from the list"
    )
    st.markdown("We cant give GridSearchCV just a list")
    st.markdown("Weve to specify n_neighbors should take on 1 through 30")
    st.markdown(
        "You can set n_jobs = -1 to run computations in parallel (if supported\
        by your computer and OS)"
    )

    st.image("outputs/nb6/model_training_accuracy_knn.png")

    st.info("### Finalize Model")

    st.image("outputs/nb6/final_model_accuracy.png")

    st.image("outputs/nb6/final_model_classification.png")

    st.success("## Summary")

    st.markdown(
        "Worked through a classification predictive modeling machine learning \
        problem from end-to-end using Python. Specifically, the steps covered \
        were:"
    )
    st.markdown("1. Problem Definition (Breast Cancer data).")
    st.markdown("2. Loading the Dataset.")
    st.markdown(
        "3. Analyze Data (same scale but different distributions of \
    data)."
    )
    st.markdown("     * Evaluate Algorithms (KNN looked good).")
    st.markdown(
        "     * Evaluate Algorithms with Standardization (KNN and SVM looked \
        good)."
    )
    st.markdown(
        "4. Algorithm Tuning (K=19 for KNN was good, SVM with an RBF kernel \
        and C=100 was best).."
    )
    st.markdown(
        "5. Finalize Model (use all training data and confirm using validation\
        dataset)"
    )

    st.title("Conclusion")

    st.markdown(
        "The purpose of this study was to create a model that can predict if a\
        tumor is benign or malignant based on the given features. The SVC \
        provided the best accuracy score with the given hyper-parameters. We \
        also observed that scaling the data improved the performance of the \
        algorithm for the sample dataset. This Model validates our hypothesis \
        made in the business case and also is above the accuracy the client \
        wanted with 94% being higher than 90%. We have met the business \
        requirements with the dashboard navigation and by showing and \
        explaining the model process with access given to the client for the \
        finalized model."
    )
