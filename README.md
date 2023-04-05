![project](project image)

## Introduction

Breast cancer (BC) is one of the most common cancers among women worldwide, representing the majority of new cancer cases and cancer-related deaths according to global statistics, making it a significant public health problem in today’s society.

The early diagnosis of BC can improve the prognosis and chance of survival significantly, as it can promote timely clinical treatment to patients. Further accurate classification of benign tumors can prevent patients undergoing unnecessary treatments. Thus, the correct diagnosis of BC and classification of patients into malignant or benign groups is the subject of much research. Because of its unique advantages in critical features detection from complex BC datasets, machine learning (ML) is widely recognized as the methodology of choice in BC pattern classification and forecast modelling.

Classification and data mining methods are an effective way to classify data. Especially in medical field, where those methods are widely used in diagnosis and analysis to make decisions.

Symptoms for specific types of cancer
According to the American Cancer Society (ACS)Trusted Source, the most common sign of breast cancer is a new lump or mass in the breast. People should become familiar with the typical look and feel of their breasts to detect any changes early on.

Breast cancer can develop in males and females, but due to differencesTrusted Source in breast tissue, the disease is much less common in males.

Below, we outline some early indications of breast cancer. We also describe the various types and treatment options. Finally, we look into some benign conditions people may mistake for breast cancer.

Benign breast conditions
Several benign breast conditions can cause symptoms that resemble those of cancer. Some of these issues require treatment, while others go away on their own.

Though these conditions are benign, they can cause:

discomfort or pain swelling lumps Some common benign breast conditions include:

Cysts: These are fluid-filled sacs that can form in many parts of the body, including the breasts.

Mastitis: This is inflammation (swelling) in the breast that is usually from an infection.

Hyperplasia: This is an overgrowth of cells, particularly in the milk ducts or lobules inside the breast.

Sclerosing adenosis: This is a condition in which lobules enlarge.

Intraductal papillomas: These are benign wart-like tumors that grow within the milk ducts of the breast.

Fibroadenoma: These are common breast tumors that develop when an overgrowth of fibrous or glandular tissue forms around a lobule.

Radial scar: Also called complex sclerosing lesions, these are a core of connective tissue that can resemble breast cancer on a mammogram.

Fat necrosis: This develops following an injury to fatty breast tissue, as can happen following surgery, radiation, or injury to the breast.

Phyllodes tumors: These are fast-growing but typically painless tumors that start in the connective tissue of the breast. Some can be cancerous.

![Breast Cancer](/assests/images/breast-cancer.webp)

## Gitpod Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In Gitpod, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.


## Dataset Content

* The dataset is on kaggel it is available for use the License is CC0: Public Domain so we can use it, it has 32 columns and over 500 rows of data.

data image to go here


## Business Requirements

Business Requirement 1
o	The client is interested in ways to create a model that can predict if a tumour is benign or malignant based on the given features.

Business Requirement 2
o	The client is interested in predicting whether a given tumour has cancer or is benign with a high degree of accuracy.



## Hypothesis and how to validate?

* We beliveve that by looking at differant features we can differentiate between benign or malignant tumours.

o This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection and hyper parameter selection. The goal is to classify whether the breast cancer is benign or malignant. To achieve this i have used machine learning classification methods to fit a function that can predict the discrete class of new input.


## The rationale to map the business requirements to the Data Visualizations and ML tasks

Business Requirement 1: Data Visualizations
1. Data Exploration
2. Categorical Data
3. Feature Scaling

Business Requirement 2: Machine Learing Classification Analysis
1. Model Selection
2. Finalize Model
3. Test Model
4. Conclusion

## ML Business Case

This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection and hyper parameter selection. The goal is to classify whether the breast cancer is benign or malignant. To achieve this i have used machine learning classification methods to fit a function that can predict the discrete class of new input. 
In our dataset we have the outcome variable or Dependent variable i.e Y having only two set of values, either M (Malign) or B(Benign). So we will use Classification algorithm of supervised learning.

We have different types of classification algorithms we will be using :-
1. Logistic Regression
2. Nearest Neighbor
3. Support Vector Machines
4. Decision Tree Algorithm
5. Random Forest Classification
We will use sklearn library to import all the methods of classification algorithms.
We will use LogisticRegression method of model selection to use Logistic Regression Algorithm.

The model success metrics are 95% or above on the test set.

![Business Case](/assests/documents/business-case.docx)

## Dashboard Design

Page 1: Quick project summary
Quick project summary
Project Terms & Jargon
Describe Project Dataset
State Business Requirements

Page 2: Data Visualization

Page 4: Hypothesis

Page 3: Model Selection

Page 4: Finalize Model/Conculsion


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people that provided support through this project.

