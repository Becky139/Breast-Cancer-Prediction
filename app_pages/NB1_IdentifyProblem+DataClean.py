#!/usr/bin/env python
# coding: utf-8

# # NB1- Identify The Problem & Data Cleaning

# ## Objectives
# 
# * Write here your notebook objective, for example, "Fetch data from Kaggle and save as raw data", or "engineer features for modelling"
# 
# ## Inputs
# 
# * Write here which data or information you need to run the notebook 
# 
# ## Outputs
# 
# * Write here which files, code or artefacts you generate by the end of the notebook 
# 
# ## Additional Comments
# 
# * In case you have any additional comments that don't fit in the previous bullets, please state them here. 

# ---

# # Change working directory

# * We are assuming you will store the notebooks in a subfolder, therefore when running the notebook in the editor, you will need to change the working directory

# We need to change the working directory from its current folder to its parent folder
# * We access the current directory with os.getcwd()

# In[1]:


import os
current_dir = os.getcwd()
current_dir


# We want to make the parent of the current directory the new current directory
# * os.path.dirname() gets the parent directory
# * os.chir() defines the new current directory

# In[2]:


os.chdir(os.path.dirname(current_dir))
print("You set a new current directory")


# Confirm the new current directory

# In[3]:


current_dir = os.getcwd()
current_dir


# ---

# # Using Predictive Analysis To Predict Diagnosis of a Breast Tumor 
# 
# ## 1. Identify the problem
# Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a results of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.
# 
# ### 1.1 Expected outcome
# Given breast cancer results from breast fine needle aspiration (FNA) test (is a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore or swelling) with a fine needle similar to a blood sample needle). Since this build a model that can classify a breast cancer tumor using two training classification:
# * 1= Malignant (Cancerous) - Present
# * 0= Benign (Not Cancerous) -Absent
# 
# ### 1.2 Objective 
# Since the labels in the data are discrete, the predication falls into two categories, (i.e. Malignant or benign). In machine learning this is a classification problem. 
#         
# > *Thus, the goal is to classify whether the breast cancer is benign or malignant and predict the recurrence and non-recurrence of malignant cases after a certain period.  To achieve this we have used machine learning classification methods to fit a function that can predict the discrete class of new input.*
# 
# ### 1.3 Identify data sources
# The [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) datasets is available machine learning repository maintained by the University of California, Irvine. The dataset contains **569 samples of malignant and benign tumor cells**. 
# * The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively. 
# * The columns 3-32 contain 30 real-value features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant. 
# 
# #### Getting Started: Load libraries and set options 

# In[4]:


#load libraries
import numpy as np         # linear algebra
import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the file "data.csv" and print the contents.
#!cat data/data.csv
data = pd.read_csv('inputs/data/data.csv', index_col=False,)


# #### Load Dataset
# 
# First, load the supplied CSV file using additional options in the Pandas read_csv function. 

# #### Inspecting the data
# The first step is to visually inspect the new data set. There are multiple ways to achieve this:
# * The easiest being to request the first few records using the DataFrame data.head()* method. By default, “data.head()” returns the first 5 rows from the DataFrame object df (excluding the header row). 
# * Alternatively, one can also use “df.tail()” to return the five rows of the data frame. 
# * For both head and  tail methods, there is an option to specify the number of records by including the required number in between the parentheses when calling either method.Inspecting the data

# In[5]:


data.head(5)


# You can check the number of cases, as well as the number of fields, using the shape method, as shown below.

# In[6]:


# Id column is redundant and not useful, we want to drop it
data.drop('id', axis =1, inplace=True)
#data.drop('Unnamed: 0', axis=1, inplace=True)
data.head(2)


# In[7]:


data.shape


# In the result displayed, you can see the data has 569 records, each with 32 columns.
# 
# The **“info()”** method provides a concise summary of the data; from the output, it provides the type of data in each column, the number of non-null values in each column, and how much memory the data frame is using.
# 
# The method **get_dtype_counts()** will return the number of columns of each type in a DataFrame:

# In[8]:


# Review data types with "info()".
data.info()


# In[9]:


# Review number of columns of each data type in a DataFrame:
data.dtypes.value_counts()


# From the above results, from the 32, variables,column id number 1 is an integer, diagnosis 569 non-null object. and rest are float. More on [python variables](https://www.tutorialspoint.com/python/python_variable_types.htm)

# In[10]:


#check for missing variables
data.isnull().any()


# In[11]:


data.diagnosis.unique()


# From the results above, diagnosis is a categorical variable, because it represents a fix number of possible values (i.e, Malignant, of Benign. The machine learning algorithms wants numbers, and not strings, as their inputs so we need some method of coding to convert them.
# 
# 

# ---

# # Push files to Repo

# In case you don't need to push files to Repo, you may replace this section with "Conclusions and Next Steps" and state your conclusions and next steps.

# In[12]:


#save the cleaner version of dataframe for future analyis
data.to_csv('outputs/data/clean_data.csv')


# > ### Now that we have a good intuitive sense of the data, Next step involves taking a closer look at attributes and data values. In nootebook title :NB_NB2_ExploratoryDataAnalys, we will explore the data further
