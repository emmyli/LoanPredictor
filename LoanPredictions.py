
# coding: utf-8

# ### Introduction
# 
# We will be predicting the loan status for applicants with a few different models. In particular, we will be using Logistic Regression, Decision Tree, Random Forest, and XGBoost to determine the loan status.
# The training data set consists of 614 applicants with 11 different variables, including Gender, Dependents, and Education. 
# 

# ### Importing the data set and libraries

# In[1]:


# Import tools for data visualization and manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Plotting defaults
get_ipython().magic(u'matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'

# Read in the data in a dataframe
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# ### Data exploration and visualization

# A brief description of each variable in the dataset is provided below.

# In[2]:


# VARIABLE                  DESCRIPTION
# Loan_ID                   Unique Loan ID
# Gender                    Male(1)/ Female(0)
# Married Applicant         Married (Y/N)
# Dependents                Number of dependents
# Education Applicant       Education (Graduate/ Under Graduate) 
# Self_Employed             Self employed (Y/N)
# ApplicantIncome           Applicant income
# CoapplicantIncome         Coapplicant income
# LoanAmount                Loan amount in thousands
# Loan_Amount_Term          Term of loan in months
# Credit_History            Credit history meets guidelines
# Property_Area             Urban/ Semi Urban/ Rural
# Loan_Status               Loan approved (Y/N)


# #### Overview

# In[3]:


# Quick summary of the data
train.info()
train.head(10)


# Note: Based on the summary of the training dataset, there are a number of fields with missing values for certain variables that will need to be addressed before we model and predict the outcomes.

# In[4]:


# Summary of numerical fields
train.describe()


# ##### Frequency Distribution

# In[5]:


# Frequency distribution of Property Area
train['Property_Area'].value_counts()


# In[6]:


# Frequency distribution of Credit History
train['Credit_History'].value_counts()


# #### Distribution Analysis

# In[7]:


# Applicant income distribution analysis - Histogram
train['ApplicantIncome'].hist(bins=50)
plt.xlabel('Applicant Income'); plt.ylabel('Count'); 


# In[8]:


# Applicant income distribution analysis - Boxplot
train.boxplot(column='ApplicantIncome')
plt.ylabel('Value'); 


# In[9]:


train.boxplot(column='ApplicantIncome', by = 'Education')
plt.ylabel('Count'); 
plt.rcParams['font.size'] = 7


# In[10]:


train.boxplot(column='ApplicantIncome', by = 'Gender')
plt.ylabel('Count'); 
plt.rcParams['font.size'] = 7


# Note: Based on the summary of the training dataset, there are a number of extreme values that will need to be addressed before we model and predict the outcomes.

# In[11]:


# Loan amount distribution analysis
train['LoanAmount'].hist(bins=50)
plt.xlabel('Loan Amount'); plt.ylabel('Count'); 


# ### Data scrubbing and wrangling

# #### Missing Values

# In[12]:


# Check for missing values in the dataset
train.apply(lambda x: sum(x.isnull()),axis=0) 


# In[13]:


# Check the frequency distribution for Gender variable
train['Gender'].value_counts()


# Since there is a ~82% chance that the missing value is '1' for the Self_employed variable, we will assume that the missing 32 values are '1'.

# In[14]:


# Replace the missing values for Gender with '1'
train['Gender'].fillna(1,inplace=True)


# Similarly, we can replace the missing values with the mode for the other variables.

# In[15]:


# Replace the missing values with the mode
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[16]:


# Fill in the missing values of Loan Amount with the mean
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)


# ##### Final Check

# In[17]:


# Re-Check for missing values in the dataset
train.apply(lambda x: sum(x.isnull()),axis=0) 


# #### Extreme Values

# Since an extreme loan can be possible due to the requirements of the applicant, we will not treat the extreme values as outliers. Instead, we can apply a log transformation to reduce the effects of the extreme values.

# In[18]:


# Extreme values of Loan Amount
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
plt.xlabel('Loan Amount'); plt.ylabel('Count'); 


# Some applicants might have extreme income values that can be justified by the income value of their co-applicant, so it makes sense to combine the ApplicantIncome and Co-applicantIncome when examining extreme values.

# In[19]:


# Extreme values of Applicant Income
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['TotalIncome_log'] = np.log(train['TotalIncome'])
train['TotalIncome_log'].hist(bins=20) 
plt.xlabel('Total Income Amount'); plt.ylabel('Count'); 


# ### Building the Predictive Models

# Since sklearn requires that all of the inputs to be numeric, convert all our categorical variables into numeric by encoding the categories.

# In[20]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
train.dtypes 


# Before applying any models to the dataset, import the required modules, and define a generic classification function that will determine the accuracy and the and cross-validation scores for all of the models.

# In[21]:


# Import required models from scikit learn nd xgboost modules
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Generic function for making a classification model and accessing performance
def classification_model(model, data, predictors, outcome):
  # Fit the model:
  model.fit(data[predictors],data[outcome])
  
  # Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  # Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  # Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    # Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  # Fit the model again so that it can be refered outside the function
  model.fit(data[predictors],data[outcome]) 


# #### Logistic Regression

# In[22]:


# Define the outcome variable
outcome_var = 'Loan_Status'

# Apply the Logistic Regression model with only the Credit History variable
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, train,predictor_var,outcome_var)


# In[23]:


# We can try different combination of variables
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, train, predictor_var,outcome_var)


# The Credit History variable is a relatively dominating predictor since the additional variables seem to have little effect on the scores.

# #### Decision Tree

# In[24]:


model = DecisionTreeClassifier()
predictor_var = ['Credit_History']
classification_model(model, train, predictor_var, outcome_var)


# In[25]:


#We can try different combination of variables:
train.head()
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, train, predictor_var,outcome_var)


# #### Random Forest

# In[26]:


model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, train, predictor_var,outcome_var)


# In[27]:


# Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[28]:


model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['Credit_History', 'TotalIncome_log','LoanAmount_log', 'Dependents','Property_Area']
classification_model(model, train, predictor_var,outcome_var)


# #### XGBoost

# In[29]:


model = XGBClassifier()
predictor_var = ['Credit_History']
classification_model(model, train, predictor_var,outcome_var)


# In[38]:


# Increase the number of predicting variables 
model = XGBClassifier()
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
classification_model(model, train, predictor_var,outcome_var)


# In[39]:


# Increase the max_depth
model = XGBClassifier(max_depth=8)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
classification_model(model, train, predictor_var,outcome_var)


# In[49]:


# Decrease the max_depth
model = XGBClassifier(max_depth=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
classification_model(model, train, predictor_var,outcome_var)


# In[53]:


# Increase lambda
model = XGBClassifier(max_depth=1, reg_lambda=0.4)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
classification_model(model, train, predictor_var,outcome_var)

