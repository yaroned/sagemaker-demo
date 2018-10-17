#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction with XGBoost
# _**Using Gradient Boosted Trees to Predict Mobile Customer Departure**_
# 
# ---
# 
# ---
# 
# ## Contents
# 
# 1. [Background](#Background)
# 1. [Setup](#Setup)
# 1. [Data](#Data)
# 1. [Train](#Train)
# 1. [Host](#Host-The-Model)
#   1. [Evaluate](#Evaluate)
# 1. [Extensions](#Extensions)
# 
# ---
# 
# ## Background
# 
# _This notebook has been adapted from an [AWS blog post](https://aws.amazon.com/blogs/ai/predicting-customer-churn-with-amazon-machine-learning/)_
# 
# Losing customers is costly for any business.  Identifying unhappy customers early on gives you a chance to offer them incentives to stay.  This notebook describes using machine learning (ML) for the automated identification of unhappy customers, also known as customer churn prediction. ML models rarely give perfect predictions though, so this notebook is also about how to incorporate the relative costs of prediction mistakes when determining the financial outcome of using ML.
# 
# We use an example of churn that is familiar to all of us–leaving a mobile phone operator.  Seems like I can always find fault with my provider du jour! And if my provider knows that I’m thinking of leaving, it can offer timely incentives–I can always use a phone upgrade or perhaps have a new feature activated–and I might just stick around. Incentives are often much more cost effective than losing and reacquiring a customer.
# 
# ---
# 
# ## Setup
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Run the below cell to Specify:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting. In this case we'll use Sagemaker's default bucket.
# - The IAM role arn used to give training and hosting access to your data.

# In[1]:


# Define IAM role
import boto3
import re
from sagemaker import get_execution_role
import sagemaker

role = get_execution_role()
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/DEMO-xgboost-churn'
print(f'Using bucket: {bucket}')


# Next, we'll import the Python libraries we'll need for the remainder of the exercise.
# In particular note:
# 1. [Pandas](https://pandas.pydata.org/) - A Python library for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
# 2. [Numpy](http://www.numpy.org/) - A Python library that adds support for large, multi-dimensional arrays and matrices.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display
import sagemaker


# ---
# ## Data
# 
# Mobile operators have historical records on which customers ultimately ended up churning and which continued using the service. We can use this historical information to construct an ML model of one mobile operator’s churn using a process called training. After training the model, we can pass the profile information of an arbitrary customer (the same profile information that we used to train the model) to the model, and have the model predict whether this customer is going to churn. Of course, we expect the model to make mistakes–after all, predicting the future is tricky business! But I’ll also show how to deal with prediction errors.
# 
# The dataset we use is publicly available and was mentioned in the book [Discovering Knowledge in Data](https://www.amazon.com/dp/0470908742/) by Daniel T. Larose. It is attributed by the author to the University of California Irvine Repository of Machine Learning Datasets.  Let's download and read that dataset in now:

# In[4]:


get_ipython().system('wget http://dataminingconsultant.com/DKD2e_data_sets.zip')
get_ipython().system('unzip -o DKD2e_data_sets.zip')
get_ipython().system("ls -l 'Data sets/churn.txt'")


# ---
# Load into a Pandas Dataframe:  
# churn.txt is a CSV file, load it into a Pandas Dataframe by using the relevant [Pandas method for reading input](https://pandas.pydata.org/pandas-docs/stable/api.html).

# ### Update the code cell

# In[6]:


churn = pd.read_csv('./Data sets/churn.txt') # Use pandas to read in CSV format
pd.set_option('display.max_columns', 500)
churn.head(5) # Print first five records


# By modern standards, it’s a relatively small dataset, with only 3,333 records, where each record uses 21 attributes to describe the profile of a customer of an unknown US mobile operator. The attributes are:
# 
# - `State`: the US state in which the customer resides, indicated by a two-letter abbreviation; for example, OH or NJ
# - `Account Length`: the number of days that this account has been active
# - `Area Code`: the three-digit area code of the corresponding customer’s phone number
# - `Phone`: the remaining seven-digit phone number
# - `Int’l Plan`: whether the customer has an international calling plan: yes/no
# - `VMail Plan`: whether the customer has a voice mail feature: yes/no
# - `VMail Message`: presumably the average number of voice mail messages per month
# - `Day Mins`: the total number of calling minutes used during the day
# - `Day Calls`: the total number of calls placed during the day
# - `Day Charge`: the billed cost of daytime calls
# - `Eve Mins, Eve Calls, Eve Charge`: the billed cost for calls placed during the evening
# - `Night Mins`, `Night Calls`, `Night Charge`: the billed cost for calls placed during nighttime
# - `Intl Mins`, `Intl Calls`, `Intl Charge`: the billed cost for international calls
# - `CustServ Calls`: the number of calls placed to Customer Service
# - `Churn?`: whether the customer left the service: true/false
# 
# The last attribute, `Churn?`, is known as the target attribute–the attribute that we want the ML model to predict.  Because the target attribute is binary, our model will be performing binary prediction, also known as binary classification.

# ---
# As Data scientists, before plugging our data into an ML algorithm, we must first explore our data to understand:
# 1. Which features are categorical and which are numeric. Which features categorical Numpy wrongly classified as numeric.
# 2. Each feature values distribution and cardinality.
# 3. Which features we should drop from the dataset because we can easily see they would not contribute to the model, or because they are highly correlated with other features.
# 
# You should really explore all features, but in this notebook, we decided to focus on specific features we know are interesting.  
# Let's start exploring:

# In[7]:


churn.State.value_counts(sort=True).plot.pie()
plt.show()


# Note `State` appears to be quite evenly distributed

# In[8]:


def crosstab(column):
    display(pd.crosstab(index=churn[column], columns='% observations', normalize='columns'))
crosstab('Phone')


# `Phone` takes on too many unique values to be of any practical use. It's possible parsing out the prefix could have some value, but without more context on how these are allocated, we should avoid using it.  
# Let's drop the `Phone` column:

# In[9]:


churn = churn.drop('Phone', axis=1)


# In[10]:


churn['Churn?'].value_counts(sort=True).plot.pie(autopct='%1.f%%')
plt.show()


# We can see only 14% of customers churned, so there is some class imabalance, but nothing extreme.

# In[11]:


# Histograms for each numeric features
#display(churn.describe())
get_ipython().run_line_magic('matplotlib', 'inline')
hist = churn.hist(bins=30, sharey=True, figsize=(10, 10))


# Most of the numeric features are surprisingly nicely distributed, with many showing bell-like gaussianity.  `VMail Message` being a notable exception (and `Area Code` showing up as a feature we should convert to non-numeric).  
# Let's convert `Area Code` to non-numeric:

# In[12]:


churn['Area Code'] = churn['Area Code'].astype(object)


# Next let's look at the relationship between each of the features and our target variable.

# In[13]:


for column in churn.select_dtypes(include=['object']).columns:
    if column != 'Churn?':
        display(pd.crosstab(index=churn[column], columns=churn['Churn?'], normalize='columns'))

for column in churn.select_dtypes(exclude=['object']).columns:
    print(column)
    hist = churn[[column, 'Churn?']].hist(by='Churn?', bins=30)
    plt.show()


# Interestingly we see that churners appear:
# - Fairly evenly distributed geographically
# - More likely to have an international plan
# - Less likely to have a voicemail plan
# - To exhibit some bimodality in daily minutes (either higher or lower than the average for non-churners)
# - To have a larger number of customer service calls (which makes sense as we'd expect customers who experience lots of problems may be more likely to churn)
# 
# In addition, we see that churners take on very similar distributions for features like `Day Mins` and `Day Charge`.  That's not surprising as we'd expect minutes spent talking to correlate with charges.  Let's dig deeper into the relationships between our features.

# In[14]:


# Show the correlation between each possible features pair as a scatter plot
pd.plotting.scatter_matrix(churn, figsize=(16, 16))
plt.show()


# We see several features that essentially have 100% correlation with one another.  Including these feature pairs in some machine learning algorithms can create catastrophic problems, while in others it will only introduce minor redundancy and bias.  Let's remove one feature from each of the highly correlated pairs: Day Charge from the pair with Day Mins, Night Charge from the pair with Night Mins, Intl Charge from the pair with Intl Mins:

# In[15]:


churn = churn.drop(['Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge'], axis=1)


# Now that we've cleaned up our dataset, let's determine which algorithm to use.  As mentioned above, there appear to be some variables where both high and low (but not intermediate) values are predictive of churn.  In order to accommodate this in an algorithm like linear regression, we'd need to generate polynomial (or bucketed) terms.  Instead, let's attempt to model this problem using gradient boosted trees.  Amazon SageMaker provides an XGBoost container that we can use to train in a managed, distributed setting, and then host as a real-time prediction endpoint.  XGBoost uses gradient boosted trees which naturally account for non-linear relationships between features and the target variable, as well as accommodating complex interactions between features.
# 
# Amazon SageMaker XGBoost can train on data in either a CSV or LibSVM format.  For this example, we'll stick with CSV.  It should:
# - Have the predictor variable in the first column
# - Not have a header row
# 
# But first, let's convert our categorical features into numeric features, as the algorithm expects numerical values only.
# This means adding each possible catagorical value as a new colume.
# Replace `REPLACE_ME` with the pandas method that Converts categorical variable into dummy/indicator variables.

# ### Update the code cell

# In[16]:


model_data = pd.get_dummies(churn) # Convert categorical variable into indicator variables


# In[17]:


# creates a new dataframe with the 'Churn?_True.' colume being the first column, the trailed by 
# all other columes except 'Churn?_False.' and 'Churn?_True.'.
model_data = pd.concat([model_data['Churn?_True.'], model_data.drop(['Churn?_False.', 'Churn?_True.'], axis=1)], axis=1)


# In[18]:


model_data.head(5) # notice we now have many more columes generated to reperesent all categorical values


# And now let's split the data into training, validation, and test sets.  This will help prevent us from overfitting the model, and allow us to test the models accuracy on data it hasn't already seen.

# In[19]:


train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
train_data.to_csv('train.csv', header=False, index=False)
validation_data.to_csv('validation.csv', header=False, index=False)


# Now we'll upload these files to S3.

# In[20]:


boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')


# ---
# ## Train
# Amazon SageMaker provides several [built-in machine learning algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) that you can use for a variety of problem types. Amazon SageMaker algorithms are packaged as Docker images. This gives you the flexibility to use almost any algorithm code with Amazon SageMaker, regardless of implementation language, dependent libraries, frameworks, and so on.
# Each built-in algorithm packed in its own Docker image found in the [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html). We'll specify the relevant container image for each training job.  
# 
# Moving onto training, first we'll need to specify the image for the XGBoost algorithm container.
# 
# In the next cell replace **algorithm_name** with the relevant algorithm name from the "Training Image and Inference Image Registry Path" colume in this [table](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html). Enter the algoritm name in lower case and without the trailing colon.

# ### Update the code cell

# In[23]:


from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'xgboost') # Choose the relevant algorithm used in this notebook


# [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/latest/)  is an open source library for training and deploying machine-learned models on Amazon SageMaker. 
# The SDK provides several high-level abstractions for working with Amazon SageMaker. These are:
# 
# * [Estimators](https://sagemaker.readthedocs.io/en/latest/estimators.html): Encapsulate training on SageMaker.
# * [Models](https://sagemaker.readthedocs.io/en/latest/model.html): Encapsulate a trained ML model. Can be deployed to an endpoint.
# * [Predictors](https://sagemaker.readthedocs.io/en/latest/predictors.html): Provide real-time inference and transformation using Python data-types against a SageMaker endpoint.
# * [Session](https://sagemaker.readthedocs.io/en/latest/session.html): Provides a collection of methods for working with SageMaker resources.
# 
# We'll start by creating the [xgboost Estimator](https://sagemaker.readthedocs.io/en/latest/estimators.html). The mandatory paramters are: image_name (str), role (str), sagemaker_session (session), train_instance_type (str), and train_instance_count (int).
# 
# For this training job, provide these parameters: **image_name = container, role=role, train_instance_count = 1, train_instance_type = 'ml.m4.xlarge', sagemaker_session = session**.  

# ### Update the code cell

# In[24]:


# Creating the SageMaker Estimator object
xgb = sagemaker.estimator.Estimator(image_name = container,
                                    role = role, 
                                    train_instance_count = 1, 
                                    train_instance_type ='ml.m4.xlarge',
                                    output_path = 's3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session = session)


# An ML algorithm is configured and tuned with specific hyperparameters, the hyperparameters changes the way the algorithm works ([what is a hyperpramaeter?](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning%29)).
# 
# The XGBoost hyperparamaters are described in the [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/parameter.html).
# 
# For this example, the required hyperparameters for the XGBoost algorithm are:
# 
# * `objective` - Specifies the learning task and the corresponding learning objective. please use **binary:logistic** for binary classification task.  
# * `num_round` - Controls the number of boosting rounds. This is essentially the subsequent models that are trained using the residuals of previous iterations. More rounds should produce a better fit on the training data, but can be computationally expensive or lead to overfitting.
# 
# A few other key hyperparameters are:
# * `max_depth` - Controls how deep each tree within the algorithm can be built. Deeper trees can lead to better fit, but are more computationally expensive and can lead to overfitting. There is typically some trade-off in model performance that needs to be explored between a large number of shallow trees and a smaller number of deeper trees.
# * `subsample` controls - Sampling of the training data. This technique can help reduce overfitting, but setting it too low can also starve the model of data.
# * `eta` - Controls how aggressive each round of boosting is. Larger values lead to more conservative boosting.
# * `gamma` - Controls how aggressively trees are grown. Larger values lead to more conservative models.
# 
# Use the [xgb.set_hyperparameters](https://sagemaker.readthedocs.io/en/latest/estimators.html) to set the hyperparameters value you choose after reading the documentation.

# ### Update the code cell

# In[25]:


# set the hyperparameters
xgb.set_hyperparameters(objective="binary:logistic",
                        num_round=1
)


# Since we're training with CSV file format, we'll create [`s3_input`](https://sagemaker.readthedocs.io/en/latest/session.html?highlight=sagemaker.session.s3_input) objects that our training function can use as a pointer to the files type and location in S3.
# 
# Run the following for the training data input and the validation data sets:

# In[26]:


# Configuring the data inputs
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation'.format(bucket, prefix), content_type='csv')


# Finally we are ready to train.
# To train use the [xgb.fit()](https://sagemaker.readthedocs.io/en/latest/estimators.html) function.  

# In[27]:


# Traing the model
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


# Pay attention to the final validation-error. A lower value is better.

# ---
# ## Host The Model
# 
# Now that we've trained the algorithm, let's create a model and deploy it to a hosted endpoint.
# 
# We'll do this using [estimator](https://sagemaker.readthedocs.io/en/latest/estimators.html) `deploy()` method.
# 
# Provide these parameters: **initial_instance_count = 1** and **instance_type = 'ml.m4.xlarge'**

# ### Update the code cell

# In[28]:


# Deploy the model
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')


# ---
# ### Evaluate
# 
# Now that we have a hosted endpoint running, we can make real-time predictions from our model very easily, simply by making an http POST request.  But first, we'll need to setup serializers and deserializers for passing our `test_data` NumPy arrays to the model behind the endpoint.

# In[35]:


from sagemaker.predictor import csv_serializer
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
xgb_predictor.deserializer = None


# Now, we'll use a simple function to:
# 1. Loop over our test dataset
# 1. Split it into mini-batches of rows 
# 1. Convert those mini-batchs to CSV string payloads
# 1. Retrieve mini-batch predictions by invoking the XGBoost endpoint
# 1. Collect predictions and convert from the CSV output, our model provides, into a NumPy array

# ### Update the code cell

# In[39]:


def predict(data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        current_predictions = xgb_predictor.predict(array).decode('utf-8')# Use the xgb_predictor predict function on the array
        predictions = ','.join([predictions, current_predictions])

    return np.fromstring(predictions[1:], sep=',') # csv to numpy array

predictions = predict(test_data.as_matrix()[:, 1:])

assert(len(predictions)==334) # Just checking


# There are many ways to compare the performance of a machine learning model, but let's start by simply by comparing actual to predicted values.  In this case, we're simply predicting whether the customer churned (`1`) or not (`0`), which produces a simple [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).

# In[41]:


confusion_matrix = pd.crosstab(index=test_data.iloc[:, 0], columns=np.round(predictions), rownames=['actual'], colnames=['predictions'], margins='true')
display(confusion_matrix)


# _Note, due to randomized elements of the algorithm, your results may differ slightly._
# 
# Of the 48 churners, we've correctly predicted ~39 of them (_True Positives (TP)_). And, we incorrectly predicted 4 customers would churn who then ended up not doing so (_False Positives (FP)_).  There are also ~9 customers who ended up churning, that we predicted would not (_False Negatives (FN)_).
# 
# Now let's calculate the accuracy, precision and recall (these are common mesurments that assists in evaluating the quality of our model and also compare between differnt models):
# 
# _Accuracy_: Overall, how often is the classifier correct? (TP+TN)/total
# 
# _Precision_: When it predicts yes, how often is it correct? TP/(predicted yes)
# 
# _Recall_: When it's actually yes, how often does it predict yes? TP/(actual yes)

# ### Update the code cell

# In[42]:


total = confusion_matrix['All'][2]
TP = [confusion_matrix[1][1]]
TN = [confusion_matrix[0][0]]
accuracy = (TP+TN)/total
precision = TP/confusion_matrix[1]['All']
recall = TP/confusion_matrix['All'][1]
print ('Accuracy: {}'.format(accuracy))
print ('Precision: {}'.format(precision))
print ('Recall: {}'.format(recall))


# An important point here is that because of the `np.round()` function above we are using a simple threshold (or cutoff) of 0.5.  Our predictions from `xgboost` come out as continuous values between 0 and 1 and we force them into the binary classes that we began with.  However, because a customer that churns is expected to cost the company more than proactively trying to retain a customer who we think might churn, we should consider adjusting this cutoff.  That will almost certainly increase the number of false positives, but it can also be expected to increase the number of true positives and reduce the number of false negatives.
# 

# ---
# ### (Must) Clean-up
# 
# If you're ready to be done with this notebook, please run the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.

# In[43]:


sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)


# ---
# ### (Optional) Bonus 1 - Tune the model using different/additional hyperparameters
# Use the [Estimator.set_hyperparameters](https://sagemaker.readthedocs.io/en/latest/estimators.html) and explore other hyperparameters, check out the [recommanded values](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html) for each hyper parameter.
# 
# Try to get a lower validation-error.

# In[ ]:


#Bonus 1 code here


# ---
# ### (Optional) Bonus 2 - Use SageMaker automatic model tuning tune the model using different/additional hyperparameters
# See the [blog post on Automatic model tuning](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/)
# Call auto model tuning to achieve the lowest validation-error. 
# Sit back, relax and appreciate how auto model tuning is far easier than manual tuning.

# In[ ]:


#Bonus 2 code here

