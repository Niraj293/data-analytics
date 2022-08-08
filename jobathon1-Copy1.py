#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# Most organizations today rely on email campaigns for effective communication with users. Email communication is one of the popular ways to pitch products to users and build trustworthy relationships with them.
# 
# 
# Email campaigns contain different types of CTA (Call To Action). The ultimate goal of email campaigns is to maximize the Click Through Rate (CTR).
# 
# 
# CTR is a measure of success for email campaigns. The higher the click rate, the better your email marketing campaign is. CTR is calculated by the no. of users who clicked on at least one of the CTA divided by the total no. of users the email was delivered to.
# 
# 
# CTR =   No. of users who clicked on at least one of the CTA / No. of emails delivered
# 
# 
# CTR depends on multiple factors like design, content, personalization, etc. 
# 
# 
# * __How do you design the email content effectively?__ 
# * __What should your subject line look like?__
# * __What should be the length of the email?__
# * __Do you need images in your email template?__
# 
# As a part of the Data Science team, in this hackathon, you will build a smart system to predict the CTR for email campaigns and therefore identify the critical factors that will help the marketing team to maximize the CTR.

# # Table of content

# * __Step 1: Importing the Relevant Libraries__
#     
# * __Step 2: Data Inspection__
#     
# * __Step 3: Data Cleaning__
#     
# * __Step 4: Exploratory Data Analysis__
#     
# * __Step 5: Building Model__
#     

# ### Importing the relevant libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# #### Data Inspection

# In[2]:


train=pd.read_csv("train_F3fUq2S.csv")
test=pd.read_csv("test_Bk2wfZ3.csv")


# In[3]:


train.shape,test.shape


#  * __We have 1888 rows and 22 columns in Train set whereas Test set has 762 rows and 21 columns.__

# In[4]:


train.head()


# In[ ]:





# In[ ]:





# In[5]:


#ratio of null values in train data set
train.isnull().sum()/train.shape[0] *100


# In[6]:


#ratio of null values in test data set
test.isnull().sum()/test.shape[0] *100


# There are  Zero null values present in the both datasets

# In[7]:


#categorical features
categorical = train.select_dtypes(include =[np.object])
print("Categorical Features in Train Set:",categorical.shape[1])

#numerical features
numerical= train.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Train Set:",numerical.shape[1])


# In[8]:


numerical.shape


# In[9]:


#categorical features
categorical = test.select_dtypes(include =[np.object])
print("Categorical Features in Test Set:",categorical.shape[1])

#numerical features
numerical= test.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Test Set:",numerical.shape[1])


# In[89]:


#importing label encoder
from sklearn.preprocessing import LabelEncoder

#applying label encoding for train daset to convert object type data to int type.
LE=LabelEncoder()
for i in categorical:
    train[i]=LE.fit_transform(train[i])


# In[200]:


#applying label encoding for test dataset to convert object type data to int type.
LE=LabelEncoder()
for i in categorical:
    test[i]=LE.fit_transform(test[i])


# In[ ]:





# ### Explorator data analysis

# In[11]:


train.describe()


# #### Observation:
# Maximum click rate is __0.897959__ . 
# average click rate is __0.041888__ .
# minimum click rate is __0__.

# In[ ]:





# In[12]:


train.columns


# #### Q) How do you design the email content effectively ?

# In[160]:


sns.relplot(x='subject_len', y="click_rate",data=train);


# In[161]:


sns.relplot(x='body_len', y="click_rate",data=train);


# In[162]:


sns.relplot(x='mean', y="click_rate",data=train);


# In[ ]:





# In[140]:


train['is_emoticons'].value_counts()


# Obesrvation: emoticons are not used in maximum emails.

# In[ ]:





# In[138]:


#Visualizing number of images and click rate
sns.catplot(x="is_emoticons", y="click_rate", kind='strip',data=train);


# Observation: 
# - Emails without emoticons have better click rate.
# - Increasing the number of emoticons does not increase click rate

# In[156]:


# subplots for each of the category of Outlet_Size
sns.catplot(x="is_quote", y="click_rate", kind='swarm',data=train)


# - Observation: 2 or more than 2 quaote are not required for increasing click rate

# In[164]:


#Visualizing number of images and click rate
sns.catplot(x="is_image", y="click_rate", kind='strip',data=train);


# Questions:
# * __How do you design the email content effectively?__ 
# * __What should your subject line look like?__
# * __What should be the length of the email?__
# * __Do you need images in your email template?__
#     

# In[ ]:





# ##### Answer:
#     To design effective cotent of email:
#    - subject_len should be between 40 to 150
#    - body_len should br less than 3000
#    - mean paragraph len shpuld less than 200
#    - there is no need of emoticons.
#    - more than 1 quote is not neccesary.
#    - less than 3 images are satisfying for email.

# In[48]:


##Checking for correlation of output variable with other attributes:
corr_matrix=train.corr()


# In[49]:


#Graphical Visualization of correlation 
plt.figure(figsize=(18,10))
sns.heatmap(corr_matrix,annot=True)


# In[68]:


train['is_timer'].nunique()


# In[14]:





# In[93]:


n=1
plt.figure(figsize=(25, 20))
for i in numerical:
  plt.subplot(5,4,n)
  sns.distplot(train[i], color='green')
  n=n+1


# In[72]:


train.head()


# In[166]:


#Checking the skewnees and removing it
train.skew()


# In[49]:


train['is_personalised'].nunique()


# In[91]:


test.skew()


# In[97]:


from sklearn.preprocessing import StandardScaler

# 4 features are taken in consideration.
features = train[["subject_len","mean_paragraph_len","no_of_CTA","mean_CTA_len"]]

# the scaler object (model)
scaler = StandardScaler()

# fit and transform the data
scaled_data1 = scaler.fit_transform(features) 
scaled_data


# In[98]:


#for test data
features = test[["subject_len","mean_paragraph_len","no_of_CTA","mean_CTA_len"]]

# the scaler object (model)
scaler = StandardScaler()

# fit and transform the data
scaled_data2 = scaler.fit_transform(features) 
scaled_data


# In[ ]:





# In[99]:


from sklearn.preprocessing import power_transform
train=power_transform(train)
train=pd.DataFrame(df_new,columns=train.columns)


# In[100]:


train.skew()


# In[110]:


#Checking for outliers
n=1
plt.figure(figsize=(25, 20))
for i in numerical:
  plt.subplot(5,4,n)
  sns.boxplot(train[i])
  n=n+1


# In[ ]:





# In[ ]:





# In[ ]:





# # Model building

# In[167]:


# Seperate Features and Target
X= train.drop(columns = ['click_rate'], axis=1)
X=X.drop(columns=['is_timer'],axis=1)
X=X.drop(columns=['campaign_id'],axis=1)
y= train['click_rate']


# In[202]:


# droping features for test data
test= test.drop(columns = ['is_timer'], axis=1)
test=test.drop(columns=['campaign_id'],axis=1)


# In[168]:


# 20% data as validation set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=22)


# In[172]:


# Model Training and Validation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#Importing Boosting models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[173]:


#Creating function for Model Training
def models(model, x_train, x_test, y_train, y_test, score, rmse):
    #Fit the algorithm on the data
    model.fit(x_train, y_train)
    
    #Predict training set:
    y_pred = model.predict(x_test)
    
    score.append(model.score(x_train, y_train)*100)
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    print('Accuracy Score :: %0.2f' %(model.score(x_train, y_train)*100))
    print('R2 Score:', r2_score(y_test, y_pred))
    print('>>> Error >>>')
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


# In[174]:


# Instantiate all models without using gridsearch cv for best parameters means call all model with deafault parameters
lreg = LinearRegression()
knr = KNeighborsRegressor()
rr = Ridge()
lr = Lasso()
enr = ElasticNet()
svr = SVR()
dct = DecisionTreeRegressor()
rf = RandomForestRegressor()
gbr=GradientBoostingRegressor()
abr=AdaBoostRegressor()


# In[133]:


all_models={'Linear Regression': lreg,
            'K-Neighbors Regressor': knr,
            'Ridge Regression': rr,
            'Lasso Regression': lr,
            'Elastic Net': enr,
            'Support Vector Regression': svr,
            'Decision Tree Regression': dct,
            'Random Forest Regressor': rf,
            'Gradient Boosting Regression': gbr,
            'AdaBoost Regression': abr
           }


# In[175]:


score,rmse = [],[]
for i, j in all_models.items():
    print('-------------', i, '------------')
    models(j, X_train, X_test, y_train, y_test, score, rmse)


# ### Randomforest regressor is showing high R2 score with default parameter

# In[ ]:





# In[196]:


from sklearn.ensemble import RandomForestRegressor
m1=RandomForestRegressor(n_estimators=100, oob_score=True,n_jobs=2,random_state =3)
m1.fit(X_train,y_train)


# In[197]:


y_pre=m1.predict(X_test)


# In[198]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[203]:


y_pred=m1.predict(test)


# In[206]:


submission=pd.DataFrame()
submission['campaign_id'] = test['campaign_id']
submission['click rate'] = y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




