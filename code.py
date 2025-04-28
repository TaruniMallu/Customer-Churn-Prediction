#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[3]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[5]:


import seaborn as sns


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv('Churn_Modelling.csv')


# In[9]:


st.write(df.head())  # to show the top rows of the dataframe
st.write(df.info())  # to display information about the dataframe



# In[10]:



# In[11]:


num_imputer = SimpleImputer(strategy='median')


# In[12]:


cat_imputer = SimpleImputer(strategy='most_frequent')


# In[13]:


for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = num_imputer.fit_transform(df[[col]])


# In[16]:


for col in df.select_dtypes(include=['object']).columns:
    df[col] = cat_imputer.fit_transform(df[[col]]).ravel()



# In[18]:


### for categorcial columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# In[20]:


print(df.columns)


# In[22]:


sns.countplot(x='Exited', data=df)
plt.xticks([0, 1], ['Stayed', 'Churned'])
plt.title('Customer Churn Distribution')
plt.show()


# In[23]:


X = df.drop('Exited', axis=1)
y = df['Exited']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# In[26]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[27]:


print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))


# In[28]:


importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar')
plt.title('Feature Importance')
plt.show()


# In[ ]:




