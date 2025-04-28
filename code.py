
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit title
st.title('Customer Churn Prediction')

# Load dataset
df = pd.read_csv('Churn_Modelling.csv')

# Display Data Preview and Info
st.write("Data Preview:", df.head())
st.write("Data Info:", df.info())

# Check for missing values and handle them
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Handle numeric and categorical columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = num_imputer.fit_transform(df[[col]])

for col in df.select_dtypes(include=['object']).columns:
    df[col] = cat_imputer.fit_transform(df[[col]]).ravel()

# Label encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Display churn distribution
st.write("Customer Churn Distribution:")
sns.countplot(x='Exited', data=df)
plt.xticks([0, 1], ['Stayed', 'Churned'])
plt.title('Customer Churn Distribution')
st.pyplot()

# Define features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Display classification report
st.write("Logistic Regression Classification Report:")
st.text(classification_report(y_test, y_pred_lr))

st.write("Random Forest Classification Report:")
st.text(classification_report(y_test, y_pred_rf))

# Feature importance visualization
importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

st.write("Feature Importance:")
feat_imp.plot(kind='bar')
plt.title('Feature Importance')
st.pyplot()
