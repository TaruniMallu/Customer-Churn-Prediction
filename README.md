
# Customer Churn Prediction

## 📌 Objective
The goal of this project is to predict customer churn for a subscription-based service using historical customer data.  
We aim to:
- Analyze factors like usage patterns, demographics, and subscription details.
- Build a predictive model to identify customers likely to leave.
- Derive insights into important factors driving churn.

## 📚 Dataset
The dataset includes features such as CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, and others.  
The target variable is `Exited`:
- 0: Customer stayed
- 1: Customer churned (left)

## 🛠 Steps to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Customer-Churn-Prediction.git
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open the notebook and execute all cells sequentially.

## 📈 Models Used
- Logistic Regression (baseline)
- Random Forest Classifier (final model)

## 🎯 Key Insights
- Random Forest performed significantly better at predicting churners compared to Logistic Regression.
- Important features influencing churn include Age, Balance, CreditScore, and IsActiveMember status.

## ✨ Final Notes
- Proper handling of missing values and encoding techniques were applied.
- Model evaluation was done using accuracy, precision, recall, and F1-score.
