# Telco Customer Churn Prediction

## Overview
This repository contains a machine learning project aimed at predicting customer churn for a hypothetical telecommunications company based on Telco customer churn dataset. The dataset includes information about 7043 customers in California regarding their usage of home phone and internet services. The goal is to develop a predictive model that can identify which customers are likely to churn (leave the service) based on various demographic and service-related features.

## Dataset
- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No)
- **Partner**: Whether the customer has a partner (Yes, No)
- **Dependents**: Whether the customer has dependents (Yes, No)
- **Tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether the customer has phone service (Yes, No)
- **MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service)
- **InternetService**: Type of internet service (DSL, Fiber optic, No)
- **OnlineSecurity**: Whether the customer has online security (Yes, No, No internet service)
- **OnlineBackup**: Whether the customer has online backup (Yes, No, No internet service)
- **DeviceProtection**: Whether the customer has device protection (Yes, No, No internet service)
- **TechSupport**: Whether the customer has tech support (Yes, No, No internet service)
- **StreamingTV**: Whether the customer has streaming TV (Yes, No, No internet service)
- **StreamingMovies**: Whether the customer has streaming movies (Yes, No, No internet service)
- **Contract**: The contract term of the customer (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether the customer has paperless billing (Yes, No)
- **PaymentMethod**: The payment method of the customer (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- **MonthlyCharges**: The amount charged to the customer monthly
- **TotalCharges**: The total amount charged to the customer
- **Churn**: Whether the customer churned or not (Yes or No)

## Problem Statement
The task is to develop a machine learning model that can predict customer churn accurately based on the provided dataset. Before developing the model, exploratory data analysis (EDA) and feature engineering steps are expected to be performed to gain insights into the data and preprocess it accordingly.

## Approach
1. **Data Understanding**: Explore the dataset to understand its structure, features, and distributions.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and perform feature scaling if necessary.
3. **Exploratory Data Analysis (EDA)**: Analyze the relationships between features and the target variable (Churn) to identify patterns and trends.
4. **Feature Engineering**: Create new features if necessary and select relevant features for model training.
5. **Model Selection**: Choose appropriate machine learning algorithms for classification tasks and evaluate their performance using suitable metrics.
6. **Hyperparameter Tuning**: Fine-tune the selected model to improve its performance.
7. **Model Evaluation**: Evaluate the final model on unseen data to assess its generalization ability.

## Dependencies
- Python 3.x
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## Contributors
- [Meryem Eroğlu](https://github.com/MeryemEroglu)

