# FEATURE ENGINEERING
# Step 1: Perform necessary operations for missing and outlier values.
# Step 2: Create new variables.
# Step 3: Perform encoding operations.
# Step 4: Standardize numeric variables.
# Step 5: Build a model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings

warnings.simplefilter(action="ignore")

from telco_churn_exploratory_data_analysis import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the data
df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# Data type conversion
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# ANALYSIS OF MISSING VALUES

# Check if there are any missing values in the DataFrame
df.isnull().values.any()
# Get the count of missing values for each column
df.isnull().sum()
# Get the count of non-missing (not null) values for each column
df.notnull().sum()
# Get the total count of missing values in the entire DataFrame
df.isnull().sum().sum()
# Filter rows containing at least one missing value (NaN) in any column
var = df[df.isnull().any(axis=1)]
# Filter rows containing at least one non-missing (not null) value in any column
var1 = df[df.notnull().any(axis=1)]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.isnull().sum()
df.isnull().sum().sum()

# BASE MODEL SETUP

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
# Drop unnecessary columns
X = dff.drop(["Churn", "customerID"], axis=1)

# Column classification function
models = [
    ('Logistic Regression', LogisticRegression(random_state=46)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier(random_state=46)),
    ('Random Forest', RandomForestClassifier(random_state=46)),
    ('Support Vector Machine', SVC(gamma='auto', random_state=46)),
    ('XGBoost', XGBClassifier(random_state=46, use_label_encoder=False, eval_metric='logloss')),
    ('LightGBM', LGBMClassifier(random_state=46)),
    ('CatBoost', CatBoostClassifier(verbose=False, random_state=46))
]

# Evaluate models using cross-validation
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~ {name} ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# Accuracy: Proportion of correct predictions (TP+TN) / (TP+TN+FP+FN)
# Recall: Proportion of true positive predictions for the positive class TP / (TP+FN)
# Precision: Proportion of true positive predictions among all positive predictions TP / (TP+FP)
# F1 Score: A balance between precision and recall, calculated as 2 * (Precision * Recall) / (Precision + Recall)
# AUC (Area Under the Curve): Measures the classifier's ability to distinguish between positive and negative classes.

