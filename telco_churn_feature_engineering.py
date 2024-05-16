import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action="ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)


# EXPLORATORY DATA ANALYSIS
# Examine the overall picture.
def check_df(dataframe, head=5):
    """
    This function provides an overview of a DataFrame including its shape, data types,
    the first 'head' rows, the last 'head' rows, the count of missing values, and selected quantiles.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be analyzed.
    head : int, optional
        Number of rows to display from the beginning and end of the DataFrame (default is 5).

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.99, 1]).T)

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Provides the names of categorical, numeric, and categorical but cardinal variables in the dataset.
    Note: Numeric-appearing categorical variables are also included in the categorical variables.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be analyzed.
    cat_th : int, optional
        The threshold value for variables that are numerical but categorical (default is 10).
    car_th : int, optional
        The threshold value for categorical but cardinal variables (default is 20).

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numeric variables.
    cat_but_car : list
        List of categorical-appearing cardinal variables.

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is within cat_cols.

    """
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(f"Numerical columns: {num_cols}")
print(f"Categorical columns {cat_cols}")
print(f"Categorical but cardinal columns: {cat_but_car}")

#ANALYSIS OF CATEGORICAL VARIABLES
def cat_summary(dataframe, categorical_col, plot=False):
    """
    Display a summary of a categorical variable in a DataFrame, including value counts and ratios.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame containing the categorical variable.
    categorical_col : str
        The name of the categorical column to be analyzed.
    plot : bool, optional
        If True, display a countplot to visualize the distribution (default is False).

    """
    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts()/len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#ANALYSIS OF NUMERICAL VARIABLES
def num_summary(dataframe, numerical_col, plot=False):
    """
    Display a summary of a numerical variable in a DataFrame, including descriptive statistics and an optional histogram.

    Parameters
    ----------
    dataframe (DataFrame): The DataFrame containing the numerical variable.
    numerical_col (str): The name of the numerical column to be analyzed.
    plot (bool, optional): If True, display a histogram to visualize the distribution (default is False).
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# When we look at tenure, we observe a high number of customers with 1-month tenure, followed by customers with 70-month tenure.
# This could be due to different contracts; let's compare the tenure of customers with monthly contracts to those with 2-year contracts.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

# Looking at MonthlyCharges, customers with monthly contracts might have higher average monthly payments.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show()