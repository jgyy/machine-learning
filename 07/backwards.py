"""
Multiple Linear Regression in Python - Backward Elimination
"""
# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
dataset = pd.read_csv("./50_Startups.csv")
X = dataset.iloc[:, :-1].values
dataset = pd.read_csv("./50_Startups.csv")
y = dataset.iloc[:, -1].values

# Encoding categorical data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Building the optimal model using Backward Elimination
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
y_range = [0, 1, 2, 3, 4, 5]
for num in [None, 2, 1, 4, 5]:
    if num in y_range:
        y_range.remove(num)
    X_opt = X[:, y_range]
    X_opt = X_opt.astype(np.float64)
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(y_range, "\n", regressor_OLS.summary())
