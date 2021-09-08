"""
Support Vector Regression (SVR)
"""
# Importing the libraries
from warnings import filterwarnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning

# Importing the dataset
filterwarnings(action="ignore", category=DataConversionWarning)
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
dataset = pd.read_csv("Data.csv")
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
regressor = SVR(kernel="rbf")
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)

# Evaluating the Model Performance
print(r2_score(y_test, y_pred))
