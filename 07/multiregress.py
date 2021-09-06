# pylint: disable=no-member
# pylint: disable=import-error
"""
Multiple Linear Regression
"""
# Importing the libraries
from numpy import array, set_printoptions, concatenate
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = read_csv("./50_Startups.csv")
x_data = dataset.iloc[:, :-1].values
y_data = dataset.iloc[:, -1].values

# Encoding categorial data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough"
)
x_data = array(ct.fit_transform(x_data))

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=0
)

# Training the Multiple Linear Regression on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)
set_printoptions(precision=2)
print(concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
