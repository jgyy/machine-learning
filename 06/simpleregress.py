# pylint: disable=no-member
# pylint: disable=import-error
"""
Simple Linear Regression
"""
# Importing the library
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import scatter, plot, title, xlabel, ylabel, show

# Importing the dataset
dataset = read_csv("./Salary_Data.csv")
x_data = dataset.iloc[:, :-1].values
y_data = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=0
)

# Training the Simple Linear Regression model n the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
print(y_pred)

# Visualising the Training set results
scatter(x_train, y_train, color="red")
plot(x_train, regressor.predict(x_train), color="blue")
title("Salary vs Experience (Training set)")
xlabel("Years of Experience")
ylabel("Salary")
show()

# Visualising the Test set results
scatter(x_test, y_test, color="red")
plot(x_train, regressor.predict(x_train), color="blue")
title("Salary vs Experience (Test set)")
xlabel("Years of Experience")
ylabel("Salary")
show()
