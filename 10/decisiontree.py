"""
Decision Tree Regression
"""
# Importing the libaries
from numpy import arange
from matplotlib.pyplot import scatter, plot, title, xlabel, ylabel, show
from pandas import read_csv
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
dataset = read_csv("Position_Salaries.csv")
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the wholed dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
result = regressor.predict([[6.5]])
print(result)

# Visualising the Decision Tree Regresion model (high resolution)
X_grid = arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
scatter(X, y, color="red")
plot(X_grid, regressor.predict(X_grid), color="blue")
title("Truth or Bluff (Decision Tree Regression)")
xlabel("Position level")
ylabel("Salary")
show()
