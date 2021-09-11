"""
Random Forest Regression
"""
# Importing the libraries
from numpy import arange
from matplotlib.pyplot import scatter, plot, title, xlabel, ylabel, show
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
dataset = read_csv("Position_Salaries.csv")
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predicting a new result
POSITION_LEVEL = 6.5
result = regressor.predict([[POSITION_LEVEL]])
print(result)

# Visualising the Random Forest Regression result (higher resolution)
X_grid = arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
scatter(X, y, color="red")
plot(X_grid, regressor.predict(X_grid), color="blue")
title("Truth or Bluff (Random Forest Regression)")
xlabel("Position Level")
ylabel("Salary")
show()
