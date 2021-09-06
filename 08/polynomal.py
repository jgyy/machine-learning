# pylint: disable=no-member
# pylint: disable=import-error
"""
Polynomial Regression
"""
# Importing the libraries
from numpy import arange
from matplotlib.pyplot import scatter, plot, title, xlabel, ylabel, show
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = read_csv("./Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
scatter(X, y, color="red")
plot(X, lin_reg.predict(X), color="blue")
title("Truth or Bluff (Linear Regression)")
xlabel("Position Level")
ylabel("Salary")
show()

# Visualising the Polynomial Regression results
scatter(X, y, color="red")
plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue")
title("Truth or Bluff (Polynomial Regression)")
xlabel("Position Level")
ylabel("Salary")
show()

# Higher resolution and smoother curve
X_grid = arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
scatter(X, y, color="red")
plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
title("Truth or Bluff (Polynomial Regression 2)")
xlabel("Position Level")
ylabel("Salary")
show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
