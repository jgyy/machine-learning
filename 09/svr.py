"""
Support Vector Regression
"""
# Importing the libraries
from numpy import arange
from matplotlib.pyplot import scatter, plot, title, xlabel, ylabel, show
from pandas import read_csv
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from warnings import filterwarnings

# Importing the dataset
dataset = read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
dataset = read_csv("Position_Salaries.csv")
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scaling
filterwarnings(action="ignore", category=DataConversionWarning)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print("X:", X)
print("y:", y)

# Training the SRV model on the whole dataset
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predicting a new result
result = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
print("sc_y.inverse_transform[6.5]:", result)

# Visualising the SRV results
scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plot(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(regressor.predict(X)),
    color="blue",
)
title("Truth or Bluff (Support Vector Regression)")
xlabel("Position level")
ylabel("Salary")
show()

# Higher resolution and smoother curve
X_grid = arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plot(
    X_grid,
    sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))),
    color="blue",
)
title("Truth or Bluff (Support Vector Regression Higher resolution)")
xlabel("Position level")
ylabel("Salary")
show()
