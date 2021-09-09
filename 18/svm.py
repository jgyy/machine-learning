"""
Support Vector Machine
"""
# Import the libraries
from pandas import read_csv
from numpy import concatenate, meshgrid, arange, array, unique
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.axes._axes import _log
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import (
    contourf,
    xlim,
    ylim,
    scatter,
    title,
    xlabel,
    ylabel,
    legend,
    show,
)

# Importing the dataset
dataset = read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
dataset = read_csv("Social_Network_Ads.csv")
y = dataset.iloc[:, -1].values

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM Model on the Training set
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)

# Predicting a new result
res = classifier.predict(sc.transform([[30, 151100]]))
print("Age 30, Salary 151.1k:", res)

# Predicting the test set result
y_pred = classifier.predict(X_test)
res = concatenate((y_pred.reshape(len(y_pred), 1), (y_test.reshape(len(y_test), 1))), 1)
print("Test set result:", res[:9])

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:", cm)
acc = accuracy_score(y_test, y_pred)
print("accuracy_score", acc)

_log.setLevel("ERROR")
# Visualising the Training set results
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = meshgrid(
    arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.5),
    arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.5),
)
contourf(
    X1,
    X2,
    classifier.predict(sc.transform(array([X1.ravel(), X2.ravel()]).T)).reshape(
        X1.shape
    ),
    alpha=0.75,
    cmap=ListedColormap(("red", "green")),
)
xlim(X1.min(), X1.max())
ylim(X2.min(), X2.max())
for i, j in enumerate(unique(y_set)):
    scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "green"))(i),
        label=j,
    )
title("Support Vector Machine (Training set)")
xlabel("Age")
ylabel("Estimated Salary")
legend()
show()

# Visualising the Test set results
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = meshgrid(
    arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.5),
    arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.5),
)
contourf(
    X1,
    X2,
    classifier.predict(sc.transform(array([X1.ravel(), X2.ravel()]).T)).reshape(
        X1.shape
    ),
    alpha=0.75,
    cmap=ListedColormap(("red", "green")),
)
xlim(X1.min(), X1.max())
ylim(X2.min(), X2.max())
for i, j in enumerate(unique(y_set)):
    scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "green"))(i),
        label=j,
    )
title("Support Vector Machine (Test set)")
xlabel("Age")
ylabel("Estimated Salary")
legend()
show()
