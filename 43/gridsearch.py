"""
Grid Search
"""
# Importing the libraries
from pandas import read_csv
from numpy import meshgrid, arange, array, unique
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Kernel SVM model on the Training set
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:", cm)
acc = accuracy_score(y_test, y_pred)
print("accuracy_score", acc)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(f"Accuracy: {accuracies.mean()*100:.2f} %")
print(f"Standard Deviation: {accuracies.std()*100:.2f} %")

# Applying Grid Search to find the best model and the best parameters
parameters = [
    {"C": [0.25, 0.5, 0.75, 1], "kernel": ["linear"]},
    {"C": [0.25, 0.5, 0.75, 1], "kernel": ["rbf"], "gamma": list(arange(0.1, 1, 0.1))},
]
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(f"Best Accuracy: {best_accuracy*100:.2f} %")
print(f"Best Parameters: {best_parameters}")

_log.setLevel("ERROR")
# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = meshgrid(
    arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
    arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
)
contourf(
    X1,
    X2,
    classifier.predict(array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
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
title("Kernel SVM with grid search (Training set)")
xlabel("Age")
ylabel("Estimated Salary")
legend()
show()

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = meshgrid(
    arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
    arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
)
contourf(
    X1,
    X2,
    classifier.predict(array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
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
title("Kernel SVM with grid search (Test set)")
xlabel("Age")
ylabel("Estimated Salary")
legend()
show()
