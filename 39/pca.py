"""
Principal Component Analysis (PCA)
"""
# Importing the libraries
from pandas import read_csv
from numpy import meshgrid, arange, array, unique
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
dataset = read_csv("Wine.csv")
X = dataset.iloc[:, :-1].values
dataset = read_csv("Wine.csv")
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Featuring Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Training the Logical Regression model on the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:\n", cm)
acc = accuracy_score(y_test, y_pred)
print("accuracy_score", acc)

_log.setLevel("ERROR")
# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = meshgrid(
    arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
contourf(
    X1,
    X2,
    classifier.predict(array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("red", "green", "blue")),
)
xlim(X1.min(), X1.max())
ylim(X2.min(), X2.max())
for i, j in enumerate(unique(y_set)):
    scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "green", "blue"))(i),
        label=j,
    )
title("Logical Regression with PCA (Training set)")
xlabel("PC1")
ylabel("PC2")
legend()
show()

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = meshgrid(
    arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
contourf(
    X1,
    X2,
    classifier.predict(array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("red", "green", "blue")),
)
xlim(X1.min(), X1.max())
ylim(X2.min(), X2.max())
for i, j in enumerate(unique(y_set)):
    scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "green", "blue"))(i),
        label=j,
    )
title("Logical Regression with PCA (Test set)")
xlabel("PC1")
ylabel("PC2")
legend()
show()
