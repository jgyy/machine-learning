"""
XGBoost
"""
# Importing the libraries
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
dataset = read_csv("Data.csv")
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Training XGBoost on the Training set
classifier = XGBClassifier()
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
