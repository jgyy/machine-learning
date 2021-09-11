"""
Artificial Neural Networks
"""
# Importing the libraries
from logging import getLogger, ERROR
from pandas import read_csv
from numpy import array, concatenate
from tensorflow import __version__
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Importing the dataset
print(__version__)
dataset = read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
dataset = read_csv("Churn_Modelling.csv")
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Label Encoding the "Gender" column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

# One Hot Encoding the "Geography" column
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = array(ct.fit_transform(X))
print(X)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
ann = Sequential()

# Adding the input layer and the first hidden layer
ann.add(Dense(units=6, activation="relu"))

# Adding the second hidden layer
ann.add(Dense(units=6, activation="relu"))

# Adding the output layer
ann.add(Dense(units=1, activation="sigmoid"))

# Compiling the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the ANN on the Training set
getLogger("tensorflow").setLevel(ERROR)
ann.fit(X_train, y_train, batch_size=32, epochs=100)


def single_observation():
    """
    Predicting the result of a single observation
    """
    geography = (1, 0, 0)  # France
    credit_score = 600
    gender = 1  # Male
    age = 40
    tenure = 3
    balance = 60000
    products = 2
    credit_card = 1  # Yes
    active_member = 1  # Yes
    salary = 50000
    return (
        ann.predict(
            sc.transform(
                [
                    [
                        *geography,
                        credit_score,
                        gender,
                        age,
                        tenure,
                        balance,
                        products,
                        credit_card,
                        active_member,
                        salary,
                    ]
                ]
            )
        )
        > 0.5
    )


result = single_observation()
print(result)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5
print(concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)
