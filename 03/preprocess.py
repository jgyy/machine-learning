"""
Section 3: Data Preprocessing in Python
"""
# Importing the Libraries
from numpy import nan, array
from pandas import read_csv
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Importing the dataset
dataset = read_csv("Data.csv")
x_data = dataset.iloc[:, :-1].values
dataset = read_csv("Data.csv")
y_data = dataset.iloc[:, -1].values
print("X data:\n", x_data)
print("Y data:\n", y_data)

# Taking care of Missing Data
imputer = SimpleImputer(missing_values=nan, strategy="mean")
imputer.fit(x_data[:, 1:3])
x_data[:, 1:3] = imputer.transform(x_data[:, 1:3])
print("X data:\n", x_data)

# Encoding Categorical Data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
x_data = array(ct.fit_transform(x_data))
le = LabelEncoder()
y_data = le.fit_transform(y_data)
print("X data:\n", x_data)
print("Y data:\n", y_data)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=1
)
print("X train:\n", x_train)
print("X test:\n", x_test)
print("Y train:\n", y_train)
print("Y test:\n", y_test)

# Feature Scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print("X train:\n", x_train)
print("X test:\n", x_test)
