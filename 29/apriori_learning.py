"""
Apriori
"""
# Importing the libraries
import numpy
import matplotlib.pyplot
from apyori import apriori
from pandas import read_csv, DataFrame

# Data Preprocessing
dataset = read_csv("Market_Basket_Optimisation.csv")
transactions = []
for i in range(len(dataset)):
    transactions.append(list(map(str, dataset.values[i])))

# Training the Apriori model on the dataset
rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2,
)

# Displaying the first result coming directly from the output of the apriori function
results = list(rules)


def inspect(res):
    """
    Putting the result well organised into a Pandas Dataframe
    """
    lhs = [tuple(result[2][0][0])[0] for result in res]
    rhs = [tuple(result[2][0][1])[0] for result in res]
    supports = [result[1] for result in res]
    confidences = [result[2][0][2] for result in res]
    lifts = [result[2][0][3] for result in res]
    return list(zip(lhs, rhs, supports, confidences, lifts))


# Displaying the result non sorted
resultsInDataFrame = DataFrame(
    inspect(results),
    columns=["Left Hand Side", "Right Hand Side", "Support", "Confidence", "Lift"],
)
print(resultsInDataFrame)

# Desplaying the result sorted by descending lifts
print(resultsInDataFrame.nlargest(n=10, columns="Lift"))
