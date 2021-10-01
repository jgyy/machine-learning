"""
Hierarchical Clustering
"""
# Importing the libraries
from random import shuffle
from pandas import read_csv
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.pyplot import title, xlabel, ylabel, show, scatter, legend

# Importing the dataset
dataset = read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
dendro = dendrogram(linkage(X, method="ward"))
title("Dendrogram")
xlabel("Customers")
ylabel("Euclidean distances")
show()

# Training the Hierarchical Clustering model on the dataset
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)
print(y_hc)

# Visualising the clusters
colors = [
    "maroon",
    "red",
    "purple",
    "fuchsia",
    "green",
    "lime",
    "olive",
    "yellow",
    "navy",
    "blue",
    "teal",
    "aqua",
]
shuffle(colors)
for i, color in enumerate(colors[:5]):
    scatter(
        X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=color, label=f"Cluster {i+1}"
    )
title("Clusters of Customers")
xlabel("Annual Income (k$)")
ylabel("Spending Score (1-100)")
legend()
show()
