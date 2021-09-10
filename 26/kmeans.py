"""
K-Means Clustering
"""
# Importing the libraries
from random import shuffle
from pandas import read_csv
from sklearn.cluster import KMeans
from matplotlib.pyplot import plot, title, xlabel, ylabel, show, scatter, legend

# Importing the dataset
dataset = read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plot(range(1, 11), wcss)
title("The Elbow Method")
xlabel("Number of clusters")
ylabel("WCSS")
show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

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
        X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=color, label=f"Cluster {i+1}"
    )
scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c=colors[5],
    label="Centroids",
)
title("Clusters of Customers")
xlabel("Annual Income (k$)")
ylabel("Spending Score (1-100)")
legend()
show()
