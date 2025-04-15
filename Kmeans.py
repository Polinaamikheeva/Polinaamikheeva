import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X, plot_steps=True):
        np.random.seed(self.random_state)
        self.centroids = X[np.random.permutation(X.shape[0])[:self.n_clusters]]

        for i in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            if plot_steps:
                plt.figure(figsize=(8, 6))
                for k in range(self.n_clusters):
                    plt.scatter(X[self.labels == k][:, 0], X[self.labels == k][:, 1])
                plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100)
                plt.title(f'Step {i + 1}')
                plt.show()

            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids


def find_optimal_clusters(X, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = MyKMeans(n_clusters=k)
        kmeans.fit(X, plot_steps=False)
        distances = np.sqrt(((X - kmeans.centroids[kmeans.labels]) ** 2).sum(axis=1))
        distortions.append(np.sum(distances))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()

    diffs = np.diff(distortions)
    optimal_k = np.argmin(diffs[1:] / diffs[:-1]) + 2
    return optimal_k


def main():
    iris = load_iris()
    X = iris.data

    optimal_k = find_optimal_clusters(X)
    print(f"Optimal clusters: {optimal_k}")

    kmeans = MyKMeans(n_clusters=optimal_k)
    kmeans.fit(X)

    data_with_clusters = np.column_stack((X, kmeans.labels))
    df = pd.DataFrame(data_with_clusters, columns=iris.feature_names + ['cluster'])

    #sns.pairplot(df, hue='cluster', palette='viridis')
    # plt.suptitle('Pairplot of Iris Data with Clusters', y=1.02)
    # plt.show()


if __name__ == '__main__':
    main()
