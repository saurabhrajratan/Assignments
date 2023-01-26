import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class KMeans:
    np.random.seed(42)
    def __init__(self, K=5, max_iters=100):
        self.K = K
        self.max_iters = max_iters
        self.objective = []

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.subtract(x1, x2) ** 2))

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        # Optimize clusters
        for _ in range(self.max_iters):
            error = 0
            for centroid_idx in range(len(self.clusters)):
                for datapoint in self.clusters[centroid_idx]:
                    error += np.sum(np.subtract(datapoint, self.centroids[centroid_idx])**2)
            self.objective.append(error)

            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [self.euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def myplot(self):
        x_axis = np.arange(1, len(self.objective), 1)
        #exclude(ignore) first initialization
        y_axis = self.objective[1:]
        print("No. of Iterations: ", x_axis)
        print("sum of sq of distances from nearest centroid:\n", self.objective)
        plt.xlabel("No. of Iterations")
        plt.ylabel("sum of sq of distances from nearest centroid")
        plt.plot(x_axis, y_axis, marker="x", color="green")
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('A2Q1.csv')
    X = df.iloc[:, 0:50].values
    y = df.iloc[:, 49:50].values

    k = KMeans(K=4, max_iters=35)
    y_pred = k.predict(X)
    k.myplot()
