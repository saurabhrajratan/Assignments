import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)
error_list = []
def KmeansClusteringAlgori(X,k,plot=False, iteration_count=1000):
    n = X.shape[0]

    dist = np.zeros((k,n))
    cluster_indicator_z = np.zeros((n,))
    means = X[np.random.choice(n,size=k,replace=False),:] 
  
    is_Kmeans_converged = 1
    iter_so_far=0
    while iter_so_far < iteration_count and is_Kmeans_converged > 0:
        # Calculating New labels of each clusters
        old_labels = cluster_indicator_z.copy()
        for j in range(k):
            dist[j,:] = np.sum((X - means[j,:])**2,axis=1)
        cluster_indicator_z = np.argmin(dist,axis=0)
        error_list.append(cluster_indicator_z)

        # if it is already converged then stop iteration
        is_Kmeans_converged = np.sum(cluster_indicator_z != old_labels)

        # Calculating New Mean of each clusters
        for cluster_label in range(k):
            means[cluster_label,:] = np.mean(X[cluster_indicator_z == cluster_label,:],axis=0)
            
        iter_so_far = iter_so_far+1

    print("Number of iterations: ", iter_so_far)
    return cluster_indicator_z

data = np.genfromtxt('A2Q1.csv', delimiter=',')

error_list = []
X = data
K = 4
KmeansClusteringAlgori(X,K,plot=True)
averaged_error = np.sum(np.array(error_list), axis=1) / np.array(error_list).shape[1]

fig = plt.figure(figsize=(8,5))
plt.title('Sum of Squared Error in KMEANS Clustering')
plt.xlabel('Initialization Axis ----->')
plt.ylabel('Sum of Squared Error Axis ----->')
plt.plot(averaged_error, color = 'red')
plt.show()


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

    def kmeansAlgo(self, X):
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
            self.clusters = self.createClusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.getCentroids(self.clusters)

        # Classify samples as the index of their clusters
        return self.getClusterLabels(self.clusters)

    def getClusterLabels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def createClusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closestCentroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closestCentroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [self.euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def getCentroids(self, clusters):
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
        fig = plt.figure(figsize=(8,5))
        plt.title('Sum of Squared Distances in KMEANS Clustering')
        plt.xlabel("No. of Iterations Axis ----->")
        plt.ylabel("sum of sq of distances from nearest centroid Axis ----->")
        plt.plot(x_axis, y_axis, marker="x", color="green")
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('A2Q1.csv')
    X = df.iloc[:, 0:50].values
    y = df.iloc[:, 49:50].values

    kmeans = KMeans(K=4, max_iters=35)
    y_pred = kmeans.kmeansAlgo(X)
    kmeans.myplot()


