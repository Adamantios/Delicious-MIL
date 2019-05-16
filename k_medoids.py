from random import randint

import numpy as np
from scipy.spatial.distance import directed_hausdorff


class KMedoidsHaussdorff:
    def __init__(self):
        self.distances = None

    def cluster(self, data: np.ndarray, k=3, max_iter=300):
        n_data = data.shape[0]

        # Initialize distances array.
        self.distances = np.empty((n_data, n_data))

        # Calculate symmetric haussdorff distances.
        for outer, x in enumerate(data):
            for inner, y in enumerate(data):
                self.distances[outer, inner] = max(directed_hausdorff(x, y)[0],
                                                   directed_hausdorff(y, x)[0])

        # Pick k random medoids.
        curr_medoids = np.array([-1] * k)
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([randint(0, n_data - 1) for _ in range(k)])
        old_medoids = np.empty(k)
        new_medoids = np.empty(k)

        # Until the medoids stop updating, do the following:
        clusters = None
        n_iter = 0
        while not ((old_medoids == curr_medoids).all()) or n_iter != max_iter:
            n_iter += 1

            # Assign each point to cluster with closest medoid.
            clusters = self._assign_points_to_clusters(curr_medoids)

            # Update cluster medoids to be lowest cost point.
            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = self._compute_new_medoid(cluster)

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]

        return clusters, curr_medoids

    def _assign_points_to_clusters(self, medoids):
        distances_to_medoids = self.distances[:, medoids]
        clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters

    def _compute_new_medoid(self, cluster):
        mask = np.ones(self.distances.shape)
        mask[np.ix_(cluster, cluster)] = 0.
        cluster_distances = np.ma.masked_array(data=self.distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)

        return costs.argmin(axis=0, fill_value=10e9)
