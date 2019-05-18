from random import sample

from pyclustering.cluster.kmedoids import kmedoids
from sklearn.base import BaseEstimator, ClusterMixin


class KMedoids(BaseEstimator, ClusterMixin):
    """K Medoids sklearn wrapper, based on pyclustering library."""

    def __init__(self, distance_matrix=None, k=3):
        self.distance_matrix = distance_matrix
        self.k = k
        self.medoids_ = None
        self.labels_ = None
        self.initial_medoid_indices = sample(range(self.distance_matrix.shape[0]), self.k)
        self.k_medoids = kmedoids(self.distance_matrix, self.initial_medoid_indices, data_type='distance_matrix')

    def fit(self, X=None, y=None):
        if self.distance_matrix is None:
            raise ValueError('Distance matrix should be initialized first.')

        return self.k_medoids.process()

    def predict(self, X, y=None):
        self.medoids_ = self.k_medoids.get_medoids()
        self.labels_ = self.k_medoids.get_clusters()

        return self.labels_
