from random import sample

import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score


class KMedoids(BaseEstimator, ClusterMixin):
    """K Medoids sklearn wrapper, based on pyclustering library."""

    def __init__(self, k=0):
        self.k = k
        self.k_medoids = None
        self.clusters = None
        self.bag = None
        self.labels = None

    def fit(self, X, y=None):
        self.k_medoids = kmedoids(X, self.k, data_type='distance_matrix')
        self.k_medoids.process()
        self.clusters = self.k_medoids.get_clusters()
        return self

    def predict(self, X):
        self.bag = self.k_medoids.get_medoids()
        return self.bag

    def score(self, X, y=None):
        self.labels = np.zeros(X.shape[0])
        for i in range(len(self.clusters)):
            ind = self.clusters[i]
            self.labels[ind] = i
        return silhouette_score(X, self.labels, metric='precomputed')

    def get_params(self, deep=True):
        return {"k": self.k}


def grid_search_cv(X, parameters, cv=10):
    mean_scores = []
    for value in parameters:
        sil = []
        for _ in range(cv):
            init_indices = sample(range(X.shape[0]), value)
            clf = KMedoids(init_indices)
            clf.fit(X)
            sil.append(clf.score(X))

        mean_scores.append(np.round(np.mean(sil), 4))

    best_score = np.max(mean_scores)

    print('Best Silhouette score found after Grid search : %.4f' % best_score)
    print('Best parameter : %s' % parameters[np.argmax(mean_scores)])

    return mean_scores
