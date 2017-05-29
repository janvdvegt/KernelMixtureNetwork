import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


def sample_center_points(y, method='all', k=100, keep_edges=False):

    y = y.ravel()

    # Keep all points as kernel centers
    if method is 'all':
        return y

    # Remove outer points to keep expressiveness at the borders
    if keep_edges:
        y = np.sort(y)
        centers = np.array([y[0], y[-1]])
        y = y[1:-1]
        # adjust k such that the final output has size k
        k -= 2
    else:
        centers = np.empty(0)

    if method is 'random':
        cluster_centers = np.random.choice(y, k, replace=False)

    # Iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method is 'distance':
        raise NotImplementedError

    # Use 1-D k-means clustering to determine k output points plus the two end points
    elif method is 'k_means':
        model = KMeans(n_clusters=k, n_jobs=-2)
        model.fit(y.reshape(-1, 1))
        cluster_centers = model.cluster_centers_

    # Use agglomerative clustering to determine k output points plus the two end points
    elif method is 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, linkage='complete')
        model.fit(y.reshape(-1, 1))
        labels = pd.Series(model.labels_, name='label')
        y_s = pd.Series(y, name='y')
        df = pd.concat([y_s, labels], axis=1)
        cluster_centers = df.groupby('label')['y'].mean().values

    else:
        raise ValueError("unknown method '{}'".format(method))

    return np.append(centers, cluster_centers)


class KernelMixtureNetwork(BaseEstimator, RegressorMixin):

    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, y=None):
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        """mean? log likelihood"""
        raise NotImplementedError



