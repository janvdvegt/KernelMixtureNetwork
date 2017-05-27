import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd


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
    else:
        centers = np.empty(0)

    if method is 'random':
        return np.random.choice(y, k)

    # Iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method is 'distance':
        raise NotImplementedError

    # Use 1-D k-means clustering to determine k output points plus the two end points
    elif method is 'k_means':
        model = KMeans(n_clusters=k, n_jobs=-2)
        model.fit(y.reshape(-1, 1))
        cluster_centers = model.cluster_centers_
        return np.append(centers, cluster_centers)

    # Use agglomerative clustering to determine k output points plus the two end points
    elif method is 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, linkage='complete')
        model.fit(y.reshape(-1, 1))
        labels = pd.Series(model.labels_, name='label')
        y_s = pd.Series(y, name='y')
        df = pd.concat([y_s, labels], axis=1)
        cluster_centers = df.groupby('label')['y'].mean().values
        np.append(centers, cluster_centers)

        return np.append(centers, cluster_centers)

    else:
        raise ValueError("unknown method '{}'".format(method))
