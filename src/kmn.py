import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from sklearn.base import BaseEstimator
from edward.models import Categorical, Mixture, Normal
from keras.layers import Dense, Dropout
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt


def sample_center_points(y, method='all', k=100, keep_edges=False):
    """
    function to define kernel centers with various downsampling alternatives
    """

    # make sure y is 1D
    y = y.ravel()

    # keep all points as kernel centers
    if method is 'all':
        return y

    # retain outer points to ensure expressiveness at the target borders
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

    # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method is 'distance':
        raise NotImplementedError

    # use 1-D k-means clustering
    elif method is 'k_means':
        model = KMeans(n_clusters=k, n_jobs=-2)
        model.fit(y.reshape(-1, 1))
        cluster_centers = model.cluster_centers_

    # use agglomerative clustering
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


class KernelMixtureNetwork(BaseEstimator):

    def __init__(self, n_samples=10, center_sampling_method='k_means', n_centers=20, keep_edges=False,
                 init_scales='default', estimator=None, X_ph=None):

        self.sess = ed.get_session()
        self.inference = None

        self.estimator = estimator
        self.X_ph = X_ph

        self.n_samples = n_samples
        self.center_sampling_method = center_sampling_method
        self.n_centers = n_centers
        self.keep_edges = keep_edges

        self.train_loss = np.empty(0)
        self.test_loss = np.empty(0)

        if init_scales is 'default':
            init_scales = np.array([1])
        self.init_scales = init_scales
        self.n_scales = len(self.init_scales)

        self.fitted = False

    def fit(self, X, y, n_epoch, **kwargs):
        """
        build and train model
        """
        # define the full model
        self._build_model(X, y)

        # setup inference procedure
        self.inference = ed.MAP(data={self.mixtures: self.y_ph})
        self.inference.initialize(var_list=tf.trainable_variables(), n_iter=n_epoch)
        tf.global_variables_initializer().run()

        # train the model
        self.partial_fit(X, y, n_epoch=n_epoch, **kwargs)
        self.fitted = True

    def partial_fit(self, X, y, n_epoch=1, eval_set=None):
        """
        update model
        """
        print("fitting model")

        # loop over epochs
        for i in range(n_epoch):

            # run inference, update trainable variables of the model
            info_dict = self.inference.update(feed_dict={self.X_ph: X, self.y_ph: y})

            train_loss = info_dict['loss'] / len(y)
            self.train_loss = np.append(self.train_loss, train_loss)

            if eval_set is not None:
                X_test, y_test = eval_set
                test_loss = self.sess.run(self.inference.loss, feed_dict={self.X_ph: X_test, self.y_ph: y_test}) / len(y_test)
                self.test_loss = np.append(self.test_loss, test_loss)

            # only print progress for the initial fit, not for additional updates
            if self.fitted is False:
                self.inference.print_progress(info_dict)

        print("mean log-loss train: {:.3f}".format(train_loss))
        if eval_set is not None:
            print("man log-loss test: {:.3f}".format(test_loss))

        print("optimal scales: {}".format(self.sess.run(self.scales)))

    def predict(self, X, y):
        """
        likelihood of a given target value
        """
        return self.sess.run(self.likelihoods, feed_dict={self.X_ph: X, self.y_ph: y})

    def predict_density(self, X, y=None, resolution=100):
        """
        conditional density over a predefined grid of target values
        """
        if y is None:
            y = np.linspace(self.y_min, self.y_max, num=resolution)

        return self.sess.run(self.densities, feed_dict={self.X_ph: X, self.y_grid_ph: y})

    def sample(self, X):
        """
        sample from the conditional mixture distributions
        """
        return self.sess.run(self.samples, feed_dict={self.X_ph: X})

    def score(self, X, y):
        """
        return mean log likelihood
        """
        likelihoods = self.predict(X, y)
        return np.log(likelihoods).mean()

    def _build_model(self, X, y):
        """
        implementation of the KMN
        """
        # create a placeholder for the target
        self.y_ph = y_ph = tf.placeholder(tf.float32, [None])

        #  store feature dimension size for placeholder
        self.n_features = X.shape[1]

        # if no external estimator is provided, create a default neural network
        if self.estimator is None:
            self.X_ph = tf.placeholder(tf.float32, [None, self.n_features])
            # two dense hidden layers with 15 nodes each
            x = Dense(15, activation='relu')(self.X_ph)
            x = Dense(15, activation='relu')(x)
            self.estimator = x

        # get batch size
        self.batch_size = tf.shape(self.X_ph)[0]

        # locations of the gaussian kernel centers
        n_locs = self.n_centers
        self.locs = locs = sample_center_points(y, method=self.center_sampling_method, k=n_locs, keep_edges=self.keep_edges)
        self.locs_array = locs_array = tf.unstack(tf.transpose(tf.multiply(tf.ones((self.batch_size, n_locs)), locs)))

        # scales of the gaussian kernels
        self.scales = scales = tf.nn.softplus(tf.Variable(self.init_scales, dtype=tf.float32, trainable=True))
        self.scales_array = scales_array = tf.unstack(tf.transpose(tf.multiply(tf.ones((self.batch_size, self.n_scales)), scales)))

        # kernel weights, as output by the neural network
        self.weights = weights = Dense(n_locs * self.n_scales, activation='softplus')(self.estimator)

        # mixture distributions
        self.cat = cat = Categorical(logits=weights)
        self.components = components = [Normal(loc=loc, scale=scale) for loc in locs_array for scale in scales_array]
        self.mixtures = mixtures = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))

        # tensor to store samples
        self.samples = mixtures.sample(sample_shape=self.n_samples)

        # store minmax of training target values for a sensible default grid for self.predict_density()
        self.y_min = y.min()
        self.y_max = y.max()
        # placeholder for the grid
        self.y_grid_ph = y_grid_ph = tf.placeholder(tf.float32)
        # tensor to store grid point densities
        self.densities = tf.transpose(mixtures.prob(tf.reshape(y_grid_ph, (-1, 1))))

        # tensor to compute likelihoods
        self.likelihoods = mixtures.prob(y_ph)

    def plot_loss(self):
        """
        plot train loss and optionally test loss over epochs
        source: http://edwardlib.org/tutorials/mixture-density-network
        """
        # new figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))

        # plot train loss
        plt.plot(np.arange(len(self.train_loss)), -self.train_loss, label='Train')

        if len(self.test_loss) > 0:
            # plot test loss
            plt.plot(np.arange(len(self.test_loss)), -self.test_loss, label='Test')

        plt.legend(fontsize=20)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('mean negative log-likelihood', fontsize=15)
        plt.show()

        return fig, axes
