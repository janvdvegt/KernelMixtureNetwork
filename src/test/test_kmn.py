import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from src.kmn import KernelMixtureNetwork
from keras.layers import Dense


class TestKernelMixtureNetwork(tf.test.TestCase):

    def create_dataset(self, n=5000):
        """
        function to create dummy data
        source: http://edwardlib.org/tutorials/mixture-density-network
        """
        y_data = np.random.uniform(-10.5, 10.5, n)
        r_data = np.random.normal(size=n)  # random noise
        x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
        x_data = x_data.reshape((n, 1))

        return train_test_split(x_data, y_data, random_state=42)

    def test_run(self):
        """test case with simulated data and network training with default settings"""

        X_train, X_test, y_train, y_test = self.create_dataset()

        kmn = KernelMixtureNetwork()

        self.assertTrue(isinstance(kmn, object))

        kmn.fit(X_train, y_train, n_epoch=100, eval_set=(X_test, y_test))

        # TODO: make this test deterministic!
        train_loss1 = kmn.train_loss[-1]
        self.assertTrue(train_loss1 < 2.)
        self.assertTrue(kmn.test_loss[-1] < 3.)

        kmn.partial_fit(X_train, y_train, n_epoch=200, eval_set=(X_test, y_test))
        self.assertTrue(kmn.train_loss[-1] <= train_loss1)

        likelihoods = kmn.predict(X_test, y_test)
        mean_loglik = np.log(likelihoods).mean()

        self.assertTrue(mean_loglik < 3.)

        score = kmn.score(X_test, y_test)
        self.assertTrue(abs(mean_loglik - score) < 0.01)

        kmn.sess.close()

        # TODO:
        # test for sample()
        # test for predict_density()
        # test for plot_loss()

    def test_external_estimator(self):
        """test case with simulated data and network training with an external estimator"""

        X_train, X_test, y_train, y_test = self.create_dataset()

        kmn1 = KernelMixtureNetwork()
        kmn1.fit(X_train, y_train, n_epoch=100)
        kmn1.sess.close()

        X_ph = tf.placeholder(tf.float32, [None, X_train.shape[1]])
        x = Dense(15, activation='relu')(X_ph)
        neural_network = Dense(15, activation='relu')(x)

        kmn2 = KernelMixtureNetwork(estimator=neural_network, X_ph=X_ph)
        kmn2.fit(X_train, y_train, n_epoch=200)
        kmn2.sess.close()

        self.assertTrue(abs(kmn1.train_loss[-1] - kmn2.train_loss[-1]) < 0.1)

    def test_sample_center_points(self):
        pass

        # TODO:
        # test sample_center_points() with all different methods

if __name__ == '__main__':
    tf.test.main()

