import unittest
from unittest import TestCase
from sklearn.model_selection import train_test_split
import numpy as np
from src.kmn import KernelMixtureNetwork


class TestKernelMixtureNetwork(TestCase):

    def create_dataset(self, n=5000):
        y_data = np.random.uniform(-10.5, 10.5, n)
        r_data = np.random.normal(size=n)  # random noise
        x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
        x_data = x_data.reshape((n, 1))

        return train_test_split(x_data, y_data, random_state=42)

    def test_run(self):
        """test case with simulated data and network training"""

        X_train, X_test, y_train, y_test = self.create_dataset()

        kmn = KernelMixtureNetwork()

        self.assertTrue(isinstance(kmn, object))

        kmn.fit(X_train, y_train, n_epoch=400, eval_set=(X_test, y_test))

        self.assertTrue(kmn.train_loss[-1] < 2.)
        self.assertTrue(kmn.test_loss[-1] < 3.)

        likelihoods = kmn.predict(X_test, y_test)
        mean_loglik = np.log(likelihoods).sum() / len(y_test)

        self.assertTrue(mean_loglik < 3.)

if __name__ == '__main__':
    unittest.main()
