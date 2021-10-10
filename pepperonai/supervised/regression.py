"""
Linear Regression
"""

__author__ = "Jon Wiggins"

import numpy as np


class LinearRegression:
    def __init__(self):
        self.fitted = False
        pass

    def train(self, X: np.array, y: np.array, lmbda: int):
        """
        Fit a linear regression model

        Args:
            X (np.array): (N, d) array
            Y (np.array): (N,) array
            lmbda (int): regularization coefficient
        """
        N = X.shape[0]
        d = X.shape[1]
        y_col = Y.reshape(-1, 1)
        # add column for coeffs
        matrix = np.hstack((np.ones((N, 1)), X))
        const = N * lmbda * np.eye(d + 1)
        self.beta = (
            np.linalg.pinv(matrix.T.dot(matrix) + const).dot(matrix.T).dot(Y_col)
        )
        self.fitted = True

    def probe(self, X: np.array) -> np.array:
        """
        Returns prediction for examples

        Args:
            X (np.array): Input examples with snape (N, d)

        Returns:
            np.array: Predicted results of shape (N, )
        """
        if not self.fitted:
            raise Exception("This model has not been fitted")
        N = X.shape[0]
        d = X.shape[1]
        matrix = np.hstack((np.ones((N, 1)), X))
        results = matrix.dot(self.beta).reshape(-1)
        return results
