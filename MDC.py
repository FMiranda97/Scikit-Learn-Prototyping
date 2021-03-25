import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin


class MDC(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self._class_list = np.unique(y, axis=0)

        self._centroids = np.zeros((len(self._class_list), X.shape[1]))  # each row is a centroid
        for i in range(len(self._class_list)):  # for each class, we evaluate its centroid
            temp = np.where(y == self._class_list[i])[0]
            self._centroids[i, :] = np.mean(np.array(X)[temp], axis=0)

    def predict(self, X):
        temp = np.argmin(
            cdist(X, self._centroids),  # distance between each pair of the two collections of inputs
            axis=1
        )
        y_pred = np.array([self._class_list[i] for i in temp])

        return y_pred

    def score(self, X, y):
        temp = np.argmin(
            cdist(X, self._centroids),  # distance between each pair of the two collections of inputs
            axis=1
        )
        y_pred = np.array([self._class_list[i] for i in temp])
        return np.mean(y == y_pred)
