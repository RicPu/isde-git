import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, value):
        self._centroids = value

    @property
    def class_labels(self):
        return self._class_labels

    @class_labels.setter
    def class_labels(self, value):
        self._class_labels = value

    def fit(self, xtr, ytr):
        """
        Trains the classifier by computing the mean (centroid) of each class based on the training data.

        Parameters
        ----------
        xtr : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the training data, where each row corresponds to a sample.
        ytr : np.ndarray
            A 1D array of shape (n_samples,) representing the class labels for the training data.

        Returns
        -------
        self : NMC
            The trained classifier.
        """
        self.class_labels = np.unique(ytr)
        n_classes = self.class_labels.size
        self.centroids = np.zeros(shape=(n_classes, xtr.shape[1]))
        for k in range(n_classes):
            idx = xtr[ytr == self.class_labels[k], :]
            self.centroids[k, :] = np.mean(idx, axis=0)
        return self

    def predict(self, xts):
        pass
