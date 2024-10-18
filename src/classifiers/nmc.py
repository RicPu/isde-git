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

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        pass

    def predict(self, xts):
        """
        Predicts the class labels for the given test data based on the nearest centroid.

        Parameters
        ----------
        xts : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the test data to classify.

        Returns
        -------
        np.ndarray
            A 1D array of shape (n_samples,) containing the predicted class labels for each sample.

        Raises
        ------
        ValueError
            If the classifier has not been trained (i.e., if centroids are None).
        """
        if self.centroids is None:
            raise ValueError("The classifier is not trained. Call fit!")

        dist_euclidean = euclidean_distances(xts, self.centroids)
        idx_min = np.argmin(dist_euclidean, axis=1)
        y_pred = self._class_labels[idx_min]
        return y_pred
