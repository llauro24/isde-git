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
        '''
        Compute the averege of each centroid
        '''
        # Identify unique class labels
        self._class_labels = np.unique(ytr)

        # Number of classes is the number of unique labels
        n_classes = self._class_labels.size

        # Number of features is the number of columns in xtr
        n_features = xtr.shape[1]

        # Initialize centroids array
        self._centroids = np.zeros((n_classes, n_features))

        # Compute centroids for each class
        for idx, label in enumerate(self._class_labels):
            # Get all samples belonging to the current class
            class_samples = xtr[ytr == label]

            # Compute the mean of the samples for the current class
            self._centroids[idx] = np.mean(class_samples, axis=0)

    def predict(self, xts):
        pass
