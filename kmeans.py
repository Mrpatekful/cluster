"""

"""

import tensorflow as tf


class KMeans:

    def __init__(self, dim, n_clusters,
                 criterion=1e-5,
                 max_iterations=100,
                 gpu=True):
        pass

    def fit(self, x):
        pass


class MiniBatchKMeans(KMeans):

    def __init__(self, batch_size,
                 n_clusters, dim,
                 criterion=1e-5,
                 max_iterations=100,
                 gpu=True):
        """

        Arguments:
            :param dim:
            :param n_clusters:
            :param criterion:
            :param gpu:
            :param batch_size:
        """
        super(MiniBatchKMeans, self).__init__(
            dim=dim,
            n_clusters=n_clusters,
            criterion=criterion,
            max_iterations=max_iterations,
            gpu=gpu)

        self._batch_size = batch_size

    def fit(self, x):
        pass
