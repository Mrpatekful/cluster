"""

"""

import tensorflow as tf
import numpy as np

from tensorflow.python.client import device_lib


# Kernel functions for non-parametric density estimation.
# In order to be able to use new kernel, it has to also be
# registered in the MeanShift _kernel_fns dictionary.

def gaussian(x, x_i, h):
    """Gaussian kernel function"""
    return tf.exp(-tf.linalg.norm((x - x_i) / h, ord=2, axis=1))


def flat(x, x_i):
    return 0


class Clustering:
    """Abstract base class for clustering algorithms."""

    @staticmethod
    def _nearest(x, c): return tf.argmin(
        tf.reduce_sum((tf.expand_dims(x, 2) - tf.expand_dims(
            tf.transpose(c), 0)) ** 2, axis=1), axis=1)

    def __init__(self, dim, criterion, max_iter):
        """Abstract base class for clustering algorithms."""
        self._criterion = criterion
        self._max_iter = max_iter
        self._dim = dim

    def fit(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_and_predict(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanShift(Clustering):
    """Implementation of mean shift clustering"""

    # Available kernel functions
    _kernel_fns = {
        'gaussian':  gaussian,
        'flat':      flat
    }

    def __init__(self, dim: int, kernel: str,
                 bandwidth: float,
                 criterion: float = 1e-5,
                 max_iter: int = 100):
        """Mean shift clustering object.

        Arguments:
            :param dim: Dimensionality of the data point vectors.
            :param kernel: Kernel function type for mean shift calculation.
            :param bandwidth: Bandwidth hyper parameter of the clustering.
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(MeanShift, self).__init__(dim, criterion, max_iter)

        assert self._kernel_fns.get(kernel) is not None, 'Invalid kernel.'
        assert bandwidth is not None

        self._kernel_fn = self._kernel_fns[kernel]
        self._bandwidth = bandwidth
        self._log_freq = max_iter // 10 if max_iter > 10 else 1

        self.x = None
        self._x = None
        self._x_t = None

        # Bandwidth
        self._h = None

        # Initial centroids
        self._i_c = None

        # Result variables
        self.centroids_ = None
        self.history_ = None
        self.n_iter_ = None
        self.m_diff_ = None

        # Number of shards
        self._n_shards = None
        self._sharded = None

        # Sharded input data tensors
        self._x_t_sh = None
        self._x_sh = None
        self._size = None

    def fit(self, x: np.ndarray) -> np.ndarray:
        """Fits the MeanShift cluster object to the given data set.

        Arguments:
            :param x: 2D Numpy array, that contains the data set.

        Returns:
            :return _y: List of labels, in the same order as the provided data.
        """
        assert x.shape[1] == self._dim, 'Invalid data dimension. Expected' \
                                        '{} and received {} for axis 1.'.\
            format(self._dim, x.shape[1])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config) as sess:
            self._size = x.shape[0]
            self._sharded, self._n_shards = \
                prepare_shards(x.shape[0], x.shape[1])
            if self._sharded:
                tf.logging.info('Data is too large, fragmenting.'
                                ' Dividing to {} fragments.'.
                                format(self._n_shards))

            crs, self.history_, it, diff = sess.run(
                [*self._create_fit_graph(), self.n_iter_, self.m_diff_],
                feed_dict={
                    self.x:    x,
                    self._i_c: x
                })

            tf.logging.info('Clustering finished in {} iterations with '
                            '{:.5} error'.format(it, diff))
            tf.logging.info('Proceeding mode selection.')
            _y, self.centroids_ = self._find_modes(crs, sess)

        return _y

    def fit_and_predict(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calculates the nearest cluster centroid for the data points.

        Arguments:
            :param x: 2D Numpy array with the data points.

        Return:
            :return _y: List of labels, in the same order as the provided data.
        """
        assert self.centroids_ is not None
        assert x.shape[1] == self._dim, 'Invalid data dimension. Expected' \
                                        '{} and received {} for axis 1.'. \
            format(self._dim, x.shape[1])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config) as sess:
            self._size = x.shape[0]
            if self._sharded:
                tf.logging.info('Data is too large, fragmenting.'
                                ' Dividing to {} fragments.'.
                                format(self._n_shards))
            labels = sess.run(self._create_predict_graph(),
                              feed_dict={self.x: x})

        return labels

    def _mean_shift(self, i, c, c_h, _):
        """Calculates the mean shift vector and refreshes the centroids."""
        ms = self._kernel_fn(tf.expand_dims(c, 2), self._x_t, self._h)
        # New centroids
        n_c = tf.reduce_sum(tf.expand_dims(ms, 2) * self._x, axis=1) / \
            tf.reduce_sum(ms, axis=1, keepdims=True)
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum((n_c - c) ** 2, axis=1))), [])

        c_h = c_h.write(i + 1, n_c)

        return i + 1, n_c, c_h, diff

    def _sharded_mean_shift(self, i, c, c_h, _):
        """Calculates the mean shift vector and refreshes the centroids.
           This method also fragments the data, since it was determined to
           be excessively large."""
        c_sh = tf.split(c, self._n_shards, 0)
        # New (sharded) centroids
        n_c = []
        for _x, _x_t, _c in zip(self._x_sh, self._x_t_sh, c_sh):
            _ms = self._kernel_fn(tf.expand_dims(_c, 2), _x_t, self._h)
            n_c.append(tf.reduce_sum(tf.expand_dims(_ms, 2) * _x, axis=1) /
                       tf.reduce_sum(_ms, axis=1, keepdims=True))

        n_c = tf.stack(tf.reshape(n_c, [self._size, self._dim]))
        n_c = tf.cond(
            tf.equal(i % self._log_freq, 0),
            lambda: tf.Print(n_c, [i], message='Iteration: '),
            lambda: n_c)
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum((n_c - c) ** 2, axis=1))), [])

        return i + 1, n_c, c_h, diff

    def _create_fit_graph(self):
        """Creates the computation graph of the clustering."""
        self.x = tf.placeholder(
            tf.float32, [self._size, self._dim], name='data')
        self._i_c = tf.placeholder(
            tf.float32, [self._size, self._dim], name='init')

        if self._sharded:
            # Sharded version of expanded x and x.T
            self._x_t_sh = [tf.expand_dims(_x_t, 0)
                            for _x_t in tf.split(tf.transpose(self.x),
                                                 self._n_shards, 1)]
            self._x_sh = [tf.expand_dims(_x, 0)
                          for _x in tf.split(self.x, self._n_shards, 0)]
        else:
            # Normal version of expanded x and x.T
            self._x_t = tf.expand_dims(tf.transpose(self.x), 0)
            self._x = tf.expand_dims(self.x, 0)

        self._h = tf.constant(self._bandwidth, tf.float32, name='bw')
        i = tf.constant(0, tf.int32)

        _c_h = tf.TensorArray(dtype=tf.float32, size=self._max_iter)
        _c_h = _c_h.write(0, self._i_c)

        self.m_diff_ = tf.constant(np.inf, tf.float32, name='diff')

        _c = self._i_c
        mean_shift = self._mean_shift if not self._sharded else \
            self._sharded_mean_shift

        self.n_iter_, _c, _c_h, self.m_diff_ = tf.while_loop(
            cond=lambda __i, __c, __c_h, diff: tf.less(self._criterion, diff),
            body=mean_shift,
            loop_vars=(i, _c, _c_h, self.m_diff_),
            maximum_iterations=self._max_iter)

        r = 1 if self._sharded else self.n_iter_ + 1
        _c_h = _c_h.gather(tf.range(r))

        return _c, _c_h

    def _create_predict_graph(self):
        """Creates the prediction computation graph."""
        self.x = tf.placeholder(tf.float32, [None, self._dim], name='data')
        _c = tf.constant(self.centroids_, dtype=tf.float32, name='centroids')
        prediction_op = self._nearest(self.x, _c)

        return prediction_op

    def _find_modes(self, y, sess):
        """Converts the tensor of converged data points to clusters."""
        def _modes(_i, _c, _n_c, _l):
            """Body of the while loop."""
            def _new(_l, _lt, _cp, n):
                """Convenience function for adding new centroid."""
                _cp = tf.concat([_cp, tf.expand_dims(y[_i], 0)], axis=0)
                return _lt.write(_i, _l + 1), _cp, n + 1

            def _exists(_l, _lt, _cp, n):
                """Convenience function for labeling."""
                return _lt.write(_i, _l), _cp, n

            distance = tf.sqrt(tf.reduce_sum((_c - y[_i]) ** 2, axis=1))
            least = tf.cast(tf.argmin(distance), tf.int32)
            value = tf.gather(distance, least)
            _l, _c, _n_c = tf.cond(
                tf.less(value, self._bandwidth * tolerance),
                lambda: _exists(least, _l, _c, _n_c),
                lambda: _new(_n_c, _l, _c, _n_c))

            return _i + 1, _c, _n_c, _l

        tolerance = 1.2
        _initial = y[None, 0]
        y = tf.constant(y, dtype=tf.float32)

        # Loop variables:
        #  1. i: Index variable.
        #  2. c: Tensor of discovered centroids (modes).
        #  3. n_c: Number of discovered centroids.
        #  4. l: TensorArray of labeled data.

        i = tf.constant(1, dtype=tf.int32)
        c = tf.constant(_initial, dtype=tf.float32)
        n_c = tf.constant(0, dtype=tf.int32)
        _labels = tf.TensorArray(dtype=tf.int32, size=self._size)

        _, _centroids, _, _labels = tf.while_loop(
            cond=lambda __i, __c, __n_c, __l: tf.less(__i, self._size),
            body=_modes,
            loop_vars=(i, c, n_c, _labels),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim]),
                tf.TensorShape([]),
                tf.TensorShape([])))

        _labels = _labels.gather(tf.range(self._size))
        labels, centroids = sess.run([_labels, _centroids])

        return labels, centroids

    @property
    def centroids(self):
        """Property for the tensor of clusters."""
        return self.centroids_

    @property
    def history(self):
        """Property for the history of the cluster centroids."""
        return self.history_


class KMeans(Clustering):
    """Implementation of K-Means clustering."""

    def __init__(self, dim: int,
                 n_clusters: int,
                 criterion: float = 1e-5,
                 max_iter: int = 100):
        """KMeans clustering object.

        Arguments:
            :param dim: Dimensionality of the data point vectors.
            :param n_clusters: The number of clusters (K parameter).
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(KMeans, self).__init__(dim, criterion, max_iter)

        self._n_clusters = n_clusters

        self.centroids_ = None

    def fit(self, x):
        raise NotImplementedError

    def fit_and_predict(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def centroids(self):
        """Property for the tensor of clusters."""
        return self.centroids_


class MiniBatchKMeans(KMeans):
    """Implementation of the batched version of K-Means clustering."""

    def __init__(self, dim: int,
                 batch_size: int,
                 n_clusters: int,
                 criterion: float = 1e-5,
                 max_iter: int = 100):
        """MiniBatchKMeans clustering object.

        Arguments:
            :param dim: Dimensionality of the data point vectors.
            :param n_clusters: The number of clusters (K parameter).
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(MiniBatchKMeans, self).__init__(dim, n_clusters, criterion,
                                              max_iter)

        self._batch_size = batch_size

    def fit(self, x):
        raise NotImplementedError

    def fit_and_predict(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def centroids(self):
        """Property for the tensor of clusters."""
        return self.centroids_


def query_memory():
    """Queries the available memory."""
    local_device_protos = device_lib.list_local_devices()
    return [x.memory_limit for x in local_device_protos
            if x.device_type == 'GPU']


def prepare_shards(size, dim):
    """Prepares the number of shards to divide the data into."""
    available_memory = query_memory()

    # TODO multi-gpu use
    available_memory = available_memory[0]
    required_memory = 4 * size ** 2 * dim

    sharded = (required_memory / available_memory) > 1
    if not sharded:
        return False, None

    shard_size = int(size // (required_memory / available_memory))
    n_shards = size // [d for d in
                        range(1, shard_size) if size % d == 0][-1]
    return True, n_shards
