"""

"""

import tensorflow as tf
import numpy as np

from tensorflow.python.client import device_lib


class Clustering:
    """Abstract base class for clustering algorithms."""

    @staticmethod
    def _nearest(a, b):
        """Finds the indexes of the closest point (Euclidean) for each point in
        tensor 'a' from the points of tensor 'b'."""
        return tf.cast(tf.argmin(
            tf.reduce_sum((tf.expand_dims(a, 2) - tf.expand_dims(
                tf.transpose(b), 0)) ** 2, axis=1), axis=1), dtype=tf.int32)

    def __init__(self, criterion, max_iter):
        """Abstract base class for clustering algorithms."""
        self._criterion = criterion
        self._max_iter = max_iter
        self._dim = None

        # Result variables
        self.centroids_ = None
        self.history_ = None
        self.n_iter_ = None
        self.m_diff_ = None

        # Initial centroids
        self._i_c = None

        self.x = None

        # Number of shards
        self._n_shards = None
        self._sharded = None

        self._size = None

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

    def fit(self, x: np.ndarray) -> np.ndarray:
        """Fits the MeanShift cluster object to the given data set.

        Arguments:
            :param x: 2D Numpy array, that contains the data set.

        Returns:
            :return _y: List of labels, in the same order as the provided data.
        """
        self._dim = x.shape[1]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config) as sess:
            self._size = x.shape[0]

            c = self._pre_process(x, sess)

            self._sharded, self._n_shards = \
                _prepare_shards(x.shape[0], x.shape[1], c.shape[0])
            if self._sharded:
                tf.logging.info('Data is too large, dividing to {} fragments.'.
                                format(self._n_shards))

            _c, self.history_, it, diff = sess.run(
                [*self._create_fit_graph(), self.n_iter_, self.m_diff_],
                feed_dict={
                    self.x:    x,
                    self._i_c: c
                })

            tf.logging.info('Clustering finished in {} iterations with '
                            '{:.5} shift delta.'.format(it, diff))
            tf.logging.info('Proceeding to post-processing.')
            _y, self.centroids_ = self._post_process(x, _c, sess)

        return _y

    def _create_predict_graph(self):
        """Creates the prediction computation graph."""
        self.x = tf.placeholder(tf.float32, [None, self._dim], name='data')
        _c = tf.constant(self.centroids_, dtype=tf.float32, name='centroids')
        pred_op = self._nearest(self.x, _c)

        return pred_op

    def _create_fit_graph(self):
        """Returns the operation of the clustering algorithm."""
        raise NotImplementedError

    def _pre_process(self, x, sess):
        """Returns the initial cluster centroids for the clustering."""
        raise NotImplementedError

    def _post_process(self, x, y, sess):
        """Processes the output tensor of the clustering. The output
        is the array of centroids."""
        raise NotImplementedError

    @property
    def centroids(self):
        """Property for the array of clusters. (num_clusters, dim) array."""
        return self.centroids_

    @property
    def history(self):
        """Property for the history of the cluster centroids. If
        the data size is sufficiently small, the improvements of
        the centroids will be stored as (num_it, num_clusters, dim) array."""
        return self.history_


# Kernel functions for non-parametric density estimation.
# In order to be able to use new kernel, it has to also be
# registered in the MeanShift _kernel_fns dictionary.

def _gaussian(x, x_i, h):
    """Gaussian kernel function. """
    return tf.exp(-tf.linalg.norm((x - x_i) / h, ord=2, axis=1))


def _epanechnikov(x, x_i, h):
    """Epanechnikov kernel function."""
    norm = tf.linalg.norm((x - x_i) / h, ord=2, axis=1)
    return tf.multiply(3 / 4 * (1 - norm),
                       tf.cast(tf.less(norm, 1), tf.float32))


def _uniform(x, x_i, h):
    """Flat kernel function. If the value is within the bounds of
    the bandwidth, it gets weight 1, otherwise 0."""
    return tf.cast(tf.less(
        tf.linalg.norm((x - x_i) / h, ord=2, axis=1), 1), tf.float32)


class MeanShift(Clustering):
    """Implementation of mean shift clustering"""

    # Available kernel functions
    _kernel_fns = {
        'gaussian':     _gaussian,
        'uniform':      _uniform,
        'epanechnikov': _epanechnikov
    }

    # noinspection PyTypeChecker
    def __init__(self,
                 bandwidth: float,
                 kernel: str = 'gaussian',
                 criterion: float = 1e-5,
                 max_iter: int = 300):
        """Mean shift clustering object.

        Arguments:
            :param kernel: Kernel function type for mean shift calculation.
            :param bandwidth: Bandwidth hyper parameter of the clustering.
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(MeanShift, self).__init__(criterion, max_iter)

        assert self._kernel_fns.get(kernel) is not None, 'Invalid kernel.'
        assert bandwidth is not None

        self._kernel_fn = self._kernel_fns[kernel]
        self._bandwidth = bandwidth
        self._log_freq = max_iter // 20 if max_iter > 10 else 1

        # Bandwidth
        self._h = None

        self._x = None
        self._x_t = None

        self._size = None

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
        for _x, _x_t, _c in zip(self._x, self._x_t, c_sh):
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
        self.x = tf.placeholder(tf.float32, [self._size, self._dim])
        self._i_c = tf.placeholder(tf.float32, [self._size, self._dim])

        if self._sharded:
            # Sharded version of expanded x and x.T
            self._x_t = [tf.expand_dims(_x_t, 0)
                         for _x_t in tf.split(tf.transpose(self.x),
                                              self._n_shards, 1)]
            self._x = [tf.expand_dims(_x, 0)
                       for _x in tf.split(self.x, self._n_shards, 0)]
        else:
            # Normal version of expanded x and x.T
            self._x_t = tf.expand_dims(tf.transpose(self.x), 0)
            self._x = tf.expand_dims(self.x, 0)

        self._h = tf.constant(self._bandwidth, tf.float32)

        _c_h = tf.TensorArray(dtype=tf.float32, size=self._max_iter)
        _c_h = _c_h.write(0, self._i_c)

        self.m_diff_ = tf.constant(np.inf, tf.float32)

        _c = self._i_c
        mean_shift = self._mean_shift if not self._sharded else \
            self._sharded_mean_shift

        self.n_iter_, cluster_op, _c_h, self.m_diff_ = tf.while_loop(
            cond=lambda __i, __c, __c_h, diff: tf.less(self._criterion, diff),
            body=mean_shift,
            loop_vars=(
                tf.constant(0, tf.int32),
                _c,
                _c_h,
                self.m_diff_),
            swap_memory=True,
            maximum_iterations=self._max_iter - 1)

        r = 1 if self._sharded else self.n_iter_ + 1
        hist_op = _c_h.gather(tf.range(r))

        return cluster_op, hist_op

    def _pre_process(self, x, _):
        """Pre-processing is not needed."""
        return x

    def _post_process(self, _, y, sess):
        """Converts the tensor of converged data points to clusters."""
        def process_point(_i, _c, _n_c, _l):
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

        _, _centroids, _, _labels = tf.while_loop(
            cond=lambda __i, __c, __n_c, __l: tf.less(__i, self._size),
            body=process_point,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                tf.constant(_initial, dtype=tf.float32),
                tf.constant(0, dtype=tf.int32),
                tf.TensorArray(dtype=tf.int32, size=self._size)),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim]),
                tf.TensorShape([]),
                tf.TensorShape([])))

        _labels = _labels.gather(tf.range(self._size))
        labels, centroids = sess.run([_labels, _centroids])

        return labels, centroids


class KMeans(Clustering):
    """Implementation of K-Means clustering."""

    def __init__(self, n_clusters: int,
                 criterion: float = 1e-5,
                 max_iter: int = 300):
        """KMeans clustering object.

        Arguments:
            :param n_clusters: The number of clusters (K parameter).
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(KMeans, self).__init__(criterion, max_iter)

        assert n_clusters is not None
        self._n_clusters = n_clusters
        self._n_parallel = 1 if self._sharded else 10

    def _kmeans(self, i, c, c_h, _):
        """Calculates the next position of the cluster centroids."""
        def step_centroid(_i, _c):
            """Moves the centroid to the cluster mean."""
            _c_i = tf.where(tf.equal(self._nearest(self._x, c), _i))
            _n_c = tf.reduce_mean(tf.gather(self._x, _c_i), axis=0)
            _c = tf.concat((_c, _n_c), axis=0)

            return _i + 1, _c

        _c_i_0 = tf.where(tf.equal(self._nearest(self._x, c), 0))
        _c_0 = tf.reduce_mean(tf.gather(self._x, _c_i_0), axis=0)
        _c_0 = tf.reshape(_c_0, shape=[1, self._dim])

        _, n_c, = tf.while_loop(
            cond=lambda _i, __c: tf.less(_i, self._n_clusters),
            body=step_centroid,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                _c_0),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim])))

        n_c = tf.reshape(n_c, shape=[self._n_clusters, self._dim])
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum((n_c - c) ** 2, axis=1))), [])

        c_h = c_h.write(i + 1, n_c)

        return i + 1, n_c, c_h, diff

    def _sharded_kmeans(self, i, c, c_h, _):
        """Calculates the mean shift vector and refreshes the centroids.
           This method also fragments the data, since it was determined to
           be excessively large."""
        def step_centroid(_i, _c):
            """Moves the centroid to the cluster mean."""
            _c_i = []
            for _x_sh in self._x:
                _c_i.append(tf.reshape(tf.where(
                    tf.equal(self._nearest(_x_sh, c), _i)), shape=[-1, 1]))
            _c_i = tf.reshape(tf.concat(_c_i, axis=0), [self._size, 1])
            _n_c = tf.reduce_mean(tf.gather(self.x, _c_i), axis=0)
            _c = tf.concat((_c, _n_c), axis=0)

            return _i + 1, _c

        _c_i_0 = []
        for _x_sh_0 in self._x:
            _c_i_0.append(tf.reshape(tf.where(
                tf.equal(self._nearest(_x_sh_0, c), 0)), shape=[-1, 1]))
        _c_i_0 = tf.reshape(tf.concat(_c_i_0, axis=0), [self._size, 1])
        _c_0 = tf.reduce_mean(tf.gather(self.x, _c_i_0), axis=0)
        _c_0 = tf.reshape(_c_0, shape=[1, self._dim])

        _, n_c, = tf.while_loop(
            cond=lambda _i, __c: tf.less(_i, self._n_clusters),
            body=step_centroid,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                _c_0),
            swap_memory=True,
            parallel_iterations=self._n_parallel,
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim])))

        n_c = tf.reshape(n_c, shape=[self._n_clusters, self._dim])
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum((n_c - c) ** 2, axis=1))), [])

        return i + 1, n_c, c_h, diff

    def _create_fit_graph(self):
        """Creates the computation graph of the clustering."""
        self.x = tf.placeholder(tf.float32, [self._size, self._dim])
        self._i_c = tf.placeholder(tf.float32, [self._n_clusters, self._dim])

        if self._sharded:
            # Sharded version of expanded x
            self._x = [_x for _x in tf.split(self.x, self._n_shards, 0)]
        else:
            # Normal version of expanded x and x.T
            self._x = self.x

        _c_h = tf.TensorArray(dtype=tf.float32, size=self._max_iter)
        _c_h = _c_h.write(0, self._i_c)

        _c = self._i_c
        kmeans = self._kmeans if not self._sharded else self._sharded_kmeans

        self.n_iter_, cluster_op, _c_h, self.m_diff_ = tf.while_loop(
            cond=lambda __i, __c, __c_h, diff: tf.less(self._criterion, diff),
            body=kmeans,
            loop_vars=(
                tf.constant(0, tf.int32),
                _c,
                _c_h,
                tf.constant(np.inf, tf.float32)),
            maximum_iterations=self._max_iter - 1)

        r = 1 if self._sharded else self.n_iter_ + 1
        _c_h = _c_h.gather(tf.range(r))
        hist_op = tf.cond(
            tf.equal(self.n_iter_, self._max_iter - 1),
            lambda: tf.Print(_c_h, [self.n_iter_],
                             message='Stopping at maximum iterations limit.'),
            lambda: _c_h)

        return cluster_op, hist_op

    def _pre_process(self, x, sess):
        """Chooses K random points from the data as initial centroids."""
        assert len(x) >= self._n_clusters, 'Too few data points. K must ' \
                                           'be smaller or equal than the ' \
                                           'number of data points.'
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx[:self._n_clusters]]

    def _post_process(self, x, y, sess):
        """No post processing is needed, returning with the centroids,
        and data labels."""
        _y = tf.constant(y, dtype=tf.float32)
        labels = sess.run(self._nearest(self.x, _y), feed_dict={self.x: x})
        return labels, y


class MiniBatchKMeans(KMeans):
    """Implementation of the batched version of K-Means clustering."""

    def __init__(self, batch_size: int,
                 n_clusters: int,
                 criterion: float = 1e-5,
                 max_iter: int = 5000):
        """MiniBatchKMeans clustering object.

        Arguments:
            :param dim: Dimensionality of the data point vectors.
            :param n_clusters: The number of clusters (K parameter).
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(MiniBatchKMeans, self).__init__(n_clusters, criterion,
                                              max_iter)

        assert batch_size is not None

        self._batch_size = batch_size
        self._idx = None

    def _pre_process(self, x, sess):
        """Chooses K random points from the data as initial centroids."""
        self._idx = tf.range(self._size, dtype=tf.int32)
        return super()._pre_process(x, sess)

    def _kmeans(self, i, c, c_h, _):
        """Calculates the next position of the cluster centroids."""
        def step_centroid(_i, _c):
            """Moves the centroid to the cluster mean."""
            _c_i = tf.where(tf.equal(self._nearest(x_b, c), _i))
            _n_c = tf.reduce_mean(tf.gather(x_b, _c_i), axis=0)
            _c = tf.concat((_c, _n_c), axis=0)

            return _i + 1, _c

        idx = tf.random_shuffle(self._idx)
        x_b = tf.gather(self.x, idx[:self._batch_size])
        _c_i_0 = tf.where(tf.equal(self._nearest(x_b, c), 0))
        _c_0 = tf.reduce_mean(tf.gather(x_b, _c_i_0), axis=0)
        _c_0 = tf.reshape(_c_0, shape=[1, self._dim])

        _, n_c, = tf.while_loop(
            cond=lambda _i, __c: tf.less(_i, self._n_clusters),
            body=step_centroid,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                _c_0),
            swap_memory=True,
            parallel_iterations=self._n_parallel,
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim])))

        n_c = tf.reshape(n_c, shape=[self._n_clusters, self._dim])
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum((n_c - c) ** 2, axis=1))), [])

        if not self._sharded:
            c_h = c_h.write(i + 1, n_c)

        return i + 1, n_c, c_h, diff

    def _sharded_kmeans(self, i, c, c_h, _):
        """Sharded implementation is not required for batched version."""
        return self._kmeans(i, c, c_h, _)


def _query_memory():
    """Queries the available memory."""
    local_device_protos = device_lib.list_local_devices()
    return [x.memory_limit for x in local_device_protos
            if x.device_type == 'GPU']


def _prepare_shards(size, dim, n_clusters):
    """Prepares the number of shards to divide the data into."""
    available_memory = _query_memory()

    # TODO multi-gpu usage
    available_memory = available_memory[0]
    required_memory = 4 * size * n_clusters * dim

    sharded = (required_memory / available_memory) > 1
    if not sharded:
        return False, None

    shard_size = int(size // (required_memory / available_memory))
    n_shards = size // [d for d in
                        range(1, shard_size) if size % d == 0][-1]
    return True, n_shards
