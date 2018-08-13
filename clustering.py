"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import device_lib

import tensorflow as tf
import numpy as np

import os


class Clustering:
    """Abstract base class for clustering algorithms."""

    @staticmethod
    def _distance_argmin(a, b):
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
        self.max_diff_ = None

        # Initial centroids
        self._initial_centroids = None

        # Data set
        self.x = None

        # Expanded version of the data
        self._X = None

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
            :return labels: List of labels, in the same order as the
                    provided data.
        """
        self._dim = x.shape[1]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config) as sess:
            self._size = x.shape[0]

            initial_centroids = self._pre_process(x, sess)

            self._sharded, self._n_shards = \
                _prepare_shards(x.shape[0], x.shape[1],
                                initial_centroids.shape[0])
            if self._sharded:
                tf.logging.info('Data is too large, dividing to {} fragments.'.
                                format(self._n_shards))

            cluster_op, hist_op = self._create_fit_graph()
            tf.summary.FileWriter(
                os.path.join('tensorboard'), sess.graph)

            centroids, self.history_, n_iter, max_diff = sess.run(
                [cluster_op, hist_op, self.n_iter_, self.max_diff_],
                feed_dict={
                    self.x: x,
                    self._initial_centroids: initial_centroids
                }
            )

            tf.logging.info('Clustering finished in {} iterations with '
                            '{:.5} shift delta.'.format(n_iter, max_diff))
            tf.logging.info('Proceeding to post-processing.')
            labels, self.centroids_ = self._post_process(x, centroids, sess)

        return labels

    def _create_predict_graph(self):
        """Creates the prediction computation graph."""
        self.x = tf.placeholder(tf.float32, [None, self._dim], name='data')
        _c = tf.constant(self.centroids_, dtype=tf.float32, name='centroids')
        pred_op = self._distance_argmin(self.x, _c)

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


def _flat(x, x_i, h):
    """Flat kernel function. If the value is within the bounds of
    the bandwidth, it gets weight 1, otherwise 0."""
    return tf.cast(tf.less(
        tf.linalg.norm((x - x_i) / h, ord=2, axis=1), h), tf.float32)


class MeanShift(Clustering):
    """Implementation of mean shift clustering"""

    # Available kernel functions
    _kernel_fns = {
        'gaussian':     _gaussian,
        'flat':         _flat,
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
        self._log_freq = max_iter // 20 if max_iter > 10 else 1

        # Bandwidth
        self._bandwidth = bandwidth

        # Transpose of expanded version of the provided data
        self._X_T = None

        # Size of the provided data
        self._size = None
        self._indices = None

    def _fast_mean_shift(self, index, centroids, history, _):
        """Calculates the mean shift vector and refreshes the centroids.
        This method"""
        ms = self._kernel_fn(tf.expand_dims(centroids, 2),
                             self._X_T, self._bandwidth)

        new_centroids = tf.reduce_sum(
            tf.expand_dims(ms, 2) * self._X, axis=1) / \
            tf.reduce_sum(ms, axis=1, keepdims=True)
        max_diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum(
                (new_centroids - centroids) ** 2, axis=1))), [])

        history = history.write(index + 1, new_centroids)

        return index + 1, new_centroids, history, max_diff

    def _flat_mean_shift(self, index, centroids, history, _):
        """Calculates the mean shift vector and refreshes the centroids.
           This method also fragments the data, since it was determined to
           be excessively large."""
        def shift_fragment(_index, _centroids):
            _ms = self._kernel_fn(
                tf.gather(self._X_T, _index),
                tf.expand_dims(self._sharded_centroids, 2), self._bandwidth)
            _centroids = _centroids.write(
                _index,
                tf.reduce_sum(tf.expand_dims(_ms, 2) *
                              tf.gather(self._X, _index), axis=1) /
                tf.reduce_sum(_ms, axis=1, keepdims=True))

            return _index + 1, _centroids

        _, new_centroids = tf.while_loop(
            cond=lambda i, _: tf.less(i, self._n_shards),
            body=shift_fragment,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                tf.TensorArray(dtype=tf.float32, size=self._n_shards)))

        new_centroids = tf.reshape(
            new_centroids.gather(
                tf.range(self._n_shards)), [self._size, self._dim])
        new_centroids = tf.cond(
            tf.equal(index % self._log_freq, 0),
            lambda: tf.Print(new_centroids, [index],
                             message='Iteration: '),
            lambda: new_centroids)
        max_diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum(
                (new_centroids - centroids) ** 2, axis=1))), [])

        return index + 1, new_centroids, history, max_diff

    def _create_fit_graph(self):
        """Creates the computation graph of the clustering."""
        self.x = tf.placeholder(tf.float32, [self._size, self._dim])
        self._initial_centroids = \
            tf.placeholder(tf.float32, [self._size, self._dim])

        if self._sharded:
            # Sharded version of expanded x and x.T
            self._X_T = [tf.expand_dims(_x_t, 0)
                         for _x_t in tf.split(tf.transpose(self.x),
                                              self._n_shards, 1)]
            self._X = [tf.expand_dims(_x, 0)
                       for _x in tf.split(self.x, self._n_shards, 0)]
            self._sharded_centroids = tf.gather(self.x, tf.random_shuffle(
                np.arange(self._size))[:self._size // self._n_shards])

        else:
            # Normal version of expanded x and x.T
            self._X_T = tf.expand_dims(tf.transpose(self.x), 0)
            self._X = tf.expand_dims(self.x, 0)

        self._bandwidth = tf.constant(self._bandwidth, tf.float32)

        history = tf.TensorArray(dtype=tf.float32, size=self._max_iter)
        history = history.write(0, self._initial_centroids)

        centroids = self._initial_centroids

        mean_shift = self._fast_mean_shift if not self._sharded else \
            self._flat_mean_shift

        self.n_iter_, cluster_op, history, self.max_diff_ = tf.while_loop(
            cond=lambda i, c, h, diff: tf.less(self._criterion, diff),
            body=mean_shift,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                centroids,
                history,
                tf.constant(np.inf, dtype=tf.float32)),
            swap_memory=True,
            maximum_iterations=self._max_iter - 1)

        r = 1 if self._sharded else self.n_iter_ + 1
        history = history.gather(tf.range(r))
        hist_op = tf.cond(
            tf.equal(self.n_iter_, self._max_iter - 1),
            lambda: tf.Print(history, [self.n_iter_],
                             message='Stopping at maximum iterations limit.'),
            lambda: history)

        return cluster_op, hist_op

    def _pre_process(self, x, _):
        """Pre-processing is not needed."""
        return x

    def _post_process(self, _, y, sess):
        """Converts the tensor of converged data points to clusters."""
        def process_point(index, _centroids, num_centroids, _labels):
            """Body of the while loop."""
            def _new(label, __labels, __centroids, _num_centroids):
                """Convenience function for adding new centroid."""
                _cp = tf.concat([__centroids,
                                 tf.expand_dims(y[index], 0)], axis=0)
                return __labels.write(index, label + 1), _cp,\
                    _num_centroids + 1

            def _exists(label, __labels, __centroids, _num_centroids):
                """Convenience function for labeling."""
                return __labels.write(index, label), __centroids, \
                    _num_centroids

            distance = tf.sqrt(tf.reduce_sum(
                (_centroids - y[index]) ** 2, axis=1))
            closest_index = tf.cast(tf.argmin(distance), tf.int32)
            value = tf.gather(distance, closest_index)
            _labels, _centroids, num_centroids = tf.cond(
                tf.less(value, self._bandwidth * tolerance),
                lambda: _exists(closest_index,
                                _labels, _centroids, num_centroids),
                lambda: _new(num_centroids, _labels, _centroids,
                             num_centroids))

            return index + 1, _centroids, num_centroids, _labels

        tolerance = 1.2
        _initial = y[None, 0]
        y = tf.constant(y, dtype=tf.float32)

        # Loop variables:
        #  1. i: Index variable.
        #  2. c: Tensor of discovered centroids (modes).
        #  3. nc: Number of discovered centroids.
        #  4. l: TensorArray of labeled data.

        _, centroids, _, labels = tf.while_loop(
            cond=lambda i, c, nc, l: tf.less(i, self._size),
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

        labels = labels.gather(tf.range(self._size))
        labels, centroids = sess.run([labels, centroids])

        return labels, centroids

    def _variable_bandwidth(self):
        pass


class DynamicMeanShift(MeanShift):
    """Dynamic version of the mean shift clustering."""

    def _fast_mean_shift(self, index, centroids, history, _):
        """Calculates the mean shift vector and refreshes the centroids."""
        ms = self._kernel_fn(tf.expand_dims(centroids, 2),
                             tf.expand_dims(tf.transpose(centroids), 0),
                             self._bandwidth)

        new_centroids = tf.reduce_sum(
            tf.expand_dims(ms, 2) * tf.expand_dims(centroids, 0), axis=1) / \
            tf.reduce_sum(ms, axis=1, keepdims=True)
        max_diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum(
                (new_centroids - centroids) ** 2, axis=1))), [])

        history = history.write(index + 1, new_centroids)

        return index + 1, new_centroids, history, max_diff


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

    def _kmeans(self, index, centroids, history, _):
        """Calculates the next position of the cluster centroids."""
        def step_centroid(_index, _centroids):
            """Moves the centroid to the cluster mean."""
            _centroid_index = tf.where(
                tf.equal(self._distance_argmin(self._X, centroids), _index))
            _centroid = tf.reduce_mean(
                tf.gather(self._X, _centroid_index), axis=0)
            _centroids = tf.concat((_centroids, _centroid), axis=0)

            return _index + 1, _centroids

        _centroid_index_0 = tf.where(
            tf.equal(self._distance_argmin(self._X, centroids), 0))
        _centroid_0 = tf.reduce_mean(
            tf.gather(self._X, _centroid_index_0), axis=0)
        _centroid_0 = tf.reshape(_centroid_0, shape=[1, self._dim])

        _, new_centroids, = tf.while_loop(
            cond=lambda i, c: tf.less(i, self._n_clusters),
            body=step_centroid,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                _centroid_0),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim])))

        new_centroids = tf.reshape(
            new_centroids, shape=[self._n_clusters, self._dim])
        max_diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum(
                (new_centroids - centroids) ** 2, axis=1))), [])

        history = history.write(index + 1, new_centroids)

        return index + 1, new_centroids, history, max_diff

    def _cached_kmeans(self, index, centroids, history, _):
        """Calculates the mean shift vector and refreshes the centroids.
           This method also fragments the data, since it was determined to
           be excessively large."""
        def step_centroid(_index, _centroids):
            """Moves the centroid to the cluster mean."""
            _sharded_centroid_index = []
            for _x_sh in self._X:
                _sharded_centroid_index.append(tf.reshape(tf.where(
                    tf.equal(self._distance_argmin(_x_sh, centroids), _index)),
                    shape=[-1, 1]))
            _centroid_index = tf.reshape(
                tf.concat(_sharded_centroid_index, axis=0), [self._size, 1])
            _centroid = \
                tf.reduce_mean(tf.gather(self.x, _centroid_index), axis=0)
            _centroids = tf.concat((_centroids, _centroid), axis=0)

            return _index + 1, _centroids

        sharded_centroid_index_0 = []
        for _x_sh_0 in self._X:
            sharded_centroid_index_0.append(tf.reshape(tf.where(
                tf.equal(self._distance_argmin(_x_sh_0, centroids), 0)),
                shape=[-1, 1]))
        _centroid_index_0 = tf.reshape(
            tf.concat(sharded_centroid_index_0, axis=0), [self._size, 1])
        _centroid_0 = tf.reduce_mean(
            tf.gather(self.x, _centroid_index_0), axis=0)
        _centroid_0 = tf.reshape(_centroid_0, shape=[1, self._dim])

        _, new_centroids, = tf.while_loop(
            cond=lambda i, c: tf.less(i, self._n_clusters),
            body=step_centroid,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                _centroid_0),
            swap_memory=True,
            parallel_iterations=self._n_parallel,
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim])))

        new_centroids = tf.reshape(
            new_centroids, shape=[self._n_clusters, self._dim])
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum(
                (new_centroids - centroids) ** 2, axis=1))), [])

        return index + 1, new_centroids, history, diff

    def _create_fit_graph(self):
        """Creates the computation graph of the clustering."""
        self.x = tf.placeholder(tf.float32, [self._size, self._dim])
        self._initial_centroids = \
            tf.placeholder(tf.float32, [self._n_clusters, self._dim])

        if self._sharded:
            # Sharded version of expanded x
            self._X = [_x for _x in tf.split(self.x, self._n_shards, 0)]
        else:
            # Normal version of expanded x and x.T
            self._X = self.x

        history = tf.TensorArray(dtype=tf.float32, size=self._max_iter)
        history = history.write(0, self._initial_centroids)

        centroids = self._initial_centroids
        kmeans = self._kmeans if not self._sharded else self._cached_kmeans

        self.n_iter_, cluster_op, history, self.max_diff_ = tf.while_loop(
            cond=lambda i, c, h, diff: tf.less(self._criterion, diff),
            body=kmeans,
            loop_vars=(
                tf.constant(0, tf.int32),
                centroids,
                history,
                tf.constant(np.inf, tf.float32)),
            maximum_iterations=self._max_iter - 1)

        r = 1 if self._sharded else self.n_iter_ + 1
        history = history.gather(tf.range(r))
        hist_op = tf.cond(
            tf.equal(self.n_iter_, self._max_iter - 1),
            lambda: tf.Print(history, [self.n_iter_],
                             message='Stopping at maximum iterations limit.'),
            lambda: history)

        return cluster_op, hist_op

    def _pre_process(self, x, sess):
        """Chooses K random points from the data as initial centroids."""
        assert len(x) >= self._n_clusters, 'Too few data points. K must ' \
                                           'be smaller or equal than the ' \
                                           'number of data points.'
        data_indices = np.arange(len(x))
        np.random.shuffle(data_indices)
        return x[data_indices[:self._n_clusters]]

    def _post_process(self, x, y, sess):
        """No post processing is needed, returning with the centroids,
        and data labels."""
        _y = tf.constant(y, dtype=tf.float32)
        labels = sess.run(self._distance_argmin(self.x, _y),
                          feed_dict={self.x: x})
        return labels, y


class MiniBatchKMeans(KMeans):
    """Implementation of the batched version of K-Means clustering."""

    def __init__(self, batch_size: int,
                 n_clusters: int,
                 criterion: float = 1e-5,
                 max_iter: int = 5000):
        """MiniBatchKMeans clustering object.

        Arguments:
            :param n_clusters: The number of clusters (K parameter).
            :param criterion: Convergence criterion.
            :param max_iter: Maximum number of iterations.
        """
        super(MiniBatchKMeans, self).__init__(n_clusters, criterion,
                                              max_iter)

        assert batch_size is not None

        self._batch_size = batch_size
        self._data_indices = None

    def _pre_process(self, x, sess):
        """Chooses K random points from the data as initial centroids."""
        self._data_indices = tf.range(self._size, dtype=tf.int32)
        return super()._pre_process(x, sess)

    def _kmeans(self, index, centroids, history, _):
        """Calculates the next position of the cluster centroids."""
        def step_centroid(_index, _c):
            """Moves the centroid to the cluster mean."""
            _c_i = tf.where(
                tf.equal(self._distance_argmin(batch, centroids), _index))
            _n_c = tf.reduce_mean(tf.gather(batch, _c_i), axis=0)
            _c = tf.concat((_c, _n_c), axis=0)

            return _index + 1, _c

        shuffled_data_indices = tf.random_shuffle(self._data_indices)
        batch = tf.gather(self.x, shuffled_data_indices[:self._batch_size])
        _centroid_index_0 = tf.where(
            tf.equal(self._distance_argmin(batch, centroids), 0))
        _centroid_0 = tf.reduce_mean(
            tf.gather(batch, _centroid_index_0), axis=0)
        _centroid_0 = tf.reshape(_centroid_0, shape=[1, self._dim])

        _, new_centroids, = tf.while_loop(
            cond=lambda i, c: tf.less(i, self._n_clusters),
            body=step_centroid,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                _centroid_0),
            swap_memory=True,
            parallel_iterations=self._n_parallel,
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim])))

        new_centroids = tf.reshape(
            new_centroids, shape=[self._n_clusters, self._dim])
        diff = tf.reshape(tf.reduce_max(
            tf.sqrt(tf.reduce_sum(
                (new_centroids - centroids) ** 2, axis=1))), [])

        if not self._sharded:
            history = history.write(index + 1, new_centroids)

        return index + 1, new_centroids, history, diff

    def _cached_kmeans(self, index, centroids, history, _):
        """Sharded implementation is not required for batched version."""
        return self._kmeans(index, centroids, history, _)


def estimate_bandwidth():
    pass


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
