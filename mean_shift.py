"""

"""

import tensorflow as tf
import numpy as np

from utils import prepare_shards


# Kernel functions:

def gaussian(x, x_i, h):
    """Gaussian kernel function"""
    return tf.exp(-tf.linalg.norm((x - x_i) / h, ord=2, axis=1))


def flat(x, x_i):
    return 0


class MeanShift:
    """Implementation of mean shift clustering"""

    # Available kernel functions
    _kernel_fns = {
        'gaussian':  gaussian,
        'flat':      flat
    }

    def __init__(self, dim: int, kernel: str, bandwidth: float,
                 criterion: float = 1e-5,
                 max_iter: int = 100):
        """Mean shift clustering object.

        Arguments:
            :param dim: Dimensionality of the data point vectors.
            :param kernel: Kernel function type for mean shift calculation.
            :param bandwidth: Bandwidth hyper parameter of the clustering.
            :param criterion: Convergence criterion.
        """

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

        self._criterion = criterion
        self._max_iter = max_iter
        self._dim = dim
        self._merged = None

        # Result variables
        self.centroids_ = None
        self._c_c_tensor = None
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

    def fit(self, x: np.ndarray):
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
            crs, hist, it, diff = sess.run(
                [*self._create_fit_graph(), self.n_iter_, self.m_diff_],
                feed_dict={
                    self.x:    x,
                    self._i_c: x
                })

            tf.logging.info('Clustering finished in {} iterations with '
                            '{:.5} error'.format(it, diff))
            tf.logging.info('Proceeding mode selection.')
            labels, self.centroids_ = self._find_modes(crs, x, sess)

        return labels

    def fit_and_predict(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray):
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
            self._sharded, self._n_shards = \
                prepare_shards(x.shape[0], x.shape[1])
            if self._sharded:
                tf.logging.info('Data is too large, fragmenting.'
                                ' Dividing to {} fragments.'.
                                format(self._n_shards))
            crs, hist, it, diff = sess.run(
                [*self._create_fit_graph(), self.n_iter_, self.m_diff_],
                feed_dict={
                    self.x:    x,
                    self._i_c: x
                })

            tf.logging.info('Clustering finished in {} iterations with '
                            '{:.5} error'.format(it, diff))
            tf.logging.info('Proceeding mode selection.')
            labels, self.centroids_ = self._find_modes(crs, x, sess)

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
        with tf.name_scope('init'):
            self.x = tf.placeholder(
                tf.float32, [self._size, self._dim], name='data')
            self._i_c = tf.placeholder(
                tf.float32, [self._size, self._dim], name='init_c')

        if self._sharded:
            with tf.name_scope('sh_data'):
                # Sharded version of expanded x and x.T
                self._x_t_sh = [tf.expand_dims(_x_t, 0)
                                for _x_t in tf.split(tf.transpose(self.x),
                                                     self._n_shards, 1)]
                self._x_sh = [tf.expand_dims(_x, 0)
                              for _x in tf.split(self.x, self._n_shards, 0)]
        else:
            with tf.name_scope('pp_data'):
                # Normal version of expanded x and x.T
                self._x_t = tf.expand_dims(tf.transpose(self.x), 0)
                self._x = tf.expand_dims(self.x, 0)

        self._h = tf.constant(self._bandwidth, tf.float32, name='bw')
        i = tf.constant(0, tf.int32)

        # TODO benchmark device placement
        with tf.device("/cpu:0"):
            _c_h = tf.TensorArray(dtype=tf.float32, size=self._max_iter,
                                  infer_shape=False)
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
        pass

    def _find_modes(self, y, x, sess):
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
            distance = tf.cond(
                tf.equal(_i % log_freq, 0),
                lambda: tf.Print(distance, [_i],
                                 message='Progress: 100/'),
                lambda: distance)
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

        log_freq = self._size // 5 if self._size > 10 else 1

        # Loop variables:
        #  1. i: Index variable.
        #  2. c: Tensor of discovered centroids (modes).
        #  3. n_c: Number of discovered centroids.
        #  4. l: TensorArray of labeled data.

        i = tf.constant(1, dtype=tf.int32)
        c = tf.constant(_initial, dtype=tf.float32)
        n_c = tf.constant(0, dtype=tf.int32)
        _labels = tf.TensorArray(dtype=tf.int32, size=self._size)

        _, centroids, _, labels = tf.while_loop(
            cond=lambda __i, __c, __n_c, __l: tf.less(__i, self._size),
            body=_modes,
            loop_vars=(i, c, n_c, _labels),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, self._dim]),
                tf.TensorShape([]),
                tf.TensorShape([])
            )
        )

        labels = labels.gather(tf.range(self._size))
        labels, centroids = sess.run([labels, centroids])

        return labels, centroids

    @property
    def centroids(self):
        """Property for the tensor of clusters."""
        return self.centroids_

    @property
    def history(self):
        """Property for the history of the cluster centroids."""
        return self.history_
