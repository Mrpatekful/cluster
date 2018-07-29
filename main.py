"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os

from mean_shift import MeanShift
from kmeans import KMeans
from kmeans import MiniBatchKMeans

tf.flags.DEFINE_float('bandwidth', None, 'Bandwidth of the kernel.')
tf.flags.DEFINE_string('data', None, 'Path of the data for clustering.')
tf.flags.DEFINE_string('kernel', 'gaussian', 'Type of the kernel.')
tf.flags.DEFINE_boolean('gpu', True, 'Use GPU.')
tf.flags.DEFINE_float('criterion', 1e-5, 'Convergence criterion.')
tf.flags.DEFINE_string('method', 'mean_shift', 'Algorithm method.')
tf.flags.DEFINE_string('verbosity', 'INFO', 'Verbosity level.')
tf.flags.DEFINE_integer('batchsize', None, 'Batch size for mini batch method.')
tf.flags.DEFINE_integer('maxiter', 100, 'Maximum number of iterations.')
tf.flags.DEFINE_integer('nclusters', None, 'Number of clusters for KMeans.')
tf.flags.DEFINE_integer('logk', 2, 'Log every k iterations.')


_verbosity_levels = {
    'DEBUG':  tf.logging.DEBUG,
    'INFO':   tf.logging.INFO,
    'WARN':   tf.logging.WARN,
    'ERROR':  tf.logging.ERROR,
    'FATAL':  tf.logging.FATAL,
}


def main(_):
    tf.logging.set_verbosity(
        _verbosity_levels[tf.flags.FLAGS.verbosity])

    assert os.path.exists(tf.flags.FLAGS.data)

    data = np.load(tf.flags.FLAGS.data)
    dim = data.shape[1]

    params = {
        'dim':          dim,
        'criterion':    tf.flags.FLAGS.criterion,
        'max_iter':     tf.flags.FLAGS.maxiter,
    }

    if tf.flags.FLAGS.method == 'mean_shift':
        ms = MeanShift(
            kernel=tf.flags.FLAGS.kernel,
            bandwidth=tf.flags.FLAGS.bandwidth,
            **params)

        ms.fit(data)

    elif tf.flags.FLAGS.method == 'kmeans':
        ms = KMeans(
            n_clusters=tf.flags.FLAGS.n_clusters,
            **params)

        ms.fit(data)

    elif tf.flags.FLAGS.method == 'mini_batch_kmeans':
        ms = MiniBatchKMeans(
            n_clusters=tf.flags.FLAGS.n_clusters,
            batch_size=tf.flags.FLAGS.batchsize,
            **params)

        ms.fit(data)

    else:
        raise ValueError('--method parameter must either '
                         'be < means_shift >'
                         '< mini_batch_mean_shift >,'
                         '< kmeans > or < mini_batch_kmeans >.')


if __name__ == '__main__':
    tf.app.run()
