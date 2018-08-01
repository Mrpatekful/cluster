"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os

from clustering import MeanShift
from clustering import KMeans
from clustering import MiniBatchKMeans
from utils import plot
from utils import load, save

tf.flags.DEFINE_float('bandwidth', None, 'Bandwidth of the kernel.')
tf.flags.DEFINE_string('data', None, 'Path of the data for clustering.')
tf.flags.DEFINE_string('kernel', 'gaussian', 'Type of the kernel.')
tf.flags.DEFINE_float('criterion', 1e-5, 'Convergence criterion.')
tf.flags.DEFINE_string('method', 'mini_batch_kmeans', 'Algorithm method.')
tf.flags.DEFINE_string('verbosity', 'INFO', 'Verbosity level.')
tf.flags.DEFINE_integer('batchsize', None, 'Batch size for mini batch method.')
tf.flags.DEFINE_integer('maxiter', 100, 'Maximum number of iterations.')
tf.flags.DEFINE_integer('nclusters', None, 'Number of clusters for KMeans.')
tf.flags.DEFINE_string('save', '.', 'Saves the output to the given directory.')

_verbosity_levels = {
    'DEBUG':  tf.logging.DEBUG,
    'INFO':   tf.logging.INFO,
    'WARN':   tf.logging.WARN,
    'ERROR':  tf.logging.ERROR,
    'FATAL':  tf.logging.FATAL,
}


def main(_):
    tf.logging.set_verbosity(_verbosity_levels[tf.flags.FLAGS.verbosity])

    assert os.path.exists(tf.flags.FLAGS.data)

    data = load(tf.flags.FLAGS.data)

    params = {
        'criterion':    tf.flags.FLAGS.criterion,
        'max_iter':     tf.flags.FLAGS.maxiter,
    }

    if tf.flags.FLAGS.method == 'mean_shift':
        cl = MeanShift(
            kernel=tf.flags.FLAGS.kernel,
            bandwidth=tf.flags.FLAGS.bandwidth,
            **params)

        labels = cl.fit(data)
        centroids = cl.centroids

    elif tf.flags.FLAGS.method == 'kmeans':
        cl = KMeans(
            n_clusters=tf.flags.FLAGS.nclusters,
            **params)

        labels = cl.fit(data)
        centroids = cl.centroids

    elif tf.flags.FLAGS.method == 'mini_batch_kmeans':
        cl = MiniBatchKMeans(
            n_clusters=tf.flags.FLAGS.nclusters,
            batch_size=tf.flags.FLAGS.batchsize,
            **params)

        labels = cl.fit(data)
        centroids = cl.centroids

    else:
        raise ValueError('--method parameter must either '
                         'be < means_shift >'
                         '< mini_batch_mean_shift >,'
                         '< kmeans > or < mini_batch_kmeans >.')

    if labels is not None and centroids is not None:
        if cl.history is None:
            tf.logging.warn('Data is too large to visualize.')
        elif data.shape[1] != 2:
            tf.logging.warn('Data must be 2 dimensional to visualize.')
        else:
            tf.logging.info('Creating plot for history visualization.')
            plot(cl.history, data, labels, centroids)

        save(os.path.join(tf.flags.FLAGS.save, 'centroids.npy'), centroids)
        save(os.path.join(tf.flags.FLAGS.save, 'labels.npy'), labels)


if __name__ == '__main__':
    tf.app.run()
