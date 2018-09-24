"""

@author:    Patrik Purgai
@copyright: Copyright 2018, tfcluster
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2018.08.17.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from clustering import KMeans
from clustering import MiniBatchKMeans
from clustering import MeanShift

from utils import animated_plot
from utils import generate_random
from utils import load, save
from utils import plot
from utils import remove_empty


tf.flags.DEFINE_float('bandwidth', None, 'Bandwidth of the kernel.')
tf.flags.DEFINE_string('data', None, 'Path of the data for clustering.')
tf.flags.DEFINE_string('kernel', None, 'Type of the kernel.')
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

methods = {
    'mean_shift':           MeanShift,
    'kmeans':               KMeans,
    'mini_batch_kmeans':    MiniBatchKMeans
}


def main(_):
    tf.logging.set_verbosity(_verbosity_levels[tf.flags.FLAGS.verbosity])

    params = {
        'criterion':    tf.flags.FLAGS.criterion,
        'max_iter':     tf.flags.FLAGS.maxiter,
        'kernel':       tf.flags.FLAGS.kernel,
        'bandwidth':    tf.flags.FLAGS.bandwidth,
        'n_clusters':   tf.flags.FLAGS.nclusters,
        'batch_size':   tf.flags.FLAGS.batchsize
    }

    data = generate_random(100, 500)

    # assert os.path.exists(tf.flags.FLAGS.data)

    if tf.flags.FLAGS.method in methods:

        cl = methods[tf.flags.FLAGS.method](**remove_empty(params))

        labels = cl.fit(data)
        centroids = cl.centroids
        history = cl.history

    else:
        history = load(
            os.path.join(tf.flags.FLAGS.save, 'history.npy'))
        centroids = load(
            os.path.join(tf.flags.FLAGS.save, 'centroids.npy'))
        labels = load(
            os.path.join(tf.flags.FLAGS.save, 'labels.npy'))

        if tf.flags.FLAGS.method == 'visualize':
            assert len(history) > 1 \
                   and history[0].shape[0] == labels.shape[0], 'Invalid ' \
                                                               'history'
            plot(history, data, labels, centroids, draw_lines=False)

        elif tf.flags.FLAGS.method == 'visualize_animated':
            assert len(history) > 1 \
                   and history[0].shape[0] == labels.shape[0], 'Invalid ' \
                                                               'history'
            animated_plot(history, labels)

        else:
            raise ValueError('--mode parameter must either '
                             'be < means_shift >'
                             '< mini_batch_mean_shift >,'
                             '< kmeans > or < mini_batch_kmeans >.')

        return

    if history is None:
        tf.logging.warn('Data is too large to visualize.')
    elif data.shape[1] != 2:
        tf.logging.warn('Data must be 2 dimensional to visualize.')
    else:
        tf.logging.info('Creating plot for history visualization.')

        plot(history, data, labels, centroids, draw_lines=False)

        save(os.path.join(tf.flags.FLAGS.save, 'history.npy'), history)
    save(os.path.join(tf.flags.FLAGS.save, 'centroids.npy'), centroids)
    save(os.path.join(tf.flags.FLAGS.save, 'labels.npy'), labels)


if __name__ == '__main__':
    tf.app.run()
