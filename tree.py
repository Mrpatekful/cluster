"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class BinaryTree:

    def __init__(self, data, leaf_size=40):
        self._data = data
        self.leaf_size = leaf_size

        n_samples = self._data.shape[0]
        n_features = self._data.shape[1]

        self.n_levels = np.log2(max(1, (n_samples - 1) / self.leaf_size)) + 1
        self.n_nodes = (2 ** self.n_levels) - 1

        self._construct_tree(0, 0, n_samples)

    def __getstate__(self):
        """
        get state for pickling
        """
        return None

    def __setstate__(self):
        """
        get state for pickling
        """
        return None

    def _construct_tree(self, index, start_index, end_index):
        pass

    def _init_node(self, index, start_index, end_index):
        pass


class BallTree(BinaryTree):

    def __init__(self, x):
        super(BallTree, self).__init__(x)

    def query_radius(self, x, r):
        pass


