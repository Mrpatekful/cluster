"""

"""
from tensorflow.python.client import device_lib

import numpy as np


def load(path):
    """Loads a numpy array."""
    return np.load(file=path)


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
