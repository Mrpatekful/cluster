# K-Means & Mean Shift clustering

This repository contains clustering algorithms written in the TensorFlow framework.
Compared to Sklearn implementations, clustering high dimensional data is more efficient, however
large data sets may cause memory error in case of low VRAM.

### Requirements

- TensorFlow
- NumPy
- Seaborn (Optional)

### Usage

Running the algorithms on a data set is possible via command line interface:

´python3 main.py --data your_data.npy --method mean_shift --bandwidth 0.1´

The clustering can be invoked from code as well, which is demonstrated in [example.ipynb](https://github.com/Mrpatekful/tf-clustering/blob/master/example.ipynb).

### Features

The current version of the code utilizes GPU acceleration for moderately large data sets.
In case of Mean Shift clustering fragmentation of the provided data set is applied in effort to
reduce the concurrently used GPU memory. Note that this method is still under testing.
K-Means and mini batch version of K-Means are not yet capable of handling data sets that require
more RAM than available. To calculate the memory usage of your data set during clustering, plug your parameters into the following formula:

    n_data_points * data_dimension * n_clusters * 4bytes.