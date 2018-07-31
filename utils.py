"""

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load(path):
    """Loads a numpy array."""
    return np.load(file=path)


def save(path, data):
    """Saves a numpy array"""
    np.save(file=path, arr=data)


def plot(history, data, labels, centroids):
    """Plots the history of the Mean Shift clustering."""

    def _draw_line(p1, p2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color='orange', linewidth=0.6, zorder=1)

    draw_line = np.vectorize(lambda p1, p2: _draw_line(p1, p2),
                             signature='(n),(n)->()')

    grid = sns.jointplot(data[:, 0], data[:, 1], kind="kde")
    grid.plot_joint(plt.scatter, c=labels, zorder=2)

    pen = len(history[0]) // 100 if len(history) > 100 else 0
    step = len(history) // 50 + pen + 1
    for i in range(0, len(history) - step, step):
        draw_line(history[i], history[i + step])

    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='white', marker="*", s=100,
                linewidth=0.8, edgecolor='black', zorder=3)

    plt.show()


def generate_random():
    a = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], [50])
    b = np.random.multivariate_normal([-3, -3], [[1, 0], [0, 1]], [50])
    c = np.random.multivariate_normal([-8, -3], [[1, 0], [0, 1]], [50])
    data = np.vstack((a, b, c))

    return data
