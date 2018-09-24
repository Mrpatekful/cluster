"""

@author:    Patrik Purgai
@copyright: Copyright 2018, tfcluster
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2018.08.17.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation

from tempfile import NamedTemporaryFile

import seaborn as sns


def load(path):
    """Loads a numpy array."""
    return np.load(file=path)


def save(path, data):
    """Saves a numpy array"""
    np.save(file=path, arr=data)


def remove_empty(dictionary):
    """Removes empty entries from a dictionary."""
    for key in list(dictionary.keys()):
        if dictionary.get(key) is None:
            del dictionary[key]
    return dictionary


def plot(history, data, labels, centroids, draw_lines=True):
    """Plots the result of clustering."""

    def _draw_line(p1, p2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color='orange', linewidth=1, zorder=1)

    draw_line = np.vectorize(lambda p1, p2: _draw_line(p1, p2),
                             signature='(n),(n)->()')

    grid = sns.jointplot(
        data[:, 0], data[:, 1], kind="kde", height=6)
    grid.plot_joint(plt.scatter, c=labels, zorder=2)

    # Speeds up drawing by skipping several history objects

    if draw_lines:
        pen = len(history[0]) // 100 if len(history) > 100 else 0
        step = len(history) // 50 + pen + 1

        for i in range(0, len(history) - step, step):
            draw_line(history[i], history[i + step])

    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='white', marker="*", s=100,
                linewidth=0.8, edgecolor='black', zorder=3)

    plt.show()


# noinspection PyProtectedMember
def animation_to_html(animation):
    """Embeds animation in html."""
    video_tag = """<video controls>
     <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""

    if not hasattr(animation, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            animation.save(f.name, fps=20, extra_args=['-vcodec', 'libx264',
                                                       '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        animation._encoded_video = video.encode("base64")

    return video_tag.format(animation._encoded_video)


def animated_plot(history, labels, show=True):
    """Plots the animated history of the clustering."""

    grid = sns.JointGrid(x=history[0, :, 0], y=history[0, :, 1], height=6)
    grid.plot_joint(sns.kdeplot, shade=True)
    grid.plot_marginals(sns.kdeplot, shade=True)
    grid.plot_joint(plt.scatter, c=labels, zorder=2)

    def prep_axes(g):
        """Prepares the axes for the next iteration."""
        g.ax_joint.clear()
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()
        plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(g.ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(g.ax_marg_y.get_xticklabels(), visible=False)

    def animate(i):
        """Redraws the plot."""
        grid.x, grid.y = history[0, :, 0], history[0, :, 1]
        prep_axes(grid)
        grid.plot_joint(sns.kdeplot, shade=True)
        grid.plot_marginals(sns.kdeplot, shade=True)
        grid.x, grid.y = history[int(i), :, 0], history[int(i), :, 1]
        grid.plot_joint(plt.scatter, c=labels, zorder=2)

    animation = matplotlib.animation.FuncAnimation(grid.fig, animate,
                                                   frames=len(history),
                                                   repeat=True)

    if show:
        plt.show()
    else:
        return animation


def generate_random(n_samples=50, n_features=2):
    """Generates random samples."""
    a = np.random.multivariate_normal(
        [1] * n_features,
        np.random.random([n_features, n_features]) * 5 + 1,
        [n_samples])
    b = np.random.multivariate_normal(
        [-6] * n_features,
        np.random.random([n_features, n_features]) * 5 + 1,
        [n_samples])
    c = np.random.multivariate_normal(
        [11] * n_features,
        np.random.random([n_features, n_features]) * 5 + 1,
        [n_samples])

    data = np.vstack((a, b, c))

    return data
