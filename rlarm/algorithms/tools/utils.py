import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from collections import deque, namedtuple
import itertools
import os
import time

import struct
import ctypes


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


####################################################################################################
### Plots for visualization ###
####################################################################################################

def plot_learning_curve(filename, value_dict, xlabel = 'step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize = (12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.join(REPO_ROOT, 'figs'), exist_ok = True)
    plt.savefig(os.path.join(REPO_ROOT, 'figs', filename))
