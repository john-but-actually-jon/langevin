import numpy as np
from numpy.typing import ArrayLike

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import L, sigma, N, scaler


def plot(conf, title=None):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    for position in conf:
        disk = plt.Circle(
            (position[0] + 0.1, position[1] + 0.1),
            0.5*sigma, 
            color = 'green',
            linewidth = 0.1
        )
        
        ax.add_artist(disk)
        container = patches.Rectangle((0.1, 0.1), width = L, height = L, edgecolor = 'grey', fill = False, linestyle = ':')
        ax.add_patch(container)
    fig.show()
    
def plot_with_velocities(conf: Tuple[ArrayLike], scaled_velocities: bool = False, title=None):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    for i, position in enumerate(conf[0]):
        disk = plt.Circle(
            (position[0] + 0.1, position[1] + 0.1),
            0.5*sigma, 
            color = 'green',
            linewidth = 0.1
        )
        ax.add_artist(disk)
        
        plt.arrow(position[0]+ 0.1, position[1]+ 0.1, conf[1][i][0], conf[1][i][1])
        
        container = patches.Rectangle((0.1, 0.1), width = L, height = L, edgecolor = 'grey', fill = False, linestyle = ':')
        ax.add_patch(container)
    fig.show()