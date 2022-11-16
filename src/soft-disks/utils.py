import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import L, sigma, N


def plot(conf, title=None):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    for i in np.arange(0, N, 1):
        disk = plt.Circle((conf[i, 0] + 0.1, conf[i, 1] + 0.1), 0.5*sigma, 
                          color = 'green', linewidth = 0.1)
        
        ax.add_artist(disk)
        container = patches.Rectangle((0.1, 0.1), width = L, height = L, edgecolor = 'grey', fill = False, linestyle = ':')
        ax.add_patch(container)
    fig.show()