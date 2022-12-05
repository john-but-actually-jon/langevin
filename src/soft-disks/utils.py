import numpy as np
from numpy.typing import ArrayLike

from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cmx
from matplotlib import colormaps as cmaps

from config import L, sigma, N, scaler, m
from data_types import Configuration

def plot(conf: Configuration, title=None):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])

    for position in conf.positions:
        disk = plt.Circle(
            (position[0], position[1]),
            0.5*sigma,
            color = 'green',
            linewidth = 0.1
        )
        ax.add_artist(disk)
    # Draw outlines on top
    for position in conf.positions:
        outline = plt.Circle(
            (position[0], position[1]),
            0.5*sigma,
            color = 'k',
            linewidth = 1,
            fill=False
        )
        ax.add_artist(outline)



        container = patches.Rectangle((0.1, 0.1), width = L, height = L, edgecolor = 'grey', fill = False, linestyle = ':')
        ax.add_patch(container)
    fig.show()

def plot_with_v_color(conf, particle_labels=False):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])

    cmap = cmaps['viridis']
    velocity_values = np.linalg.norm(conf.velocities, axis = -1)
    norm = Normalize(0,velocity_values.max())
    for position, velocity in zip(conf.positions, velocity_values):
        disk = plt.Circle(
            (position[0], position[1]),
            0.5*sigma,
            color = cmap(norm(velocity)),
            linewidth = 0.1
        )
        ax.add_artist(disk)
    # Draw outlines on top
    for i,position in enumerate(conf.positions):
        outline = plt.Circle(
            (position[0], position[1]),
            0.5*sigma,
            color = 'k',
            linewidth = 1,
            fill=False
        )
        ax.add_artist(outline)
        if particle_labels:
            plt.text(*position, str(i))

    cbar = fig.colorbar(cmx.ScalarMappable(norm, cmap), aspect=10, orientation='vertical', ax=ax, label='Velocity')
    cbar.set_ticks([])

    container = patches.Rectangle((0.1, 0.1), width = L, height = L, edgecolor = 'grey', fill = False, linestyle = ':')
    ax.add_patch(container)
    fig.show()


def plot_trajectory(particle_index:int, confs: List):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    trajectory = [conf.positions[particle_index,:] for conf in confs]
    for i, position in enumerate(trajectory):
        if not i:
            disk = plt.Circle(
                (position[0], position[1]),
                0.5*sigma,
                color = 'green',
                linewidth = 0.1
            )
            ax.add_artist(disk)
        else:
            disk = plt.Circle(
                (position[0], position[1]),
                0.5*sigma,
                linewidth = 1,
                fill=False,
                linestyle='--',
                color='black'
            )
            ax.add_artist(disk)
        try:
            plt.arrow(*position, *(trajectory[i+1]-position))
        except IndexError:
            break
    fig.show()

def plot_with_velocities(conf: Configuration, scaled_velocities: bool = False, title=None):
    fig, ax = plt.subplots()
    ax.set_ylim(0, L + 0.2)
    ax.set_xlim(0, L + 0.2)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    for i, position in enumerate(conf.positions):
        disk = plt.Circle(
            (position[0] + 0.1, position[1] + 0.1),
            0.5*sigma,
            color = 'green',
            linewidth = 0.1
        )
        ax.add_artist(disk)

        plt.arrow(position[0]+ 0.1, position[1]+ 0.1, conf.velocities[i][0], conf.velocities[i][1])

        container = patches.Rectangle((0.1, 0.1), width = L, height = L, edgecolor = 'grey', fill = False, linestyle = ':')
        ax.add_patch(container)
    fig.show()

def prog_bar(iteration, max_iterations) -> None:
    percent_done = iteration/max_iterations
    d = "#" * round(percent_done * 84)
    t = "-" * (84-round(percent_done * 84))
    s = f"|{d+t}|{round(percent_done*100, 2)}%"
    if percent_done==1:
        s+='\n'
    if iteration>0:
        s = '\r'+s
    print(s, end="", flush=True)