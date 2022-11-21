import numpy as np
from numpy.typing import ArrayLike

from typing import Tuple

from config import N, dt, L
from utils import plot
from build_ensemble import hex_build, square_build

def move(starting_configuration: Tuple[ArrayLike], timestep: float = dt):
    """
    Updates the positions in 
    `starting_configuration` with the particles' 
    associated velocities.
    
    Parameters:
        - `starting_configuration` (Tuple
        [ArrayLike], required): The Tuple of 
        position and velocity arrays that define 
        the current configuration.
        - `timestep` (float, optional): The length 
        of time to evolve over, uses the `dt` 
        parameter in the config.py file by default
    Returns:
        The updated position array, the forces and 
        thus the velocities are calculated from 
        the new position later.
    """
    positions = np.empty((N,2))
    _positions = starting_configuration[0] + timestep * starting_configuration[1]
    for i, position in enumerate(_positions):
        _x, _y = position
        while _x < 0 or _x > L:
            _x = _x - np.sign(_x)*L
        while _y < 0 or _y > L:
            _y = _y - np.sign(_y)*L
        positions[i] = [_x, _y]
    return positions

def build_ewald_image(position_array: ArrayLike) -> ArrayLike:
    """"""
    ewald_images = {
        "TL" : position_array + np.array([-L, L]),
        "CL" : position_array + np.array([-L, 0]),
        "BL" : position_array + np.array([-L, -L]),
        "BC" : position_array + np.array([0, -L]),
        "BR" : position_array + np.array([L, -L]),
        "CR" : position_array + np.array([L, 0]),
        "TR" : position_array + np.array([L, L]),
        "TC" : position_array + np.array([0, L]),
    }
    return np.concatenate([position_array, *ewald_images.values()])

def find_distances(position_array: ArrayLike) -> ArrayLike:
    """"""
    ewald_image = build_ewald_image(position_array)
    r = np.full((N, 9*N), np.nan)
    for i in range(N):
        assert ewald_image[i, 0] > 0 and ewald_image[i, 0] < L # Check that atom is indeed in centre square
        assert ewald_image[i, 1] > 0 and ewald_image[i, 1] < L
        position = ewald_image[i]
        distances = np.linalg.norm(ewald_image-position, axis=1)
        r[i,:] =  np.where(distances<L/2, distances, np.nan)
    return r

def force(position_array: ArrayLike) -> ArrayLike:
    radial_distances = find_distances(position_array)
    
    
def equilibriate():
    pass



if __name__=="__main__":
    from build_ensemble import square_build
    from utils import plot

    plot(move(square_build()))