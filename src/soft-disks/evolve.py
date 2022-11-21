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

def find_distances(position_array: ArrayLike) -> ArrayLike:
    pass

def force(position_array: ArrayLike) -> ArrayLike:
    pass
    
def equilibriate():
    pass



if __name__=="__main__":
    from build_ensemble import square_build
    from utils import plot

    plot(move(square_build()))