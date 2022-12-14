import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

from config import sigma, N, L, init_spacing, velocity_variance
from utils import plot
from data_types import Configuration

def hex_build() -> Tuple[ArrayLike]:
    """ Note: Only works for perfect square number of particles """
    initial_positions,  = np.empty([N, 2])
    initial_velocities = np.random.normal(0, velocity_variance, size=(N,2))
    vertalarr = np.arange(0, 11, 1)
    initial_positions[0, 0] = init_spacing/2
    initial_positions[0, 1] = init_spacing/2

    for i in range(N): #loop over all particles
        if i % 20 <= 9:
            initial_positions[i, 0] = (i%10)*init_spacing + init_spacing/2
            initial_positions[i, 1] = np.sqrt(3)*((i-(i%10))/20)*init_spacing + init_spacing/2
        else:
            initial_positions[i, 0] = (i%10)*init_spacing + init_spacing
            initial_positions[i, 1] = np.sqrt(3)*init_spacing*((i-(i%10))/20) + init_spacing/2 
            
    return Configuration(positions=initial_positions, velocities=initial_velocities, forces = np.full((N, 2), np.nan))
     

def square_build(n = N, l = L) -> Configuration:
    """
    Return a square array of particles.
    
    Args:
        - `n` (optional): The number of particles to initialize. Defaults to the value specified in `config.py`
        - `l` (optional): The length of the box containing the configuration. Defaults to the value specified in `config.py`
    """
    _velocities = np.random.normal(0, np.sqrt(velocity_variance), size=(n,2))
    mean_velocity = np.mean(_velocities, axis=0)
    initial_velocities = _velocities - mean_velocity
    if int(np.sqrt(n))**2 == n:
        square_length = n
    else:
        square_length = (int(np.sqrt(n)) + 1)**2
    
    X,Y = np.mgrid[sigma/2:l-sigma/2:(l-0.2)/np.sqrt(square_length), sigma/2:l-sigma/2:(l-0.2)/np.sqrt(square_length)]
    initial_positions = np.array(list(zip(X.flatten(), Y.flatten())))
    
    conf_metadata = {'N':n, 'L':l}
    return Configuration(
        positions=initial_positions[:n,:], 
        velocities=initial_velocities, 
        forces = np.full((n, 2), np.nan),
        metadata=conf_metadata
    )

if __name__ == "__main__":
    print(hex_build())
    