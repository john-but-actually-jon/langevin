import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

from config import sigma, N, L, init_spacing, velocity_variance
from utils import plot
from data_types import Configuration

def hex_build() -> Tuple[ArrayLike]:
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
     

def square_build() -> Tuple[ArrayLike]:
    _velocities = np.random.normal(0, np.sqrt(velocity_variance), size=(N,2))
    mean_velocity = np.mean(_velocities, axis=0)
    initial_velocities = _velocities - mean_velocity

    X,Y = np.mgrid[sigma/2:L-sigma/2:(L-0.2)/np.sqrt(N), sigma/2:L-sigma/2:(L-0.2)/np.sqrt(N)]
    initial_positions = np.array(list(zip(X.flatten(), Y.flatten())))
    
    return Configuration(positions=initial_positions, velocities=initial_velocities, forces = np.full((N, 2), np.nan))

if __name__ == "__main__":
    print(hex_build())
    