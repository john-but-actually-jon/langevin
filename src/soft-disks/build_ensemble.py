import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

from config import sigma, N, L, init_spacing, velocity_variance
from utils import plot

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
            
    return (initial_positions, initial_velocities)

def square_build() -> ArrayLike:
    initial_positions = np.empty([N, 2])
    initial_velocities = np.random.normal(0, velocity_variance, size=(N,2))
    vertalarr = np.arange(0, 11, 1)
    initial_positions[0, 0] = init_spacing/2
    initial_positions[0, 1] = init_spacing/2
    
    for i in np.arange(1, N, 1):    
        initial_positions[i, 0] = (i%10)*init_spacing + init_spacing/2
        for j in range(10):
            if vertalarr[j] <= i/10:
                if vertalarr[j+1] > i/10:
                    initial_positions[i, 1] = vertalarr[j]*init_spacing + init_spacing/2
                else:
                    pass
    return (initial_positions, initial_velocities)

if __name__ == "__main__":
    print(hex_build()[0])
    