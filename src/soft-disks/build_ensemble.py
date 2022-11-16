import numpy as np
from numpy.typing import ArrayLike

from config import sigma, N, L
from utils import plot

def hex_build() -> ArrayLike:
    initial_positions, initial_velocities = np.empty([N, 2]), np.empty([N, 2])
    vertalarr = np.arange(0, 11, 1)
    initial_positions[0, 0] = sigma/2
    initial_positions[0, 1] = sigma/2

    for i in range(N): #loop over all particles
        if i % 20 <= 9:
            initial_positions[i, 0] = (i%10)*sigma + sigma/2
            initial_positions[i, 1] = np.sqrt(3)*sigma*((i-(i%10))/20) + sigma/2
        else:
            initial_positions[i, 0] = (i%10)*sigma + sigma
            initial_positions[i, 1] = np.sqrt(3)*sigma*((i-(i%10))/20) + sigma/2 
            
    return (initial_positions, initial_velocities)

def square_build() -> ArrayLike:
    initial_positions, initial_velocities = np.empty([N, 2]), np.empty([N, 2])
    vertalarr = np.arange(0, 11, 1)
    initial_positions[0, 0] = sigma/2
    initial_positions[0, 1] = sigma/2
    for i in np.arange(1, N, 1):
        
        initial_positions[i, 0] = (i%10)*sigma + sigma/2
        for j in range(10):
            if vertalarr[j] <= i/10:
                if vertalarr[j+1] > i/10:
                    initial_positions[i, 1] = vertalarr[j]*sigma + sigma/2
                else:
                    pass
    return (initial_positions, initial_velocities)

if __name__ == "__main__":

    plot(ensemble:=hex_build())
    print(ensemble)