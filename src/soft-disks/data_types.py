from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from config import N

@dataclass
class Configuration:
    """
    Class that conains all data pertaining to a given 
    ensemble configuration. Including:
        - `positions`: Of all the particles
        - `velocities`: ''
        - `forces`: On all particles in the given 
        position configuration, since this value is 
        calculated and need again in the velocity 
        Verlet algorithm,
    """
    positions: ArrayLike
    velocities: ArrayLike
    forces: ArrayLike
    
    def __post_init__(self):
        assert np.array(self.positions).shape == (N ,2)
        assert np.array(self.velocities).shape == (N ,2)        
        assert np.array(self.forces).shape == (N ,2)

        