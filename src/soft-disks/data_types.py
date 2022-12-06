from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from numpy.typing import ArrayLike

from config import N

@dataclass
class Configuration:
    """
    Class that contains all data pertaining to a given
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
    metadata: Dict[str, Any]

    def __post_init__(self):
        estring = f"Velocity and position properties' shapes do not match. Position shape: {self.positions.shape}, velocities shape: {self.velocities.shape}."
        assert np.array(self.positions).shape == np.array(self.velocities).shape, estring


