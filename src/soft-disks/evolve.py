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

def find_particle_quadrant(position: ArrayLike) -> int:
    if position[0] < L/2:
        if position[1] < L/2:
            return 3
        else:
            return 1
    elif position[1] > L/2: return 2
    else: return 4

def build_ewald_image(quadrant: int, position_array: ArrayLike) -> ArrayLike:
    """
    Build the appropriate set of Ewald images according to the least images convention
    """
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
    quadrant_keys = [
        ["TL", "TC", "CL"],
        ["TC", "TR", "CR"],
        ["CL", "BL", "BC"],
        ["CR", "BC", "BR"]
    ]
    images = [ewald_images[key] for key in quadrant_keys[quadrant]]
    images.insert(0, position_array)
    return np.concatenate(images)


def find_directions(position_0: ArrayLike, position_array: ArrayLike)-> ArrayLike:
    """
    Find the unit direction vectors from a particle with `position_0` and all other
    particles determined by the least image convention.
    
    NOTE: Remember to make this negative when calculating the force!
    
    Args:
        - `position_0` (required): The position of the particle in question
        - `position_array` (required): The positions of all particles in the 
        centre image
    """
    
    # Determine relevant images
    image_positions = build_ewald_image(
        find_particle_quadrant(position_0),
        position_array
    )

    # Calculate the displacement vectors and their norms
    displacement_norms = np.linalg.norm(
        _displacements := position_array-position_0
        , axis=1
    )
    # Filter for cutoff
    _displacements = _displacements[displacement_norms < L/2, :]
    displacement_norms = displacement_norms[displacement_norms < L/2]
    
    unit_displacements = (
        _displacements / np.concatenate(
            [displacement_norms, displacement_norms]
            ).reshape(
                [len(displacement_norms),-1],
                order='F'
            )
    )
    
    return (unit_displacements, displacement_norms)

def calculate_force(unit_displacements: ArrayLike, displacement_norms: ArrayLike):
    r7 = displacement_norms ** 7
    r13 = displacement_norms ** 13
    

def force(position_array: ArrayLike) -> ArrayLike:
    forces = np.full(N, 2, np.nan)
    
    # Loop over particles and find the force for each of them
    for i, particle_position in enumerate(position_array):
        unit_displacements, displacement_norms = find_directions(
            particle_position, 
            position_array.delete(position_array, 0, axis=0)
        )
        forces[i,:] = calculate_force(unit_displacements, displacement_norms)
    
    
def equilibriate():
    pass



if __name__=="__main__":
    from build_ensemble import square_build
    from utils import plot

    x = square_build()[0]
    # print(x + [L, -L])
    # a = x[0, :]
    # print(a)
    # x = np.delete(x,0, axis=0)
    # print(find_directions(a,x))