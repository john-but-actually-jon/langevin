import numpy as np
from numpy.typing import ArrayLike

from typing import Tuple, List

from data_types import Configuration
from config import N, dt, L, epsilon, sigma, m
from utils import plot, prog_bar
from build_ensemble import hex_build, square_build


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
    images = [ewald_images[key] for key in quadrant_keys[quadrant-1]]
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
    absolute_force = 24*epsilon*(sigma/r7 - 2*sigma/r13) 
    
    return np.sum(
        unit_displacements * np.concatenate(
            [absolute_force, absolute_force]
            ).reshape(
                [len(absolute_force),-1],
                order='F'
            ),
            axis=0
    )

def force(position_array: ArrayLike) -> ArrayLike:
    forces = np.full((N, 2), np.nan)
    
    # Loop over particles and find the force for each of them
    for i, particle_position in enumerate(position_array):
        unit_displacements, displacement_norms = find_directions(
            particle_position, 
            np.delete(position_array, i, axis=0)
        )
        _f = calculate_force(unit_displacements, displacement_norms)
        forces[i, :] = _f
    return forces


def update_position(configuration: Configuration, timestep: float = dt):
    """
    Updates the positions in 
    `starting_configuration` with the particles' 
    associated velocities.
    
    Parameters:
        - `starting_configuration` (Tuple
        [ArrayLike], required): The Tuple of 
        position and  arrays that define 
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
    _positions = configuration.positions + timestep * configuration.velocities
    for i, position in enumerate(_positions):
        _x, _y = position
        while _x < 0 or _x > L:
            _x = _x - np.sign(_x)*L
        while _y < 0 or _y > L:
            _y = _y - np.sign(_y)*L
        positions[i] = [_x, _y]
    return positions
 
def update_velocity(configuration: Configuration, updated_position_array: ArrayLike) -> ArrayLike:
    """ 
    Updates the velocities of a given configuration based on the new positions, giving the force, and the old velocities
    
    Args:
        - `configuration`: Configuration with the old positions ($q_{m-1}$)
        - `updated_position_array`: The updated positions.
        
    Returns:
        A tuple containing the updated velocities and forces
    """
    forces = force(updated_position_array)
    velocities = configuration.velocities + 0.5 * dt * (configuration.forces + forces)
    
    return (velocities, forces)

def move(configuration: Tuple[ArrayLike]) -> Tuple[ArrayLike]:
    positions = update_position(configuration)
    velocities, forces = update_velocity(configuration, positions)
    
    return Configuration(positions, velocities, forces)


def calculate_energies(configuration: Tuple[ArrayLike]) -> Tuple[float]:
    """
    Calculate the kinetic and potential energies of a configuration
    
    Args:
        - `position_array`: The positions of the particles in a given configuration
        - `_array`: The velocities of the particles in a given array
        
    Returns:
        Tuple containing `kinetic_energy` and `potential_energy`
    """
    position_array, _array = configuration.positions, configuration.velocities
    
    potential_energy = 0.0
    kinetic_energy = np.sum(0.5*m*(np.linalg.norm(_array, axis=1)**2))
    
    for i, particle_position in enumerate(position_array):
        _unit_displacements, displacement_norms = find_directions(
            particle_position, 
            np.delete(position_array, i, axis=0)
        )
        r12 = (r6 := displacement_norms ** 6)**2
        potential_energy += -np.sum(4*epsilon*(sigma/r12 - sigma/r6))
    
    return (kinetic_energy, potential_energy)


def equilibriate(initial_configuration: Configuration, nsteps: int = 10000) -> Tuple[Configuration, ArrayLike]:
    kinetic_energies, potential_energies = [], []
    configuration = Configuration(
        initial_configuration.positions,
        initial_configuration.velocities,
        force(initial_configuration.positions)  
    )

    for i in range(nsteps):
        try:
            configuration = move(configuration)
            if not i % 10:
                _energies = calculate_energies(configuration)
                kinetic_energies.append(_energies[0])
                potential_energies.append(_energies[1])
            prog_bar(i, nsteps)
        except KeyboardInterrupt:
            return (configuration, (np.array(kinetic_energies), np.array(potential_energies)))
        
    energies = (np.array(kinetic_energies), np.array(potential_energies))
    return (configuration, energies)


if __name__=="__main__":
    from build_ensemble import square_build
    from utils import plot
    import matplotlib.pyplot as plt

    x = square_build()

    newconf, energies = equilibriate(x, nsteps=100)