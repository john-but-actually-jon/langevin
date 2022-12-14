import numpy as np
from numpy.typing import ArrayLike
from numba import jit
from typing import Tuple, List, Dict, Any
from pathlib import Path

from data_types import Configuration
from config import N, dt, L, epsilon, sigma, m, lj_cutoff
from utils import plot, prog_bar, save_configs
from build_ensemble import hex_build, square_build




def radial_pbc(r, l=L):
    """ Find the PBC displacement vector between two particles. Apply after the subtraction method"""
    rx, ry = r
    if np.abs(rx) > l/2:
        rx = rx - np.sign(rx) * l
    if np.abs(ry) > l/2:
        ry = ry - np.sign(ry) * l
    return np.array([rx, ry])

v_radial_pbc = np.vectorize(radial_pbc, signature='(2),()->(2)')

def force(position_array: ArrayLike, metadata: Dict[str, Any]) -> ArrayLike:

    # Loop over particles and find the force for each of them
    forces = np.empty((position_array.shape[0], 2))
    positions = np.stack([position_array]*position_array.shape[0])
    difference_array = np.array([[position]*position_array.shape[0] for position in position_array])
    displacement_vectors = v_radial_pbc(positions-difference_array, metadata['L'])

    r2 = np.sum(np.square(displacement_vectors), axis=-1)

    for i, particle_image in enumerate(r2):
        valid_vectors = []
        # Get all valid displacement vectors
        for j, displacement in enumerate(particle_image):
            if displacement <= lj_cutoff**2 and displacement != 0:
                valid_vectors.append(displacement_vectors[i,j,:])

        valid_vectors = np.array(valid_vectors)
        # Sum the force contributions from valid displacement vectors
        norms = np.linalg.norm(valid_vectors, axis=-1)
        inverse_norms = np.divide(1, norms, out=np.zeros_like(norms), where=(norms>0))
        abs_forces = 48 * (np.power(inverse_norms, 13) - 0.5*np.power(inverse_norms, 7))
        try:
            assert norms.size==valid_vectors.shape[0]
        except AssertionError:
            forces[i, :] = np.array([0,0])
            continue
        forces[i, :] = np.sum(-valid_vectors / np.dstack([norms]*2) * np.dstack([abs_forces]*2), axis=1)[0,:]
    return forces


def rebox(position: ArrayLike, l=L) -> ArrayLike:
    x = position[0] - np.floor((position[0] - position[0] % l)/l) * l
    y = position[1] - np.floor((position[1] - position[1] % l)/l) * l
    return np.array([x,y])

v_rebox = np.vectorize(rebox, signature=f'(2),()->(2)')

@jit(forceobj=True)
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
    positions = v_rebox(
    configuration.positions + timestep * configuration.velocities,
    configuration.metadata['L']
    )
    forces = force(positions, configuration.metadata)
    new_conf = Configuration(
        positions,
        configuration.velocities,
        forces,
        configuration.metadata
    )

    return new_conf

@jit(forceobj=True)
def update_velocity(configuration: Configuration) -> Configuration:
    """
    Finds the discretized "m plus one half" velocity. Apply twice, once with the old coordinates and forces and once with the new coordinates and forces to get the full velocity update.
    """

    velocities = configuration.velocities + 0.5 * dt * configuration.forces

    return Configuration(
        configuration.positions,
        velocities,
        configuration.forces,
        configuration.metadata
    )

def move(configuration: Configuration) -> Configuration:
    v_m_plus_half_conf = update_velocity(configuration)
    q_m_plus_one_conf = update_position(v_m_plus_half_conf)

    return update_velocity(q_m_plus_one_conf)

@jit(forceobj=True)
def p_energy(position_array: ArrayLike, metadata: Dict) -> float:
    mask = np.dstack([np.tril([True]*position_array.shape[0])]*2)
    positions = np.stack([position_array]*position_array.shape[0])
    difference_array = np.array([[position]*position_array.shape[0] for position in position_array])

    # r2 = np.sum(np.square(np.where(~mask, v_rebox(positions-difference_array), 0)), axis=-1)
    r2 = np.sum(np.square(v_rebox(positions-difference_array, metadata['L'])), axis=-1)
    r2 = r2[(r2 <= lj_cutoff**2) & (r2 !=0)]
    inverse = np.divide(1,r2, out=np.zeros_like(r2), where=(r2>0))
    potential_energy = 4*(
        np.power(inverse, 6) - np.power(inverse, 3)
    ) + 1
    return np.sum(potential_energy)


def calculate_energies(configuration: Configuration) -> Tuple[float]:
    """
    Calculate the kinetic and potential energies of a configuration
    Args:
        - `configuration`: COnfiguration object with data in question
    Returns:
        Tuple containing `kinetic_energy` and `potential_energy`
    """
    position_array, velocity_array = configuration.positions, configuration.velocities

    kinetic_energy = np.sum(0.5*m*(np.power(velocity_array, 2)))
    potential_energy = p_energy(configuration.positions)

    return (kinetic_energy, potential_energy)



def equilibriate(initial_configuration: Configuration, nsteps: int = 5000, cache_interval=None, folder_name=None) -> Tuple[Configuration, Tuple[ArrayLike], List[Configuration]]:
    """
    Updates a position a specified number of times. Option to save the resulting configurations at certain intervals under a specific `folder_name`
    
    Args:
        - `initial_configuration` (required): The starting configuration to evolve.
        - `nsteps` (required): The desired number of steps to evolve by.
        - `cache_interval` (optional): The interval between configs that are saved. Defaults to None which saves none.
        - `folder_name` (optional): The name of the folder under `data` into which the configs are saved. Must be specified if `cache_interval is specified.
        
    Returns:
        Tuple containing the final configuration, the energy tuple and the average velocity array.
    """
    kinetic_energies, potential_energies = [], []
    average_velocities = []
    saved_configs = []

    starting_configuration = Configuration(
        initial_configuration.positions,
        initial_configuration.velocities,
        force(initial_configuration.positions),
        initial_configuration.metadata
    )
    configuration = starting_configuration
    for i in range(nsteps):
        try:
            configuration = move(configuration)
            average_velocities.append(np.mean(np.abs(configuration.velocities), axis=0))
            if not i % 1:
                _energies = calculate_energies(configuration)
                kinetic_energies.append(_energies[0])
                potential_energies.append(_energies[1])
            prog_bar(i, nsteps)
            if cache_interval and not i % cache_interval:
                assert folder_name, 'No folder name provided for saving configs!'
                saved_configs.append(configuration)
        except KeyboardInterrupt:
            return (configuration, (np.array(kinetic_energies), np.array(potential_energies)),  np.array(average_velocities))
    prog_bar(nsteps,nsteps)
    energies = (np.array(kinetic_energies), np.array(potential_energies))
    if folder_name:
        save_configs(saved_configs, folder_name)
    return (configuration, energies, np.array(average_velocities))


if __name__=="__main__":
    from build_ensemble import square_build
    from utils import plot

    x = square_build()

    newconf, energies = equilibriate(x, nsteps=100)
