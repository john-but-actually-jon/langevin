import numpy as np
from numpy.typing import ArrayLike
from numba import jit
from typing import Tuple, List

from data_types import Configuration
from config import N, dt, L, epsilon, sigma, m, lj_cutoff
from utils import plot, prog_bar
from build_ensemble import hex_build, square_build

# @jit
def tile_and_remove_self_reference(position_array: ArrayLike, ndims: int=2) -> ArrayLike:
    """
    Tile the input array of positions such that each particle has an
    associated copy of the position array.
    Then remove the diagonal elements of resulting matrix to remove
    the position of the particle to which the new position array
    belongs.

    Args:
        - `position_array` (required): The array containing the
        positions of `ndim`
        - `ndim` (optional): Number of dimensions specified by the position elements. Default: 2
    """
    tiled = np.tile(position_array, (position_array.shape[0],1,1))
    return tiled[~np.eye(tiled.shape[0],tiled.shape[1], dtype=bool)].reshape([tiled.shape[0], -1, ndims])

def find_directions(position_array: ArrayLike)-> ArrayLike:
    """
    Return an Nx(N-1)x2 matrix that contains the inter-particle displacement vectors for every particle, excluding itself according to the least image convention.
    """
    # Determine relevant image
    dediagonalised_positions = tile_and_remove_self_reference(position_array)
    difference_array = position_array.repeat(
        dediagonalised_positions.shape[1],
        axis=0
    ).reshape(dediagonalised_positions.shape)
    image_displacements = dediagonalised_positions-difference_array

    # Calculate the displacement vectors and their norms
    displacement_norms = np.linalg.norm(
        image_displacements
        , axis=-1
    )

    # Filter for cutoff
    mask = (displacement_norms <= lj_cutoff)
    vector_mask = np.dstack([mask, mask])
    displacements = np.where(vector_mask, image_displacements, np.nan)
    displacement_norms = np.where(mask, displacement_norms, np.nan)

    # Divide vectors by their norms
    unit_vectors = np.full(displacements.shape, np.nan)
    if ~np.isnan(displacement_norms).all():
        np.divide(
            displacements,
            np.dstack([displacement_norms]*2),
            out=unit_vectors,
            where=vector_mask
        )
    # Return MASKED vectors, to allow for use later on
    return (
        np.ma.array(unit_vectors, mask=np.isnan(unit_vectors), fill_value=0),
        np.ma.array(displacement_norms, mask=np.isnan(displacement_norms), fill_value=0)
    )


def calculate_force(unit_displacements: ArrayLike, displacement_norms: ArrayLike) -> ArrayLike:
    r7 = displacement_norms ** 7
    r13 = displacement_norms ** 13
    absolute_force = 24*epsilon*(sigma/r7 - 2*sigma/r13)

    return np.sum(
        unit_displacements * np.dstack([absolute_force]*2),
        axis=1
    ).data

def force(position_array: ArrayLike) -> ArrayLike:
    forces = np.full((position_array.shape[0], 2), 0)

    try:
        unit_displacements, displacement_norms = find_directions(
            position_array
        )
        forces = calculate_force(
            unit_displacements,
            displacement_norms
        )
        assert forces.shape == position_array.shape

    except ValueError:
        return forces
    return forces

def rebox(position: ArrayLike) -> ArrayLike:
    x = position[0] - np.floor((position[0] - position[0] % L)/L) * L
    y = position[1] - np.floor((position[1] - position[1] % L)/L) * L
    return np.array([x,y])

v_rebox = np.vectorize(rebox, signature=f'(2)->(2)')

def update_position(configuration: Configuration, timestep: float = dt) -> Configuration:
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
    new_conf = Configuration(
        v_rebox(
        configuration.positions + timestep * configuration.velocities
        ),
        configuration.velocities,
        configuration.forces
    )

    return new_conf


def update_velocity(configuration: Configuration) -> Configuration:
    """
    Finds the discretized "m plus one half" velocity. Apply twice, once with the old coordinates and once with the new coordinates to get the full velocity update.
    """

    velocities = configuration.velocities - 0.5 * dt * (configuration.forces)

    return Configuration(
        configuration.positions,
        velocities,
        configuration.forces
    )

def move(configuration: Configuration) -> Configuration:
    v_m_plus_half_conf = update_velocity(configuration)
    q_m_plus_one_conf = update_position(v_m_plus_half_conf)
    q_m_plus_one_conf.forces = force(q_m_plus_one_conf.positions)

    return update_velocity(q_m_plus_one_conf)

@jit(forceobj=True)
def p_energy(absolute_distances: ArrayLike) -> float:
    r6 = absolute_distances ** 6
    r12 = r6**2
    return np.sum(4*epsilon*(sigma/r12 - sigma/r6)) + epsilon


def calculate_energies(configuration: Configuration) -> Tuple[float]:
    """
    Calculate the kinetic and potential energies of a configuration

    Args:
        - `configuration`: COnfiguration object with data in question

    Returns:
        Tuple containing `kinetic_energy` and `potential_energy`
    """
    position_array, velocity_array = configuration.positions, configuration.velocities

    potential_energy = 0.0
    kinetic_energy = np.sum(0.5*m*(np.linalg.norm(velocity_array, axis=1)**2))

    try:
        if not position_array.shape[0]==2:
            _unit_displacements, displacement_norms = find_directions(
                position_array,
            )
        else:
            print("THis shit wack")
            # displacement_norms = np.linalg.norm(position_array - particle_position)
            # potential_energy += p_energy(displacement_norms)
            # break
        potential_energy += p_energy(displacement_norms)
    except IndexError:
        raise IndexError # yep, this is gross


    return (kinetic_energy, potential_energy)


def equilibriate(initial_configuration: Configuration, nsteps: int = 10000, cache_interval=None) -> Tuple[Configuration, Tuple[ArrayLike], List[Configuration]]:
    kinetic_energies, potential_energies = [], []
    initial_energies = calculate_energies(initial_configuration)
    kinetic_energies.append(initial_energies[0])
    potential_energies.append(initial_energies[1])
    saved_configs = []


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
            if cache_interval and i % cache_interval:
                saved_configs.append(configuration)
        except KeyboardInterrupt:
            return (configuration, (np.array(kinetic_energies), np.array(potential_energies)), saved_configs)

    energies = (np.array(kinetic_energies), np.array(potential_energies))
    return (configuration, energies, saved_configs)


if __name__=="__main__":
    from build_ensemble import square_build
    from utils import plot
    import matplotlib.pyplot as plt

    x = square_build()

    newconf, energies = equilibriate(x, nsteps=100)
