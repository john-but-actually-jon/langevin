import numpy as np
from numpy.typing import ArrayLike
from numba import jit
from typing import Tuple, List

from data_types import Configuration
from config import N, dt, L, epsilon, sigma, m, lj_cutoff
from utils import plot, prog_bar
from build_ensemble import hex_build, square_build


def find_particle_quadrant(positions) -> int:
    if positions[0] < L/2:
        if positions[1] < L/2:
            return 3
        else:
            return 1
    elif positions[1] > L/2: return 2
    else: return 4

quadrantizer = np.vectorize(find_particle_quadrant, signature='(2)->()')

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
    images = [ewald_images[key] for key in quadrant_keys[int(quadrant)-1]]
    images.insert(0, position_array)
    return np.concatenate(images,axis=0).reshape([-1,2])

image_builder = np.vectorize(
    build_ewald_image, 
    signature="(),(n,2)->(m,2)"
)

def calculate_unit_vectors(displacement_norms: ArrayLike, displacement_vectors) -> ArrayLike:
    # Ensure that displacement norms can divide displacement_vectors, (shape=(n,n-1,2))
    unit_displacements = (
            np.divide(displacement_vectors, np.concatenate(
                [displacement_norms, displacement_norms]
                ).reshape(
                    [len(displacement_norms),-1],
                    order='F'
                )
            )
        )
    return unit_displacements

def find_directions(quadrants: int, position_array: ArrayLike)-> ArrayLike:
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
    image_positions = image_builder(
        quadrants,
        position_array
    )
    # Calculate displacements of particles
    displacement_vectors = image_positions - np.tile(
        position_array, 
        (1, image_positions.shape[1])
    ).reshape(image_positions.shape)
    
    # Calculate the displacement vectors and their norms
    displacement_norms = np.linalg.norm(
        displacement_vectors
        , axis=-1
    )
    
    # Filter for cutoff
    displacement_vectors = np.array(
        [
            arr[np.where(displacement_norms < lj_cutoff, True, False)[i,:]] for i, arr in enumerate(displacement_vectors)
        ]
    )
    displacement_norms = np.array(
        [
            arr[~np.isnan(arr)] for arr in np.where(displacement_norms<=3, displacement_norms, np.nan)
        ]
    )
    if displacement_norms.size:
        unit_displacements = (
            displacement_vectors / np.concatenate(
                [displacement_norms, displacement_norms]
                ).reshape(
                    [len(displacement_norms),-1],
                    order='F'
                )
        )
    else:
        unit_displacements=np.array([])
    return (unit_displacements, displacement_norms)


def calculate_force(unit_displacements: ArrayLike, displacement_norms: ArrayLike) -> ArrayLike:
    r7 = displacement_norms ** 7
    r13 = displacement_norms ** 13
    absolute_force = 24*epsilon*(sigma/r7 - 2*sigma/r13)

    return np.sum(
        unit_displacements * np.concatenate(
            (absolute_force, absolute_force)
            ).reshape(
                [len(absolute_force),-1],
                order='F'
            ),
            axis=0
    )

def force(position_array: ArrayLike) -> ArrayLike:
    forces = np.full((position_array.shape[0], 2), 0)
    quadrants = quadrantizer(position_array)
    # Loop over particles and find the force for each of them
    for i, particle_position in enumerate(position_array):
        try:
            unit_displacements, displacement_norms = find_directions(
                quadrants,
                position_array
                # particle_position,
                # quadrants,
                # np.delete(position_array, i, axis=0)
            )
            forces[i, :] = calculate_force(
                unit_displacements,
                displacement_norms
            )
        except ValueError:
            return forces
    return forces

def rebox(position: ArrayLike) -> ArrayLike:
    x = position[0] - np.floor((position[0] - position[0] % L)/L) * L
    y = position[1] - np.floor((position[1] - position[1] % L)/L) * L
    return np.array([x,y])

v_rebox = np.vectorize(rebox, signature=f'(2)->(2)')

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
    # positions = np.empty((configuration.positions.shape[0],2))
    return v_rebox(
        configuration.positions + timestep * configuration.velocities
    )


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

@jit(forceobj=True)
def p_energy(absolute_distances: ArrayLike) -> float:
    r6 = absolute_distances ** 6
    r12 = r6**2
    return np.sum(4*epsilon*(sigma/r12 - sigma/r6)) + r6.size*epsilon


def calculate_energies(configuration: Tuple[ArrayLike]) -> Tuple[float]:
    """
    Calculate the kinetic and potential energies of a configuration

    Args:
        - `configuration`: COnfiguration object with data in question

    Returns:
        Tuple containing `kinetic_energy` and `potential_energy`
    """
    position_array, velocity_array = configuration.positions, configuration.velocities

    quadrants = quadrantizer(position_array)
    potential_energy = 0.0
    kinetic_energy = np.sum(0.5*m*(np.linalg.norm(velocity_array, axis=1)**2))


    for i, particle_position in enumerate(position_array):
        try:
            if not position_array.shape[0]==2:
                _unit_displacements, displacement_norms = find_directions(
                    particle_position,
                    quadrants[i],
                    np.delete(position_array, slice(0,i+1), axis=0)
                )
            else:
                displacement_norms = np.linalg.norm(position_array - particle_position)
                potential_energy += p_energy(displacement_norms)
                break
            potential_energy += p_energy(displacement_norms)
        except IndexError:
            break


    return (kinetic_energy, potential_energy)


def equilibriate(initial_configuration: Configuration, nsteps: int = 10000) -> Tuple[Configuration, Tuple[ArrayLike]]:
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
