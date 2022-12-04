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



def find_particle_quadrant(positions) -> int:
    if positions[0] < L/2:
        if positions[1] < L/2:
            return 3
        else:
            return 1
    elif positions[1] > L/2: return 2
    else: return 4

quadrantizer = np.vectorize(find_particle_quadrant, signature='(2)->()')
