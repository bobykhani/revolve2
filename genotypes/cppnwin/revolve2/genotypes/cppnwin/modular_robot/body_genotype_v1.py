"""Functions for CPPNWIN genotypes for a modular robot body."""

import math
from dataclasses import dataclass
from queue import Queue
from typing import Any, List, Optional, Set, Tuple

import multineat
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module, PassiveBone

from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1


def random_v1(
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
    multineat_params: multineat.Parameters,
    output_activation_func: multineat.ActivationFunction,
    num_initial_mutations: int,
) -> Genotype:
    """
    Create a CPPNWIN genotype for a modular robot body.

    :param innov_db: Multineat innovation database_karine_params. See Multineat library.
    :param rng: Random number generator.
    :param multineat_params: Multineat parameters. See Multineat library.
    :param output_activation_func: Activation function for the output layer. See Multineat library.
    :param num_initial_mutations: The number of times to mutate to create a random network.
    :returns: The created genotype.
    """
    return base_random_v1(
        innov_db,
        rng,
        multineat_params,
        output_activation_func,
        5,  # bias(always 1), pos_x, pos_y, pos_z, chain_length
        6,  # empty, brick, activehinge, rot0, rot90, PassiveBone, PBSize
        num_initial_mutations,
    )


@dataclass
class __Module:
    position: Tuple[int, int, int]
    forward: Tuple[int, int, int]
    up: Tuple[int, int, int]
    chain_length: int
    module_reference: Module


def develop_v1(
    genotype: Genotype,
) -> Body:
    """
    Develop a CPPNWIN genotype into a modular robot body.

    It is important that the genotype was created using a compatible function.

    :param genotype: The genotype to create the body from.
    :returns: The create body.
    :raises RuntimeError: In case a module is encountered that is not supported.
    """
    max_parts = 15

    body_net = multineat.NeuralNetwork()
    genotype.genotype.BuildPhenotype(body_net)

    to_explore: Queue[__Module] = Queue()
    grid: Set[Tuple[int, int, int]] = set()

    body = Body()

    to_explore.put(__Module((0, 0, 0), (0, -1, 0), (0, 0, 1), 0, body.core))
    grid.add((0, 0, 0))
    part_count = 1

    while not to_explore.empty():
        module = to_explore.get()

        children: List[Tuple[int, int]] = []  # child index, rotation

        if isinstance(module.module_reference, Core):
            children.append((Core.FRONT, 0))
            children.append((Core.LEFT, 1))
            children.append((Core.BACK, 2))
            children.append((Core.RIGHT, 3))
        elif isinstance(module.module_reference, Brick):
            children.append((Brick.FRONT, 0))
            children.append((Brick.LEFT, 1))
            children.append((Brick.RIGHT, 3))
        elif isinstance(module.module_reference, ActiveHinge):
            children.append((ActiveHinge.ATTACHMENT, 0))
        elif isinstance(module.module_reference, PassiveBone):
            children.append((PassiveBone.ATTACHMENT, 0))
        else:  # Should actually never arrive here but just checking module type to be sure
            raise RuntimeError()

        for (index, rotation) in children:
            if part_count < max_parts:
                child = ___add_child(body_net, module, index, rotation, grid, size=0.1)
                if child is not None:
                    to_explore.put(child)
                    part_count += 1

    body.finalize()
    return body


def __evaluate_cppn(
    body_net: multineat.NeuralNetwork,
    position: Tuple[int, int, int],
    chain_length: int,
) -> Tuple[Any, int]:
    """
    Get module type and orientation from a multineat CPPN network.

    :param body_net: The CPPN network.
    :param position: Position of the module.
    :param chain_length: Tree distance of the module from the core.
    :returns: (module type, orientation)
    """
    body_net.Input(
        [1.0, position[0], position[1], position[2], chain_length]
    )  # 1.0 is the bias input
    body_net.ActivateAllLayers()
    outputs = body_net.Output()

    # get module type from output probabilities
    type_probs = [outputs[0], outputs[1], outputs[2], outputs[3]]
    types = [None, Brick, ActiveHinge, PassiveBone]
    module_type = types[type_probs.index(min(type_probs))]

    # get rotation from output probabilities
    rotation_probs = [outputs[4]]
    rotation = rotation_probs.index(min(rotation_probs))

    #change range to bone size range
    #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

    #Y = (((outputs[6] - 0) / (1 - 0)) * (0.2 - 0.1)) + 0.1
    Y = ((outputs[5] - (-1)) / (1 - (-1))) * (0.2 - 0.1) + 0.1

    size = Y #0
    # if outputs[5]>0.5:
    #     size = 0.2
    # else:
    #     size = 0.1

    return (module_type, rotation, size)


def ___add_child(
    body_net: multineat.NeuralNetwork,
    module: __Module,
    child_index: int,
    rotation: int,
    grid: Set[Tuple[int, int, int]],
    size: float,

) -> Optional[__Module]:
    forward = __rotate(module.forward, module.up, rotation)
    position = __add(module.position, forward)
    chain_length = module.chain_length + 1

    # if grid cell is occupied, don't make a child
    # else, set cell as occupied
    if position in grid:
        return None
    else:
        grid.add(position)

    child_type, orientation, size = __evaluate_cppn(body_net, position, chain_length)
    if child_type is None:
        return None
    up = __rotate(module.up, forward, orientation)

    try:
        if child_type == PassiveBone:
            child = child_type(0, size=size)
        else:
            child = child_type(orientation * (math.pi / 2.0))
    except Exception as a:
        print("in",a)
    module.module_reference.children[child_index] = child

    return __Module(
        position,
        forward,
        up,
        chain_length,
        child,
    )


def __add(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def __timesscalar(a: Tuple[int, int, int], scalar: int) -> Tuple[int, int, int]:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def __cross(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def __dot(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def __rotate(
    a: Tuple[int, int, int], b: Tuple[int, int, int], angle: int
) -> Tuple[int, int, int]:
    """
    Rotates vector a a given angle around b.

    Angle from [0,1,2,3].
    90 degrees each.

    :param a: Vector a.
    :param b: Vector b.
    :param angle: The angle to rotate.
    :returns: A copy of a, rotated.
    """
    cosangle: int
    sinangle: int
    if angle == 0:
        cosangle = 1
        sinangle = 0
    elif angle == 1:
        cosangle = 0
        sinangle = 1
    elif angle == 2:
        cosangle = -1
        sinangle = 0
    else:
        cosangle = 0
        sinangle = -1

    return __add(
        __add(
            __timesscalar(a, cosangle),
            __timesscalar(__cross(b, a), sinangle),
        ),
        __timesscalar(b, __dot(b, a) * (1 - cosangle)),
    )
