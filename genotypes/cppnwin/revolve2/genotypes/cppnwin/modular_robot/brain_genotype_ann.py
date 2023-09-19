import multineat

from revolve2.core.modular_robot import Body

from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1
from ._proprioception_brain_v1 import ProprioceptionCPPNNetwork as BrainANN
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import (
    Develop as body_develop,)

def random_v1(
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
    multineat_params: multineat.Parameters,
    output_activation_func: multineat.ActivationFunction,
    num_initial_mutations: int,
    body: Body,
    # n_env_conditions: int,
    # plastic_brain: int,
) -> Genotype:
    assert multineat_params.MutateOutputActivationFunction == False
    # other activation functions could work too, but this has been tested.
    # if you want another one, make sure it's output is between -1 and 1.
    #assert output_activation_func == multineat.ActivationFunction.SIGNED_SINE
    bd = body_develop(body).develop()
    robot_parts = bd[1]
    active_hinge_count = sum(1 for part in robot_parts.values() if "_active_hinge.ActiveHinge" in str(type(part)))

    if active_hinge_count == 0:
        active_hinge_count = 8

    # using more than 8 inpurs gives me segmentation fault! why?
    return base_random_v1(
        innov_db,
        rng,
        multineat_params,
        output_activation_func,
        active_hinge_count, #spider (8 joints)
        active_hinge_count,
        num_initial_mutations,
    )


#def develop_v1(genotype: Genotype, body: Body, env_condition: int,
#                n_env_conditions: int, plastic_brain: int) -> BrainANN:
#     return BrainANN(genotype.genotype)
def develop_v1(genotype: Genotype, body: Body, mask) -> BrainANN:
    return BrainANN(genotype.genotype, mask)