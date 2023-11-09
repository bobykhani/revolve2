import multineat
from ._genotype import Genotype

def random_v1(
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
    multineat_params: multineat.Parameters,
    output_activation_func: multineat.ActivationFunction,
    num_inputs: int,
    num_outputs: int,
    num_initial_mutations: int,
) -> Genotype:
    """
    Create a random CPPNWIN genotype.
    A CPPNWIN network starts empty.
    A random network is created by mutating `num_initial_mutations` times.
    """
    print("random_v1 is called")

    try:
        genotype = multineat.Genome(
            0,  # ID
            num_inputs,
            0,  # n_hidden
            num_outputs,
            False,  # FS_NEAT
            output_activation_func,  # output activation type
            multineat.ActivationFunction.TANH,  # hidden activation type
            0,  # seed_type
            multineat_params,
            0,  # number of hidden layers
        )
    except Exception as e:
        print(f"Error creating initial genome: {e}")
        raise

    print("random_v1 is middle")

    try:
        for i in range(num_initial_mutations):
            print(f"Mutation {i+1} of {num_initial_mutations}")
            genotype = genotype.MutateWithConstraints(
                False,
                multineat.SearchMode.BLENDED,
                innov_db,
                multineat_params,
                rng,
            )
    except IndexError as index_err:
        print(f"IndexError during mutation: {index_err}")
        # Additional logging or error handling specific to IndexError
        raise
    except Exception as general_err:
        print(f"General error during mutation: {general_err}")
        raise

    print("random_v1 is finished")

    return Genotype(genotype)
