"""Setup and running of the optimize modular program."""

import logging
import random
from random import Random

import multineat
from genotype import random as random_genotype
from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId
from revolve2.core.optimization import ProcessIdGen, Process
from revolve2.core.config import Config
from revolve2.standard_resources import terrains

async def main() -> None:
    args = Config()._get_params()
    mainpath = args.mainpath

    """Run the optimization process."""
    # number of initial mutations for body and brain CPPNWIN networks
    NUM_INITIAL_MUTATIONS = 10

    SIMULATION_TIME = 20
    SAMPLING_FREQUENCY = 8
    CONTROL_FREQUENCY = 5

    POPULATION_SIZE = 10
    OFFSPRING_SIZE = 10
    NUM_GENERATIONS = 10

    FITNESS_MEASURE = 'speed_y'#'sum_mask'

    ROBOT = 'blokky'


    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info("Starting optimization")

    # prepares params for environmental conditions
    seasonal_conditions_parsed = []
    seasonal_conditions = args.seasons_conditions.split('#')
    for seasonal_condition in seasonal_conditions:
        params = seasonal_condition.split('_')
        seasonal_conditions_parsed.append([params[0], params[1], params[2], params[3], params[4]])

    # random number generator
    rng = Random()
    rng.seed(random.random())

    # database_karine_params
    database = open_async_database_sqlite("./database", create=True)

    # process_id_gen = ProcessIdGen()

    # unique database_karine_params identifier for optimizer
    db_id = DbId.root("test1")

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    initial_population = [
        random_genotype(innov_db_body, innov_db_brain, rng, NUM_INITIAL_MUTATIONS, ROBOT)
        for _ in range(POPULATION_SIZE)
    ]
    # process_id = process_id_gen.gen()
    maybe_optimizer = await Optimizer.from_database(
        database=database,
        db_id=db_id,
        # process_id=process_id,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        # process_id_gen=process_id_gen,
        fitness_measure=FITNESS_MEASURE,
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:
        optimizer = await Optimizer.new(
            database=database,
            # process_id=process_id,
            # process_id_gen=process_id_gen,
            fitness_measure=FITNESS_MEASURE,
            db_id=db_id,
            initial_population=initial_population,
            rng=rng,
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            offspring_size=OFFSPRING_SIZE,
            experiment_name=args.experiment_name,
            max_modules=args.max_modules,
            crossover_prob=0.8,
            mutation_prob=0.2,
            substrate_radius=args.substrate_radius,
            run_simulation=args.run_simulation,
            simulator=args.simulator,
            robot = ROBOT,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info("Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
