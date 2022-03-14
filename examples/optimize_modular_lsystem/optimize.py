import logging
from random import Random

from revolve2.core.database.sqlite import Database as DbSqlite

from genotype import Genotype
from optimizer import Optimizer


async def main():

    SIMULATION_TIME = 5
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    POPULATION_SIZE = 100
    OFFSPRING_SIZE = 50
    NUM_GENERATIONS = 100

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting optimization")

    # random number generator
    rng = Random()
    rng.seed(10)

    # database
    database = await DbSqlite.create(f"database")

    initial_population = [
        Genotype.random()
        for _ in range(POPULATION_SIZE)
    ]

    ep = await Optimizer.create(
        database,
        initial_population=initial_population,
        initial_fitness=None,
        rng=rng,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
        population_size=POPULATION_SIZE,
        offspring_size=OFFSPRING_SIZE,
    )

    logging.info("Starting optimization process..")

    await ep.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
