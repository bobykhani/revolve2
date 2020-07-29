import string

from pyrevolve.evolutionary.fitness import DisplacementFitness
from pyrevolve.shared.configuration import Configuration


class PopulationConfiguration(Configuration):

    def __init__(self, filename: string = "population.config"):
        super().__init__(filename)
        self.selection_size: int = 5
        self.tournament_size: int = 2
        self.population_size: int = 10


class SpeciationConfiguration(PopulationConfiguration):

    def __init__(self):
        super().__init__("speciation.config")


class RobotConfiguration(Configuration):

    def __init__(self):
        super().__init__("evolutionary.config")
        self.number_of_robots = 10


class EvolutionaryConfiguration(Configuration):

    def __init__(self):
        super().__init__("evolutionary.config")
        self.number_of_robots = 10
        self.fitness = DisplacementFitness()


class EvoSphereConfiguration(Configuration):

    def __init__(self):
        super().__init__("evosphere.config")
        self.number_of_environments = 1
        self.number_of_generations = 50


class RepresentationConfiguration(Configuration):

    def __init__(self):
        super().__init__("robogen.config")
        self.genome_size = 10
        self.minimum_value = -1.0
        self.maximum_value = 1.0


class MutationConfiguration(Configuration):
    def __init__(self):
        super().__init__("mutation.config")
        self.mutation_probability = 0.25


class RecombinationConfiguration(Configuration):
    def __init__(self):
        super().__init__("mutation.config")
        self.recombination_probability = 0.5