import unittest

from nca.core.abstract.configurations import PopulationConfiguration
from nca.core.actor.actors import Actors
from nca.core.actor.individual_factory import IndividualFactory
from nca.core.ecology import PopulationEcology
from nca.core.ecology.population_management import PopulationManagement
from nca.core.evolution.evolutionary_algorithm import EvolutionaryAlgorithm
from nca.core.evolution.evolutionary_configurations import GeneticAlgorithmConfiguration


class TestEvolutionaryAlgorithm(unittest.TestCase):

    def test_initialize(self):

        configuration = GeneticAlgorithmConfiguration()
        evolutionary_algorithm = EvolutionaryAlgorithm(configuration)

        population_ecology = PopulationEcology(PopulationManagement())
        population_ecology.initialize(IndividualFactory().create(10))

        for population in population_ecology.management.populations():
            for individual in population.individuals:
                self.assertIsNotNone(individual.representation)

    def test_run(self):

        configuration = GeneticAlgorithmConfiguration()
        evolutionary_algorithm = EvolutionaryAlgorithm(configuration)

        population_ecology = PopulationEcology(PopulationManagement())
        population_ecology.initialize(IndividualFactory().create(PopulationConfiguration().population_size))

        evolutionary_algorithm.run(population_ecology.management.populations()[0], evaluator)

        for population in population_ecology.management.populations():
            for individual in population.individuals:
                self.assertIsNotNone(individual.representation)


def evaluator(offspring: Actors):
    return offspring
