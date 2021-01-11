import unittest

from nca.core.actor.fitness_evaluation import OnesFitness
from nca.core.actor.individual import Individual
from nca.core.genome.operators.initialization import UniformInitialization, BinaryInitialization
from nca.core.genome.representations.valued_representation import ValuedRepresentation
from revolve.evosphere.fitness_evaluation import DisplacementFitness, RotationalFitness, DisplacementRotationalFitness
from revolve.robot.robot import Robot
from test.simulation.simulator.mock_measures import MockPerformanceMeasures


class TestFitness(unittest.TestCase):

    def test_displacement(self):

        fitness = DisplacementFitness()
        initialization = BinaryInitialization()
        individual = Robot(ValuedRepresentation(initialization))
        individual.measures = MockPerformanceMeasures()

        fitness_value = fitness(individual)
        self.assertNotEqual(fitness_value, 0.0)

    def test_rotational(self):

        fitness = RotationalFitness()
        individual = Robot(ValuedRepresentation(initialization=BinaryInitialization()))
        individual.measures = MockPerformanceMeasures()

        fitness_value = fitness(individual)
        self.assertNotEqual(fitness_value, 0.0)

    def test_ones(self):

        fitness = OnesFitness()
        individual = Individual(ValuedRepresentation(initialization=UniformInitialization()))

        fitness_value = fitness(individual)
        self.assertNotEqual(fitness_value, 0.0)

    def test_multi(self):

        fitness = DisplacementRotationalFitness()
        individual = Individual(ValuedRepresentation(initialization=BinaryInitialization()))
        individual.measures = MockPerformanceMeasures()
        fitness_value = fitness(individual)
        self.assertNotEqual(fitness_value, 0.0)