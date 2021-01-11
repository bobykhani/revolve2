from typing import List

from nca.core.actor.fitness_evaluation import OnesFitness, FitnessEvaluation
from nca.core.actor.individual_factory import ActorFactory
from nca.core.ecology import PopulationEcology
from nca.core.evolution.evolutionary_configurations import EvolutionaryConfiguration, GeneticAlgorithmConfiguration
from nca.evolution import Evolution


class RepeatedEvolution(Evolution):

    def __init__(self, evolutionary_configuration: EvolutionaryConfiguration = GeneticAlgorithmConfiguration(),
                 fitness_evaluation: FitnessEvaluation = OnesFitness(), individual_factory: ActorFactory = ActorFactory(),
                 debug=True, repetitions: int = 10):
        super().__init__(evolutionary_configuration, fitness_evaluation, individual_factory, debug)
        self.repetitions: int = repetitions

    def evolve(self):
        results: List[PopulationEcology] = []
        for _ in range(self.repetitions):
            results.append(super().evolve())

        #return Summary().analyze(statistics_list)