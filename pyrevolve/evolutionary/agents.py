import random
from typing import List
import numpy as np

from pyrevolve.evolutionary.individual import Individual, TestIndividual
from pyrevolve.evolutionary.fitness import Fitness
from pyrevolve.shared.abstract.iterator import Iterator


class Agents(Iterator):

    def __init__(self, agents: List[Individual] = []):
        super().__init__(agents)

    def add(self, agent: Individual):
        super().add(agent)

    def remove(self, agent: Individual):
        super().remove(agent)

    def compound(self, agents):
        return self.collection.extend(agents.collection)

    def get_best(self):
        best_agent: Individual = None
        best_fitness: Fitness = Fitness.worst()

        for evolutionary_agent in self:
            if best_fitness > evolutionary_agent.fitness:
                best_fitness = evolutionary_agent.fitness
                best_agent = evolutionary_agent

        return best_agent

    def get_worst(self):
        worst_agent: Individual = None
        worst_fitness: Fitness = Fitness.best()

        for agent in self:
            if worst_fitness < agent.fitness:
                worst_fitness = agent.fitness
                worst_agent = agent

        return worst_agent

    def average_fitness(self):
        # TODO individual.fitness
        return np.mean([individual.fitness.fitness if individual.fitness is not None else 0.0 for individual in self])

    def update_age(self):
        for agent in self:
            agent.age.update()


class TestAgents(Agents):

    def __init__(self, n: int):
        super().__init__([TestIndividual() for _ in range(n)])