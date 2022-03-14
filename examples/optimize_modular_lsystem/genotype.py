from __future__ import annotations

import sys
from random import Random

from revolve2.serialization import StaticData, Serializable, SerializeError
from revolve2.core.optimization.ea.modular_robot import BodybrainGenotype

from core.revolve2.genotypes.Lsystem.Lsystem_genotype import lsystem,LsystemConfig
from core.revolve2.genotypes.Lsystem.initialization import random_initialization
from core.revolve2.genotypes.Lsystem.mutation.standard_mutation import standard_mutation
from core.revolve2.genotypes.Lsystem.crossover.standard_crossover import standard_crossover
from core.revolve2.genotypes.Lsystem.crossover.crossover import CrossoverConfig
from core.revolve2.genotypes.Lsystem.mutation.mutation import MutationConfig
from core.revolve2.genotypes.Lsystem.crossover import crossover

def main():
    gen: lsystem = Genotype.random()
    gen.mutate()

class Genotype(lsystem, Serializable):
    @classmethod
    def random(cls) -> Genotype:
        return random_initialization(LsystemConfig())

    @classmethod
    def mutate(self, Genotype) -> Genotype:
        genconf = LsystemConfig()
        mutconf = MutationConfig(mutation_prob=0.8, genotype_conf=genconf)
        return standard_mutation(Genotype, mutconf)

    @classmethod
    def crossover(self, parents):
        conf = CrossoverConfig(0.8)
        genconf = LsystemConfig()
        return standard_crossover(parents, crossover_conf=conf, genotype_conf=genconf)
        #return parents[0]

    def serialize(self) -> StaticData:
        test = {
            "body": self._body_genotype.serialize()
        }
        return test

    @classmethod
    def deserialize(cls, data: StaticData) -> Genotype:
        #if type(data) != dict:
        #    raise SerializeError()
        return lsystem.deserialize(data)
