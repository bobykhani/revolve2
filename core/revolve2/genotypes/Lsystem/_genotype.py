from typing import Generic, TypeVar

from revolve2.core.modular_robot import ModularRobot

from revolve2.serialization import StaticData
from revolve2.serialization import Serializable


class BodybrainGenotype(Serializable):
    def develop(self) -> ModularRobot:
        """
        Develops the genome into a revolve_bot (proto-phenotype)
        :return: a RevolveBot instance
        :rtype: RevolveBot
        """
        raise NotImplementedError("Method must be implemented by genome")

    def random(self):
        raise NotImplementedError("Method must be implemented by genome")

    def mutate(self):
        raise NotImplementedError("Method must be implemented by genome")

    def crossover(self, parents):
        raise NotImplementedError("Method must be implemented by genome")

    def serialize(self) -> StaticData:
        raise NotImplementedError("Method must be implemented by genome")

    @classmethod
    def deserialize(cls, data: StaticData) -> Serializable:
        raise NotImplementedError("Method must be implemented by genome")
