import os
import string
from typing import List

from nca.core.actor.agent import Agent
from nca.core.actor.fitness import Fitness
from nca.core.actor.individual_factory import ActorFactory
from nca.core.ecology import PopulationEcology
from nca.core.evolution.evolutionary_configurations import EvolutionaryConfiguration, GeneticAlgorithmConfiguration
from revolve.evosphere.biosphere import Biosphere, IndividualBiosphere

from revolve.evosphere.ecosphere import Ecosphere
from revolve.evosphere.evosphere import AgentEvosphere
from revolve.robot.birth_clinic import BirthClinic, AgentBirthClinic
from revolve.robot.brain.brain import Brain
from revolve.robot.brain.brain_builder import BrainBuilder, AgentBrainBuilder
from revolve.robot.brain.neural_network_brain import NeuralNetworkBrain
from revolve.robot.development_request import BrainDevelopmentRequest
from simulation.simulation_manager import SimulationManager
from simulation.simulator.simulator_type import SimulatorType


class CoppeliaBrainBuilder(AgentBrainBuilder):
    def __init__(self, brain_type: type(Brain) = NeuralNetworkBrain):
        super().__init__(brain_type)

    def create(self, brain_development_request: BrainDevelopmentRequest) -> Brain:
        return self.brain_type()


class CoppeliaBirthClinic(AgentBirthClinic):
    def __init__(self, brain_builder: BrainBuilder = CoppeliaBrainBuilder()):
        super().__init__()
        self.brain_builder: BrainBuilder = brain_builder

    def _create(self, development_request: BrainDevelopmentRequest) -> object:
        return Agent(development_request.genotype, self.brain_builder.create(development_request))


class CoppeliaEcosphere(Ecosphere):

    def __init__(self, filename: string = "evoman", fitness_type: type(Fitness) = None):
        super().__init__(filename, fitness_type, SimulatorType.COPPELIA)

        if not os.path.exists(filename):
            os.makedirs(filename)

        self.filename = filename


class CoppeliaBiosphere(IndividualBiosphere):

    def __init__(self,
                 population_ecology: PopulationEcology = PopulationEcology(),
                 actor_factory: ActorFactory = ActorFactory(),
                 birth_clinic: BirthClinic = CoppeliaBirthClinic(),
                 ecospheres: List[Ecosphere] = None):
        if ecospheres is None:
            ecospheres = [CoppeliaEcosphere()]
        super().__init__(population_ecology, actor_factory, birth_clinic, ecospheres)


class CoppeliaEvosphere(AgentEvosphere):
    def __init__(self, biosphere: Biosphere = CoppeliaBiosphere(),
                 evolutionary_configuration: EvolutionaryConfiguration = GeneticAlgorithmConfiguration(),
                 simulation: SimulationManager = SimulationManager(), debug=True):
        super().__init__(biosphere, evolutionary_configuration, simulation, debug)