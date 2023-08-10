from typing import List, Set, Tuple, cast
import random
import multineat
from squaternion import Quaternion
import pprint
import numpy as np
import math

from revolve2.core.modular_robot import ActiveHinge, Body
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import (
    BatchResults,
)

class ProprioceptionCPPNNetwork(ActorController):
    _genotype: multineat.Genome

    def __init__(self, genotype: multineat.Genome):
        self._genotype = genotype
        self._dof_targets = []
        self._sensors = None
        self._n_joints = 0
        self._dof_ranges = 1
        self._steps = 0

    def set_sensors(self, sensors: BatchResults):
        self._sensors = sensors

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        self._n_joints = len(dof_ids)
        self._dof_targets = [0] * self._n_joints

        return self

    def step(self, dt: float) -> None:
        # Only build the network if it doesn't exist, otherwise reuse
        if not hasattr(self, 'brain_net'):
            self.brain_net = multineat.NeuralNetwork()
            self._genotype.BuildPhenotype(self.brain_net)

        sin = math.sin(self._steps)

        # Assuming you want to add sin to each sensor value
        closed_loop = [(sensor + sin) for sensor in self._sensors]

        output = self._evaluate_network(self.brain_net, closed_loop)
        output = list(np.clip(output, a_min=-self._dof_ranges, a_max=self._dof_ranges))

        # Option 1: Smoothed Transition to New Target
        alpha = 0.1
        self._dof_targets = [(alpha * new_output + (1 - alpha) * old_output) for new_output, old_output in
                             zip(output, self._dof_targets)]

        # Option 2: Limit Rate of Change (can be combined with Option 1 if needed)
        max_rate_of_change = 0.05
        change = [new - old for new, old in zip(output, self._dof_targets)]
        change = [np.clip(delta, -max_rate_of_change, max_rate_of_change) for delta in change]
        self._dof_targets = [old + delta for old, delta in zip(self._dof_targets, change)]

        self._steps += 1

    # def step(self, dt: float) -> None:
    #     brain_net = multineat.NeuralNetwork()
    #     self._genotype.BuildPhenotype(brain_net)
    #
    #     sin = math.sin(self._steps)
    #     self._sensors = [sin] * len(self._sensors)
    #     #closed_loop = self._sensors# + sin
    #     closed_loop = [(sensor + sin) for sensor in self._sensors]
    #
    #     output = self._evaluate_network(
    #         brain_net, closed_loop,
    #     )
    #
    #     output =  list(np.clip(output, a_min=-self._dof_ranges, a_max=self._dof_ranges))
    #
    #     #sum current dof targets with previous dof targets
    #
    #     self._dof_targets = output#[self._dof_targets[i] + output[i] for i in range(len(output))]
    #
    #     #################################################
    #     self._steps += 1

    def get_dof_targets(self) -> List[float]:
        return self._dof_targets

    @staticmethod
    def _evaluate_network(
        network: multineat.NeuralNetwork, inputs: List[float]
    ) -> float:
        network.Input(inputs)
        network.ActivateAllLayers()
        return cast(float, network.Output())  # TODO missing multineat typing
    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, data):
        pass