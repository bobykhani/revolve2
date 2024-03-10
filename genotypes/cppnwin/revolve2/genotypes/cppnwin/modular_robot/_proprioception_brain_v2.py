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
from typing import List, cast
import multineat  # I assume you've imported this earlier

from .NN_Controller import NeuralNetwork
import matplotlib.pyplot as plt
import networkx as nx
import os


def apply_mask(inputs: List[float], mask: List[int]) -> List[float]:
    # Apply the mask only to the first 'n' elements of the inputs, where 'n' is the length of the mask
    masked_part = [input_val * mask_val for input_val, mask_val in zip(inputs[:len(mask)], mask)]

    # Keep the remaining part of the inputs unchanged
    remaining_part = inputs[len(mask):]

    # Combine the masked and remaining parts
    return masked_part + list(remaining_part)

class ProprioceptionCPPNNetwork(ActorController):
    _genotype: multineat.Genome

    def __init__(self, genotype: multineat.Genome, mask, joint_count):
        self._genotype = genotype
        self._dof_targets = []
        self._sensors = None
        self._n_joints = joint_count
        self._dof_ranges = 1
        self._steps = 0
        self._mask = mask
        self.controller = None
        # self.develop_controller()

    def develop_controller(self):
        # self._n_joints = len(dof_ids)
        # self._dof_targets = [0] * self._n_joints
        if not hasattr(self, 'brain_net'):
            self.brain_net = multineat.NeuralNetwork()
            self._genotype.BuildPhenotype(self.brain_net)
        # check if the controller connections are already developed
        if self.controller != None:
            return

        size = self._n_joints


        nn = NeuralNetwork(self._n_joints, self._n_joints, [self._n_joints,self._n_joints])
        self.controller = nn.develop_network()
        for i in range(size):
            for j in range(size):

                weight = self._evaluate_network(self.brain_net, [0, i, j])
                # print(f"Input to Hidden Weight ({i}, {j}): {weight}")  # Debugging statement

                nn.add_connection(nn.input_neurons[i], nn.hidden_neurons[0][j], weight)

        # Establish connections between the first and the second hidden layer
        for i in range(size):
            for j in range(size):
                weight = self._evaluate_network(self.brain_net, [1, i, j])
                nn.add_connection(nn.hidden_neurons[0][i], nn.hidden_neurons[1][j], weight)
                # print(f"Hidden to Hidden Weight ({i}, {j}): {weight}")  # Debugging statement
        # Establish connections between the second hidden layer and the output layer
        for i in range(size):
            for j in range(size):
                weight = self._evaluate_network(self.brain_net, [2, i, j])
                nn.add_connection(nn.hidden_neurons[1][i], nn.output_neurons[j], weight)

        # Threshold to identify and prune weak connections
        threshold = 0.7
        pruned_connections = []
        for connection in nn.connections:
            # Check if the connection's weight is below the threshold
            if abs(connection.weight) < threshold:
                # Ensure the connection is not from an input neuron and not to an output neuron
                if connection.from_neuron.id not in [neuron.id for neuron in nn.input_neurons] and \
                        connection.to_neuron.id not in [neuron.id for neuron in nn.output_neurons]:
                    pruned_connections.append(connection)

        # Print connections before pruning for debugging
        # print("Connections before pruning:")
        # for conn in nn.connections:
        #     print(f"{conn.from_neuron.id} -> {conn.to_neuron.id}, weight: {conn.weight}")

        # Remove pruned connections from the network
        for connection in pruned_connections:
            nn.connections.remove(connection)

        # Print connections after pruning for debugging
        # print("Connections after pruning:")
        # for conn in nn.connections:
        #     print(f"{conn.from_neuron.id} -> {conn.to_neuron.id}, weight: {conn.weight}")
        self.plot_nn(nn)

        self.controller = nn



    def set_sensors(self, sensors: BatchResults):
        self._sensors = sensors

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        self._n_joints = len(dof_ids)
        self._dof_targets = [0] * self._n_joints

        # self.develop_controller()

        return self

    def step(self, dt: float) -> None:
        # Only build the network if it doesn't exist, otherwise reuse
        # if not hasattr(self, 'brain_net'):
        #     self.brain_net = multineat.NeuralNetwork()
        #     self._genotype.BuildPhenotype(self.brain_net)
        self.develop_controller()

        sin = math.sin(self._steps)

        # Assuming you want to add sin to each sensor value
        self._sensors = apply_mask(self._sensors,self._mask)

        closed_loop = [(sensor + sin) for sensor in self._sensors]

        if self._mask != None:
            mask_str = str(self._mask)
            #
            # # Save the mask string to a text file in a new line
            # with open('mask_strings.txt', 'a') as file:
            #     file.write(mask_str + '\n')

            # closed_loop = apply_mask(closed_loop, self._mask)

        output = self.controller.feedforward(closed_loop)
        # output = self._evaluate_network(self.brain_net, closed_loop)
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


    def get_dof_targets(self) -> List[float]:
        return self._dof_targets


    @staticmethod
    def _evaluate_network(
            network: multineat.NeuralNetwork, inputs: List[float]
    ) -> float:
        # Convert the mask to a string representation (depends on the type of 'mask')

        network.Input(inputs)
        network.ActivateAllLayers()
        return cast(float, network.Output())[0]  # TODO missing multineat typing

    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, data):
        pass


    def plot_nn(self, nn):

        # Create a directed graph
        G = nx.DiGraph()

        # Add all neurons as nodes first to ensure they appear even if they have no connections
        for neuron in nn.input_neurons:
            G.add_node(neuron.id)
        for layer in nn.hidden_neurons:
            for neuron in layer:
                G.add_node(neuron.id)
        for neuron in nn.output_neurons:
            G.add_node(neuron.id)

        # Add edges for connections with non-zero weights and ensure we reference the neuron IDs
        for connection in nn.connections:
            if connection.weight != 0:
                G.add_edge(connection.from_neuron.id, connection.to_neuron.id, weight=connection.weight)

        # Define positions of nodes
        pos = {}
        layer_width = max(len(nn.input_neurons), max(len(layer) for layer in nn.hidden_neurons), len(nn.output_neurons))
        input_y = layer_width // 2
        output_y = layer_width // 2

        # Position input neurons
        for i, neuron in enumerate(nn.input_neurons):
            pos[neuron.id] = (0, input_y - i)  # Place inputs on the left, spaced out

        # Position hidden neurons
        hidden_layer_spacing = 2  # Adjust this value to space out the hidden layers appropriately
        for h, layer in enumerate(nn.hidden_neurons):
            hidden_y = layer_width // 2
            for j, neuron in enumerate(layer):
                pos[neuron.id] = (1 + h * hidden_layer_spacing, hidden_y - j)  # Space out neurons in hidden layers

        # Position output neurons
        for i, neuron in enumerate(nn.output_neurons):
            pos[neuron.id] = (
            1 + len(nn.hidden_neurons) * hidden_layer_spacing, output_y - i)  # Place outputs on the right, spaced out

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold',
                arrows=True, arrowsize=20)

        # Add edge labels to display weights
        edge_labels = {(connection.from_neuron.id, connection.to_neuron.id): f'{connection.weight:.2f}' for connection
                       in nn.connections if connection.weight != 0}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        # make a folder nn_plots if not exist and save plots in it
        if not os.path.exists('nn_plots'):
            os.makedirs('nn_plots')
        import time
        plt.savefig('nn_plots/nn_plot'+str(time.time())+'.png')
        # plt.show()
        plt.close()