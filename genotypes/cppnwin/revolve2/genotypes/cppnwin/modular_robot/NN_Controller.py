# controller.py

from typing import List
import numpy as np
import random


class Neuron:
    _id_counter = 0  # Class-level counter to give each neuron a unique ID

    def __init__(self, bias=0.1):
        self.bias = bias
        self.value = 0.0
        self.id = Neuron._id_counter  # Assign a unique ID to the neuron
        Neuron._id_counter += 1


class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight


class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers=None):
        self.input_size = input_size
        self.output_size = output_size
        self.input_neurons = [Neuron() for _ in range(input_size)]
        self.output_neurons = [Neuron() for _ in range(output_size)]
        self.hidden_neurons = self.create_hidden_layers(hidden_layers) if hidden_layers else []
        self.connections = []

    def create_hidden_layers(self, hidden_layers):
        return [[Neuron() for _ in range(size)] for size in hidden_layers]

    def random_init(self, weight_range=(-1.0, 1.0)):
        # Initialize connections with random weights
        layers = [self.input_neurons] + self.hidden_neurons + [self.output_neurons]
        for i in range(len(layers) - 1):
            for from_neuron in layers[i]:
                for to_neuron in layers[i + 1]:
                    weight = random.uniform(*weight_range)
                    self.add_connection(from_neuron, to_neuron, weight)
                    # print(f"Connection from {from_neuron} to {to_neuron.id} with weight {weight}")

    def add_connection(self, from_neuron, to_neuron, weight):
        # print(f"Adding connection from neuron {from_neuron.id} to neuron {to_neuron.id} with weight {weight}")
        self.connections.append(Connection(from_neuron, to_neuron, weight))
        # print(
        #     f"Total connections: {len(self.connections)}")  # This will print the number of connections after adding the new one

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, input):
        if input > 0:
            return input
        else:
            return 0

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def feedforward(self, inputs: List[float]) -> List[float]:
        # Reset neuron values for input, hidden, and output neurons
        for neuron in self.input_neurons:
            neuron.value = 0.0
        for layer in self.hidden_neurons:
            for neuron in layer:
                neuron.value = 0.0
        for neuron in self.output_neurons:
            neuron.value = 0.0

        # Set input values
        for i, value in enumerate(inputs):
            self.input_neurons[i].value = value

        # Propagate the values forward through the network
        # For connections from input neurons to the first hidden layer or directly to output neurons
        for connection in self.connections:
            connection.to_neuron.value += connection.from_neuron.value * connection.weight

        # Apply activation function (e.g., tanh) to hidden and output neuron values
        for layer in self.hidden_neurons:
            for neuron in layer:
                # neuron.value = self.sig(neuron.value)
                # neuron.value = np.arctan(neuron.value)
                # neuron.value =self.softmax(neuron.value)
                # neuron.value = self.relu(neuron.value)
                # neuron.value = self.sig(neuron.value)
                # neuron.value = self.softmax(neuron.value)
                neuron.value = np.tanh(neuron.value)

        for neuron in self.output_neurons:
            # neuron.value = self.sig(neuron.value)
            neuron.value = np.tanh(neuron.value)
            # neuron.value = self.relu(neuron.value)
            # neuron.value = np.arctan(neuron.value)
            # neuron.value = self.softmax(neuron.value)

        # Return output values
        return [neuron.value for neuron in self.output_neurons]

    def develop_network(self):
        # Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
        nn = NeuralNetwork(input_size=self.input_size, output_size=self.output_size, hidden_layers=[self.input_size, self.output_size])
        return nn



