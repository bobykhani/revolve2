from .NN_Controller import NeuralNetwork, Neuron, Connection


class ESNeuralNetwork(NeuralNetwork):
    def __init__(self, input_size=0, output_size=0):
        super().__init__(input_size, output_size)
        # Initialize without fixed hidden layers since ES-HyperNEAT will determine the network structure
        self.input_neurons = [Neuron() for _ in range(input_size)]
        self.output_neurons = [Neuron() for _ in range(output_size)]
        self.hidden_neurons = []  # Hidden neurons will be added dynamically
        self.connections = []
        self.neurons_by_position = {}  # Maps positions to neuron instances

    def initialize_io_neurons(self, num_inputs, num_outputs):
        # Dynamically adjust the number of input and output neurons
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.input_neurons = [Neuron() for _ in range(num_inputs)]
        self.output_neurons = [Neuron() for _ in range(num_outputs)]

    def add_neuron_to_hidden_layer(self, bias=0.1):
        # Dynamically adds a neuron to the hidden layer
        neuron = Neuron(bias=bias)
        self.hidden_neurons.append(neuron)
        return neuron

    def add_connection(self, from_neuron, to_neuron, weight):
        # Adds a connection between two neurons
        connection = Connection(from_neuron, to_neuron, weight)
        self.connections.append(connection)

    def develop_from_cppn(self, cppn, substrate_dimensions, num_inputs, num_outputs, connection_threshold=0.2,
                          initial_bias_range=(-1, 1), weight_range=(-2, 2)):
        """
        Develops the neural network based on outputs from a CPPN.

        Parameters:
        - cppn: The CPPN used for querying connection weights and biases.
        - substrate_dimensions: The dimensions of the substrate grid (e.g., (width, height) for 2D).
        - num_inputs, num_outputs: Number of input and output neurons.
        - connection_threshold: The threshold above which a CPPN output indicates a connection should be made.
        - initial_bias_range: The range from which to randomly select initial biases for neurons.
        - weight_range: The range from which to randomly select weights for connections if not specified by the CPPN.
        """
        # Initialize input and output neurons based on specified counts
        self.initialize_io_neurons(num_inputs, num_outputs)

        # Process inputs and outputs to place them on the substrate; for simplicity, assume inputs are on top edge and outputs on bottom
        input_positions = [(x, 0) for x in range(num_inputs)]
        output_positions = [(x, substrate_dimensions[1] - 1) for x in range(num_outputs)]

        # Initialize hidden neurons - in this example, we're not adding them until we find a connection that requires it
        hidden_positions = [(x, y) for x in range(substrate_dimensions[0]) for y in
                            range(1, substrate_dimensions[1] - 1)]

        # Example loop to go through all possible connections in a simplified manner
        for from_pos in input_positions + hidden_positions:
            for to_pos in hidden_positions + output_positions:
                # Query CPPN for the connection weight
                weight = cppn.query(from_pos, to_pos)

                # If weight is above the threshold, create the connection
                if abs(weight) > connection_threshold:
                    # Find or create neurons at these positions
                    from_neuron = self.find_or_create_neuron_at_position(from_pos)
                    to_neuron = self.find_or_create_neuron_at_position(to_pos)

                    # Apply weight
                    self.add_connection(from_neuron, to_neuron, weight)

    def find_or_create_neuron_at_position(self, position):
        """
        Finds an existing neuron at the given position, or creates a new one if it doesn't exist.

        Parameters:
        - position: A tuple representing the position of the neuron.

        Returns:
        - A Neuron instance at the given position.
        """
        # Check if there's already a neuron at the given position
        if position in self.neurons_by_position:
            return self.neurons_by_position[position]

        # If not, create a new neuron, add it to the dictionary, and also appropriately
        # add it to either input_neurons, output_neurons, or hidden_neurons
        new_neuron = Neuron()

        # Assuming positions (0, y) are inputs, positions (max_x, y) are outputs,
        # and everything else is hidden. Adjust according to your substrate design.
        if position[0] == 0:  # Example logic for determining neuron type by position
            self.input_neurons.append(new_neuron)
        elif position[0] == self.substrate_dimensions[0] - 1:
            self.output_neurons.append(new_neuron)
        else:
            self.hidden_neurons.append(new_neuron)

        self.neurons_by_position[position] = new_neuron
        return new_neuron
