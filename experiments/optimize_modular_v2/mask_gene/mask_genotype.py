import random


class MaskGenome:
    def __init__(self, length):
        self.length = length
        self.genome = [random.choice([0, 1]) for _ in range(length)]

    def mask_inputs(self, inputs):
        return [input_val * mask_val for input_val, mask_val in zip(inputs, self.genome)]

    def mutate(self, mutation_rate=0.02):
        for i in range(self.length):
            if random.random() < mutation_rate:
                self.genome[i] = 1 - self.genome[i]

    def crossover(self, other, crossover_rate=0.8):
        new_genome = MaskGenome(self.length)
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, self.length - 1)
            new_genome.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        else:
            new_genome.genome = self.genome.copy()  # Always return a new genome object
        return new_genome

    def clone(self):
        new_genome = MaskGenome(self.length)
        new_genome.genome = self.genome.copy()
        return new_genome

# # Testing the MaskGenome class
# genome1 = MaskGenome(10)
# genome2 = genome1.crossover(MaskGenome(10), 0.5)  # Test crossover with a crossover rate of 0.5
# genome3 = genome1.clone
