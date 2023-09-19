import random


class MaskGenome:
    def __init__(self, length):
        self.length = length
        self.genome = [random.choice([0, 1]) for _ in range(length)]

    def mask_inputs(self, inputs):
        return [input_val * mask_val for input_val, mask_val in zip(inputs, self.genome)]

    def mutate(self, mutation_rate=0.1):
        for i in range(self.length):
            if random.random() < mutation_rate:
                self.genome[i] = 1 - self.genome[i]

    def crossover(self, other):
        new_genome = MaskGenome(self.length)
        crossover_point = random.randint(1, self.length - 1)

        new_genome.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        return new_genome

    def clone(self):
        new_genome = MaskGenome(self.length)
        new_genome.genome = self.genome.copy()
        return new_genome
