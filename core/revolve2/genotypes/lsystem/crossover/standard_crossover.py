import random

from ..lsystem import Alphabet, LsystemConfig, lsystem


def generate_child_genotype(parent_genotypes, genotype_conf, crossover_conf):
    """
    Generates a child (individual) by randomly mixing production rules from two parents

    :param parents: parents to be used for crossover

    :return: child genotype
    """
    grammar = {}
    crossover_attempt = random.uniform(0.0, 1.0)
    if crossover_attempt > crossover_conf.crossover_prob:
        grammar = parent_genotypes[0].grammar
    else:
        for letter in Alphabet.modules():
            parent = random.randint(0, 1)
            # gets the production rule for the respective letter
            grammar[letter[0]] = parent_genotypes[parent].grammar[letter[0]]

    genotype = lsystem(genotype_conf, "tmp")
    genotype.grammar = grammar
    return genotype.clone()


def standard_crossover(parent_individuals, genotype_conf, crossover_conf):
    """
    Creates a child (individual) through crossover with two parents

    :param parent_genotypes: genotypes of the parents to be used for crossover
    :return: genotype result of the crossover
    """
    parent_genotypes = [p for p in parent_individuals]
    new_genotype = generate_child_genotype(
        parent_genotypes, genotype_conf, crossover_conf
    )
    return new_genotype
