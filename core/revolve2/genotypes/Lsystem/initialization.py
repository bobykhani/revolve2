import random

from .Lsystem_genotype import Alphabet, lsystem

def _generate_random_grammar(conf):
    """
    Initializing a new genotype,
    :param conf: e_max_groups, maximum number of groups of symbols
    :return: a random new Genome
    :rtype: dictionary
    """
    s_segments = random.randint(1, conf.e_max_groups)
    grammar = {}

    for symbol in Alphabet.modules():

        if symbol[0] == conf.axiom_w:
            grammar[symbol[0]] = [[conf.axiom_w, []]]
        else:
            grammar[symbol[0]] = []

        for s in range(0, s_segments):
            symbol_module = random.randint(1, len(Alphabet.modules()) - 1)
            symbol_morph_moving = random.randint(
                0, len(Alphabet.morphology_moving_commands()) - 1
            )
            symbol_morph_mounting = random.randint(
                0, len(Alphabet.morphology_mounting_commands()) - 1
            )

            grammar[symbol[0]].extend(
                [
                    lsystem.build_symbol(Alphabet.modules()[symbol_module], conf),
                    lsystem.build_symbol(
                        Alphabet.morphology_moving_commands()[symbol_morph_moving], conf
                    ),
                    lsystem.build_symbol(
                        Alphabet.morphology_mounting_commands()[symbol_morph_mounting], conf
                    ),
                ]
            )
    return grammar


def random_initialization(conf):
    """
    Initializing a random genotype.
    :type conf: PlasticodingConfig
    :return: a Genome
    :rtype: lsystem
    """
    genotype = lsystem(conf, 0)
    genotype.grammar = _generate_random_grammar(conf)

    return genotype
