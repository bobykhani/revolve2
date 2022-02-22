import random

from ..Lsystem_genotype import Alphabet, lsystem


def handle_deletion(genotype):
    """
    Deletes symbols from genotype

    :param genotype: genotype to be modified

    :return: genotype
    """
    target_production_rule = random.choice(list(genotype.grammar))
    if (len(genotype.grammar[target_production_rule])) > 1:
        symbol_to_delete = random.choice(genotype.grammar[target_production_rule])
        if symbol_to_delete[0] != Alphabet.CORE_COMPONENT:
            genotype.grammar[target_production_rule].remove(symbol_to_delete)
#            genotype_logger.info(
#                f'mutation: remove in {genotype.id} for {target_production_rule} at {symbol_to_delete[0]}.')
    return genotype


def handle_swap(genotype):
    """
    Swaps symbols within the genotype

    :param genotype: genotype to be modified

    :return: genotype
    """
    target_production_rule = random.choice(list(genotype.grammar))
    if (len(genotype.grammar[target_production_rule])) > 1:
        symbols_to_swap = random.choices(population=genotype.grammar[target_production_rule], k=2)
        for symbol in symbols_to_swap:
            if symbol[0] == Alphabet.CORE_COMPONENT:
                return genotype
        item_index_1 = genotype.grammar[target_production_rule].index(symbols_to_swap[0])
        item_index_2 = genotype.grammar[target_production_rule].index(symbols_to_swap[1])
        genotype.grammar[target_production_rule][item_index_2], genotype.grammar[target_production_rule][item_index_1] = \
            genotype.grammar[target_production_rule][item_index_1], genotype.grammar[target_production_rule][item_index_2]
#        genotype_logger.info(
#            f'mutation: swap in {genotype.id} for {target_production_rule} between {symbols_to_swap[0]} and {symbols_to_swap[1]}.')
    return genotype


def generate_symbol(genotype_conf):
    """
    Generates a symbol for addition

    :param genotype_conf: configuration for the genotype

    :return: symbol
    """
    symbol_category = random.randint(1, 2)
    # Modules
    if symbol_category == 1:
        alphabet = random.randint(1, len(Alphabet.modules()) - 1)
        symbol = lsystem.build_symbol(Alphabet.modules()[alphabet], genotype_conf)
    # Morphology moving commands
    elif symbol_category == 2:
        alphabet = random.randint(0, len(Alphabet.morphology_moving_commands()) - 1)
        symbol = lsystem.build_symbol(Alphabet.morphology_moving_commands()[alphabet], genotype_conf)
    else:
        raise Exception(
            'random number did not generate a number between 1 and 2. The value was: {}'.format(symbol_category))

    return symbol


def handle_addition(genotype, genotype_conf):
    """
    Adds symbol to genotype

    :param genotype: genotype to add to
    :param genotype_conf: configuration for the genotype

    :return: genotype
    """
    target_production_rule = random.choice(list(genotype.grammar))
    if target_production_rule == Alphabet.CORE_COMPONENT:
        if(len(genotype.grammar[target_production_rule]) > 1):
            addition_index = random.randint(1, len(genotype.grammar[target_production_rule]) - 1)
        else:
            return genotype
    else:
        addition_index = random.randint(0, len(genotype.grammar[target_production_rule]) - 1)
    symbol_to_add = generate_symbol(genotype_conf)
    genotype.grammar[target_production_rule].insert(addition_index, symbol_to_add)
    return genotype


def standard_mutation(genotype, mutation_conf):
    """
    Mutates genotype through addition/removal/swapping of symbols

    :param genotype: genotype to be mutated
    :param mutation_conf: configuration for mutation

    :return: modified genotype
    """
    new_genotype = genotype.clone()
    mutation_attempt = random.uniform(0.0, 1.0)
    if mutation_attempt > mutation_conf.mutation_prob:
        return new_genotype
    else:
        mutation_type = random.randint(1, 3)  # NTS: better way?
        if mutation_type == 1:
            modified_genotype = handle_deletion(new_genotype)
        elif mutation_type == 2:
            modified_genotype = handle_swap(new_genotype)
        elif mutation_type == 3:
            modified_genotype = handle_addition(new_genotype, mutation_conf.genotype_conf)
        else:
            raise Exception(
                'mutation_type value was not in the expected range (1,3). The value was: {}'.format(mutation_type))
        return modified_genotype
