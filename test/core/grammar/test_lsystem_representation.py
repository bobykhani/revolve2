import unittest

from nca.core.abstract.configurations import RepresentationConfiguration
from nca.core.genome.grammar.grammar import Grammar, ReplacementRules
from nca.core.genome.grammar.lsystem_representation import LSystemRepresentation
from test.core.grammar.test_alphabet import TestColorSymbol


class LSystemRepresentationTest(unittest.TestCase):

    def test_same(self):
        rules: ReplacementRules = {TestColorSymbol.GREEN: [[TestColorSymbol.RED]],
                                  TestColorSymbol.BLUE: [[TestColorSymbol.RED]]}

        representation = LSystemRepresentation(Grammar(TestColorSymbol.alphabet(), rules))
        representation.algorithm()

        outcome = [TestColorSymbol.RED for _ in range(RepresentationConfiguration().genome_size)]
        self.assertEqual(representation.genome, outcome)

    def test_array(self):
        rules: ReplacementRules = {TestColorSymbol.GREEN: [[TestColorSymbol.RED]],
                                  TestColorSymbol.BLUE: [[TestColorSymbol.RED]]}

        representation = LSystemRepresentation(Grammar(TestColorSymbol.alphabet(), rules))
        representation.algorithm()

        outcome = [TestColorSymbol.RED for _ in range(len(representation.genome))]
        self.assertEqual(representation.genome, outcome)
