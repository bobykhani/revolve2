import unittest

from nca.core.abstract.structural.tree.tree import Tree2D
from nca.core.abstract.structural.tree.tree_helper import Orientation


class TreeRegistryTest(unittest.TestCase):

    def test_overlap(self):

        root = Tree2D()
        child01 = root.add(Orientation.LEFT)
        child10 = child01.append(Orientation.DOWN)
        child11 = child10.append(Orientation.RIGHT)

        try:
            root.add(Orientation.DOWN)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)

    def test_node(self):

        root = Tree2D()
        child01 = root.add(Orientation.LEFT)
        child10 = child01.append(Orientation.DOWN)
        child11 = child10.append(Orientation.RIGHT)

        try:
            root.add(Orientation.TOP)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)