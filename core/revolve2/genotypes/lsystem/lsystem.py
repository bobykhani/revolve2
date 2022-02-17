# no mother classes have been defined yet! not sure how to separate the the filed in folders...
from abc import ABC
from enum import Enum
from typing import cast, List, Tuple

import core.revolve2.core.modular_robot
from ._genotype import BodybrainGenotype as Genotype
import random
import math
import copy
import itertools
from revolve2.core.modular_robot import ModularRobot, Body, Brick, ActiveHinge
from random import Random
from revolve2.core.modular_robot.brains import CpgRandom

from ...core.database import StaticData
from revolve2.core.database.serialization import Serializable, SerializeError


class Alphabet(Enum):

    # Modules
    CORE_COMPONENT = "C"
    JOINT_HORIZONTAL = "AJ1"
    JOINT_VERTICAL = "AJ2"
    BLOCK = "B"

    # MorphologyMovingCommands
    MOVE_RIGHT = "mover"
    MOVE_FRONT = "movef"
    MOVE_LEFT = "movel"
    MOVE_BACK = "moveb"

    @staticmethod
    def modules():
        return [
            [Alphabet.CORE_COMPONENT, []],
            [Alphabet.JOINT_HORIZONTAL, []],
            [Alphabet.JOINT_VERTICAL, []],
            [Alphabet.BLOCK, []],
        ]

    @staticmethod
    def morphology_moving_commands():
        return [
            [Alphabet.MOVE_RIGHT, []],
            [Alphabet.MOVE_FRONT, []],
            [Alphabet.MOVE_LEFT, []],
            [Alphabet.MOVE_BACK, []],
        ]


class lsystem(Genotype, Serializable, ABC):
    """
    L-system genotypic representation, enhanced with epigenetic capabilities for phenotypic plasticity, through Genetic Programming.
    """

    def __init__(self, conf, robot_id):
        """
        :param conf: configurations for lsystem
        :param robot_id: unique id of the robot
        :type conf: PlasticodingConfig
        """
        self.conf = conf
        self.id = str(robot_id)
        self.grammar = {}

        # Auxiliary variables
        self.substrate_coordinates_all = {(0, 0): "1"}
        self.valid = False
        self.intermediate_phenotype = None
        self.phenotype = None
        self.morph_mounting_container = None
        self.mounting_reference = None
        self.mounting_reference_stack = []
        self.quantity_modules = 1
        self.quantity_nodes = 0
        self.index_symbol = 0
        self.index_params = 1
        self.inputs_stack = []
        self.outputs_stack = []
        self.edges = {}
        self.reference = None
        self.direction = None

    def clone(self):
        return copy.deepcopy(self)

    def load_genotype(self, genotype_file):
        with open(genotype_file) as f:
            lines = f.readlines()

        for line in lines:
            line_array = line.split(" ")
            repleceable_symbol = Alphabet(line_array[0])
            self.grammar[repleceable_symbol] = []
            rule = line_array[1 : len(line_array) - 1]
            for symbol_array in rule:
                symbol_array = symbol_array.split("_")
                symbol = Alphabet(symbol_array[0])
                if len(symbol_array) > 1:
                    params = symbol_array[1].split("|")
                else:
                    params = []
                self.grammar[repleceable_symbol].append([symbol, params])

    def develop(self):
        self.early_development()
        phenotype = self.late_development()
        return phenotype

    def early_development(self):

        self.intermediate_phenotype = [[self.conf.axiom_w, []]]

        for i in range(0, self.conf.i_iterations):

            position = 0
            for aux_index in range(0, len(self.intermediate_phenotype)):

                symbol = self.intermediate_phenotype[position]
                if [symbol[self.index_symbol], []] in Alphabet.modules():
                    # removes symbol
                    self.intermediate_phenotype.pop(position)
                    # replaces by its production rule
                    for ii in range(0, len(self.grammar[symbol[self.index_symbol]])):
                        self.intermediate_phenotype.insert(
                            position + ii, self.grammar[symbol[self.index_symbol]][ii]
                        )
                    position = position + ii + 1
                else:
                    position = position + 1
        # logger.info('Robot ' + str(self.id) + ' was early-developed.')

    def late_development(self):

        rng = Random()
        rng.seed(6)

        self.phenotype = Body()

        self.intermediate_phenotype
        for symbol in self.intermediate_phenotype:
            if symbol[self.index_symbol] == Alphabet.CORE_COMPONENT:
                self.phenotype = Body()
                self.reference = self.phenotype.core

            if symbol[self.index_symbol] == Alphabet.BLOCK:
                if self.direction is not None:
                    if type(self.reference) == ActiveHinge:
                        self.reference.attachment = Brick(0.0)
                        self.reference = self.reference.attachment
                    else:
                        if self.direction == "Front" and self.reference.front is None:
                            self.reference.front = Brick(0.0)
                            self.reference = self.reference.front
                        if self.direction == "Right" and self.reference.right is None:
                            self.reference.right = Brick(0.0)
                            self.reference = self.reference.right
                        if self.direction == "Left" and self.reference.left is None:
                            self.reference.left = Brick(0.0)
                            self.reference = self.reference.left
                        if type(self.reference) != Brick:
                            if self.direction == "Back" and self.reference.back is None:
                                self.reference.back = Brick(0.0)
                                self.reference = self.reference.back

            if symbol[self.index_symbol] == Alphabet.JOINT_HORIZONTAL:
                if self.direction is not None:
                    if type(self.reference) == ActiveHinge:
                        self.reference.attachment = ActiveHinge(math.pi / 2.0)
                        self.reference = self.reference.attachment
                    else:
                        if self.direction == "Front" and self.reference.front is None:
                            self.reference.front = ActiveHinge(math.pi / 2.0)
                            self.reference = self.reference.front
                        if self.direction == "Right" and self.reference.right is None:
                            self.reference.right = ActiveHinge(math.pi / 2.0)
                            self.reference = self.reference.right
                        if self.direction == "Left" and self.reference.left is None:
                            self.reference.left = ActiveHinge(math.pi / 2.0)
                            self.reference = self.reference.left
                        if type(self.reference) != Brick:
                            if self.direction == "Back" and self.reference.back is None:
                                self.reference.back = ActiveHinge(math.pi / 2.0)
                                self.reference = self.reference.back

            if [symbol[self.index_symbol], []] in Alphabet.morphology_moving_commands():
                self.move_in_body(symbol)

        brain = CpgRandom(rng)
        robot = ModularRobot(self.phenotype, brain)
        return robot

    def move_in_body(self, symbol):
        if symbol[0] == Alphabet.MOVE_FRONT:
            self.direction = "Front"
        if symbol[0] == Alphabet.MOVE_RIGHT:
            self.direction = "Right"
        if symbol[0] == Alphabet.MOVE_LEFT:
            self.direction = "Left"
        if symbol[0] == Alphabet.MOVE_BACK:
            self.direction = "Back"

    @staticmethod
    def build_symbol(symbol, conf):
        """
        Adds params for alphabet symbols (when it applies).
        :return:
        """
        index_symbol = 0
        index_params = 1

        if (
            symbol[index_symbol] is Alphabet.JOINT_HORIZONTAL
            or symbol[index_symbol] is Alphabet.JOINT_VERTICAL
        ):

            symbol[index_params] = [
                random.uniform(conf.weight_min, conf.weight_max),
                random.uniform(conf.oscillator_param_min, conf.oscillator_param_max),
                random.uniform(conf.oscillator_param_min, conf.oscillator_param_max),
                random.uniform(conf.oscillator_param_min, conf.oscillator_param_max),
            ]

        return symbol

    def serialize(self) -> StaticData:
        pass

    @classmethod
    def deserialize(cls, data: StaticData) -> Serializable:
        pass


from .initialization import random_initialization

#
#
class LsystemConfig:
    def __init__(
        self,
        initialization_genome=random_initialization,
        e_max_groups=3,
        oscillator_param_min=1,
        oscillator_param_max=10,
        weight_param_min=-1,
        weight_param_max=1,
        weight_min=-1,
        weight_max=1,
        axiom_w=Alphabet.CORE_COMPONENT,
        i_iterations=3,
        max_structural_modules=15,
        robot_id=0,
    ):
        self.initialization_genome = initialization_genome
        self.e_max_groups = e_max_groups
        self.oscillator_param_min = oscillator_param_min
        self.oscillator_param_max = oscillator_param_max
        self.weight_param_min = weight_param_min
        self.weight_param_max = weight_param_max
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.axiom_w = axiom_w
        self.i_iterations = i_iterations
        self.max_structural_modules = max_structural_modules
        self.robot_id = robot_id
