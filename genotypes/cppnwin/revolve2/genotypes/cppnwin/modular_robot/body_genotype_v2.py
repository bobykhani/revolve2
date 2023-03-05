import math
import multineat
import random
import operator
import sys

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module, PassiveBone
from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1


def random_v1(
        innov_db: multineat.InnovationDatabase,
        rng: multineat.RNG,
        multineat_params: multineat.Parameters,
        output_activation_func: multineat.ActivationFunction,
        num_initial_mutations: int,
        n_env_conditions: int,
        plastic_body: int,
) -> Genotype:
    if plastic_body != 0:
        return base_random_v1(
            innov_db,
            rng,
            multineat_params,
            output_activation_func,
            # 4,  # bias(always 1), pos_x, pos_y, inclined
            5,  # pos_x, pos_y, inclined
            6,  # brick, activehinge, rot0, rot90
            num_initial_mutations,
        )
    else:
        return base_random_v1(
            innov_db,
            rng,
            multineat_params,
            output_activation_func,
            # 3,  # bias(always 1), pos_x, pos_y
            5,  # pos_x, pos_y
            6,  # brick, activehinge, rot0, rot90
            num_initial_mutations,
        )


class Develop:

    def __init__(self, genotype, max_modules=15, substrate_radius=5, querying_seed=1000, env_condition=0, n_env_conditions=0,
                 plastic_body=0):

        self.max_modules = max_modules
        self.quantity_modules = 0
        self.substrate_radius = substrate_radius
        self.genotype = genotype
        self.querying_seed = querying_seed
        self.env_condition = env_condition
        self.n_env_conditions = n_env_conditions
        self.plastic_body = plastic_body
        self.development_seed = None
        self.random = None
        self.cppn = None
        # the queried substrate
        self.queried_substrate = {}
        self.phenotype_body = None
        self.parents_ids = []
        self.outputs_count = {
            'b_module': 0,
            'a_module': 0}

    def develop(self):

        self.random = random.Random(self.querying_seed)
        self.quantity_nodes = 0
        # the queried substrate
        self.queried_substrate = {}
        self.free_slots = {}
        self.outputs_count = {
            'b_module': 0,
            'a_module': 0}

        self.cppn = multineat.NeuralNetwork()
        self.genotype.genotype.BuildPhenotype(self.cppn)

        self.develop_body()
        self.phenotype_body.finalize()

        return self.phenotype_body, self.queried_substrate

    def develop_body(self):

        self.place_head()
        self.attach_body()
        return self.phenotype_body

    def calculate_coordinates(self, parent, slot):
        # calculate the actual 2d direction and coordinates of new module using relative-to-parent position as reference
        dic = {Core.FRONT: 0,
               Core.LEFT: 1,
               Core.BACK: 3,
               Core.RIGHT: 2}

        inverse_dic = {0: Core.FRONT,
                       1: Core.LEFT,
                       3: Core.BACK,
                       2: Core.RIGHT}

        direction = dic[parent.turtle_direction] + dic[slot]
        if direction >= len(dic):
            direction = direction - len(dic)

        turtle_direction = inverse_dic[direction]
        if turtle_direction == Core.LEFT:
            coordinates = (parent.substrate_coordinates[0] - 1,
                           parent.substrate_coordinates[1])
        if turtle_direction == Core.RIGHT:
            coordinates = (parent.substrate_coordinates[0] + 1,
                           parent.substrate_coordinates[1])
        if turtle_direction == Core.FRONT:
            coordinates = (parent.substrate_coordinates[0],
                           parent.substrate_coordinates[1] + 1)
        if turtle_direction == Core.BACK:
            coordinates = (parent.substrate_coordinates[0],
                           parent.substrate_coordinates[1] - 1)

        return coordinates, turtle_direction

    def choose_free_slot(self):
        parent_module_coor = self.random.choice(list(self.free_slots.keys()))
        parent_module = self.queried_substrate[parent_module_coor]
        direction = self.random.choice(list(self.free_slots[parent_module_coor]))

        return parent_module_coor, parent_module, direction

    def attach_body(self):
        # size of substrate is (substrate_radius*2+1)^2

        parent_module_coor = (0, 0)

        self.free_slots[parent_module_coor] = [Core.LEFT,
                                               Core.FRONT,
                                               Core.RIGHT,
                                               Core.BACK]

        parent_module_coor, parent_module, direction = self.choose_free_slot()

        for q in range(0, self.max_modules):

            # calculates coordinates of potential new module
            potential_module_coord, turtle_direction = self.calculate_coordinates(parent_module, direction)

            radius = self.substrate_radius

            # substrate limit
            if radius >= potential_module_coord[0] >= -radius and radius >= potential_module_coord[1] >= -radius:

                # queries potential new module given coordinates
                module_type, rotation, size = \
                    self.query_body_part(potential_module_coord[0], potential_module_coord[1])

                # if position in substrate is not already occupied
                if potential_module_coord not in self.queried_substrate.keys():

                    new_module = self.new_module(module_type, rotation, parent_module)

                    new_module.substrate_coordinates = potential_module_coord

                    new_module.turtle_direction = turtle_direction


                    # attaches module
                    if isinstance(new_module, PassiveBone):
                        new_module._size = size

                    parent_module.children[direction] = new_module
                    self.queried_substrate[potential_module_coord] = new_module

                    # joints branch out only to the front
                    if module_type is ActiveHinge:
                        directions = [ActiveHinge.ATTACHMENT]
                    elif module_type is PassiveBone:
                        directions = [PassiveBone.ATTACHMENT]
                    else:
                        directions = [Core.LEFT,
                                      Core.FRONT,
                                      Core.RIGHT]

                    self.free_slots[parent_module_coor].remove(direction)
                    if len(self.free_slots[parent_module_coor]) == 0:
                        self.free_slots.pop(parent_module_coor)

                    # adds new slots fo list of free slots
                    self.free_slots[potential_module_coord] = directions

                    # fins new free slot
                    parent_module_coor, parent_module, direction = self.choose_free_slot()

                # use this for not stopping ofter finding an intersection for the first time
                # else:
                # parent_module_coor = self.choose_free_slot()

    def place_head(self):

        module_type = Core
        self.phenotype_body = Body()
        self.phenotype_body.core.turtle_direction = Core.FRONT
        orientation = 0
        self.phenotype_body.core._id = self.quantity_modules
        self.phenotype_body.core._rotation = orientation * (math.pi / 2.0)
        self.phenotype_body.core._orientation = 0
        self.phenotype_body.core.rgb = self.get_color(module_type, orientation)
        self.phenotype_body.core.substrate_coordinates = (0, 0)
        self.queried_substrate[(0, 0)] = self.phenotype_body.core

    def new_module(self, module_type, orientation, parent_module):

        # calculates _absolute_rotation
        absolute_rotation = 0
        if (module_type == ActiveHinge or module_type == PassiveBone) and orientation == 1:
            if (type(parent_module) == ActiveHinge or type(parent_module) == PassiveBone) and parent_module._rotation == 1:
                _rotation = 0
            else:
                _rotation = 1
        else:
            if (type(parent_module) == ActiveHinge or type(parent_module) == PassiveBone) and parent_module._rotation == 1:
                _rotation = 1

        # makes sure it wont rotate bricks, so to prevent 3d shapes
        if module_type == Brick and (type(parent_module) == ActiveHinge or type(parent_module) == PassiveBone) and parent_module._rotation == 1:
            # inverts it no absolute rotation
            orientation = 1

        if module_type == PassiveBone:
            module = module_type(orientation * (math.pi / 2.0), size=0.1)
        else:
            module = module_type(orientation * (math.pi / 2.0))
        self.quantity_modules += 1
        module._id = str(self.quantity_modules)
        module._rotation = absolute_rotation
        module.rgb = self.get_color(module_type, orientation)
        return module

    def query_body_part(self, x_dest, y_dest):

        # # Applies regulation according to environmental conditions.
        if self.plastic_body == 0:
            # self.cppn.Input(  [1.0, x_dest, y_dest]  )  # 1.0 is the bias input
            self.cppn.Input([x_dest, y_dest])
        else:

            staticfriction, dynamicfriction, yrotationdegrees, platform, toxic = \
                float(self.env_condition[0]), \
                float(self.env_condition[1]), \
                float(self.env_condition[2]), \
                float(self.env_condition[3]), \
                float(self.env_condition[4])

            # TODO: make conditions-checking dynamic

            # if inclined
            # if yrotationdegrees > 0:
            #     inclined = -1
            # else:
            #     inclined = 1

            # obsolete name: toxic here means just a change in task
            if toxic > 0:
                toxicenv = 1
            else:
                toxicenv = -1

            self.cppn.Input(
                # [x_dest, y_dest, inclined]
                [x_dest, y_dest, toxicenv]
                #   [1.0, x_dest, y_dest, inclined]  # 1.0 is the bias input
            )

        self.cppn.ActivateAllLayers()
        outputs = self.cppn.Output()

        # get module type from output probabilities
        type_probs = [outputs[0], outputs[1],outputs[2]]
        types = [Brick, ActiveHinge, PassiveBone]
        module_type = types[type_probs.index(max(type_probs))]

        # get rotation from output probabilities
        if module_type is ActiveHinge:
            rotation_probs = [outputs[3]]
            rotation = rotation_probs.index(max(rotation_probs))
        else:
            rotation = 0

        Y = ((outputs[4] - (-1)) / (1 - (-1))) * (0.2 - 0.1) + 0.1

        size = Y

        return module_type, rotation, size

    def get_color(self, module_type, rotation):
        rgb = []
        if module_type == Brick:
            rgb = [0, 0, 1]
        if module_type == ActiveHinge:
            if rotation == 0:
                rgb = [1, 0.08, 0.58]
            else:
                rgb = [0.7, 0, 0]
        if module_type == Core:
            rgb = [1, 1, 0]
        return rgb
