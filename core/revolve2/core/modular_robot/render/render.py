import math

import cairo
from .canvas import Canvas
from .grid import Grid
from revolve2.core.modular_robot import Core, ActiveHinge, Brick, PassiveBone


class Render:

    def __init__(self):
        """Instantiate grid"""
        self.grid = Grid()

    FRONT = 0
    BACK = 3
    RIGHT = 2
    LEFT = 1

    def get_active_hinge_ids_ordered(self, module):
        """
        Traverse the robot structure and collect IDs of all Active Hinge modules.
        Sort them by their position in the grid from upper-left to bottom-right.
        @param module: The root module of the robot.
        @return: Sorted list of IDs of Active Hinge modules.
        """
        active_hinge_ids = []

        # Recursive helper function to traverse and collect data
        def traverse(module, x=0, y=0):
            if isinstance(module, ActiveHinge):
                grid_position = y * grid_width + x  # Calculate grid position
                active_hinge_ids.append((module.id, grid_position))

            if module.has_children():
                for core_slot, child_module in enumerate(module.children):
                    if child_module is not None:
                        new_x, new_y = self.update_coordinates(x, y, core_slot, module)
                        traverse(child_module, new_x, new_y)

        # After traversal
        active_hinge_ids.sort(key=lambda item: item[1])  # Sort based on grid position

        # Extract and return only the IDs and their grid positions
        return active_hinge_ids

        # Sort by y-coordinate first (for rows), then by x-coordinate (within a row)
        active_hinge_ids.sort(key=lambda item: (item[1][1], item[1][0]))

        # Extract and return only the IDs
        return [id for id, _ in active_hinge_ids]

    def update_coordinates(self, x, y, slot, parent_module):
        if slot == 0:  # assuming slot 0 means 'right'
            return x + 1, y
        elif slot == 1:  # assuming slot 1 means 'left'
            return x - 1, y
        elif slot == 2:  # assuming slot 2 means 'down'
            return x, y + 1
        elif slot == 3:  # assuming slot 3 means 'up'
            return x, y - 1
        else:
            return x, y  # Default case, if slot is not recognized

    def parse_body_to_draw(self, canvas, module, slot, parent_rotation):
        """
        Parse the body to the canvas to draw the png
        @param canvas: instance of the Canvas class
        @param module: body of the robot
        @param slot: parent slot of module
        """
        #TODO: map slots to enumerators

        if isinstance(module, Core):
            canvas.draw_controller(module.id)
        elif isinstance(module, ActiveHinge):
            canvas.move_by_slot(slot)
            absolute_rotation = (parent_rotation + module.rotation) % math.pi
            Canvas.rotating_orientation = absolute_rotation
            canvas.draw_hinge(module.id)
            canvas.draw_connector_to_parent()
        elif isinstance(module, PassiveBone):
            canvas.move_by_slot(slot)
            absolute_rotation = (parent_rotation + module.rotation) % math.pi
            Canvas.rotating_orientation = absolute_rotation
            canvas.draw_bone(module.id, module._size)
            canvas.draw_connector_to_parent()

        elif isinstance(module, Brick):
            canvas.move_by_slot(slot)
            absolute_rotation = (parent_rotation + module.rotation) % math.pi
            Canvas.rotating_orientation = absolute_rotation
            canvas.draw_module(module.id)
            canvas.draw_connector_to_parent()

        if module.has_children():
            # Traverse children of element to draw on canvas
            for core_slot, child_module in enumerate(module.children):
                if child_module is None:
                    continue
                self.parse_body_to_draw(canvas, child_module, core_slot, module.rotation)
            canvas.move_back()
        else:
            # Element has no children, move back to previous state
            canvas.move_back()

    def traverse_path_of_robot(self, module, slot, include_sensors=True):
        """
        Traverse path of robot to obtain visited coordinates
        @param module: body of the robot
        @param slot: attachment of parent slot
        @param include_sensors: add sensors to visisted_cooridnates if True
        """
        if isinstance(module, ActiveHinge) or isinstance(module, PassiveBone) or isinstance(module, Brick):
            self.grid.move_by_slot(slot)
            self.grid.add_to_visited(include_sensors, False)
        if module.has_children():
            # Traverse path of children of module
            for core_slot, child_module in enumerate(module.children):
                if child_module is None:
                    continue
                self.traverse_path_of_robot(child_module, core_slot, include_sensors)
            self.grid.move_back()
        else:
            # Element has no children, move back to previous state
            self.grid.move_back()

    def render_robot(self, body, image_path):
        """
        Render robot and save image file
        @param body: body of robot
        @param image_path: file path for saving image
        """
        # Calculate dimensions of drawing and core position
        self.traverse_path_of_robot(body, Render.FRONT)
        self.grid.calculate_grid_dimensions()
        core_position = self.grid.calculate_core_position()

        # Draw canvas
        cv = Canvas(self.grid.width, self.grid.height, 100)
        cv.set_position(core_position[0], core_position[1])

        # Draw body of robot
        self.parse_body_to_draw(cv, body, Render.FRONT, 0)

        # Draw sensors after, so that they don't get overdrawn
        #cv.draw_sensors()

        cv.save_png(image_path)

        # Reset variables to default values
        cv.reset_canvas()
        self.grid.reset_grid()

