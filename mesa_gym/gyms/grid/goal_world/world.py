# -*- coding: utf-8 -*-

import mesa
from enum import Enum


#######################
# symbols for map
#######################

class Symbol(Enum):
    MOUSE = 1
    CHEESE = 2

    def symbol_to_entity(symbol):
        if symbol == str(Symbol.MOUSE):
            return Mouse
        elif symbol == str(Symbol.CHEESE):
            return Cheese
        else:
            raise RuntimeError("Unknown symbol '%s'." % (symbol))

    def __str__(self):
        if self == Symbol.MOUSE:
            return "☺"
        elif self == Symbol.CHEESE:
            return "♠"
        else:
            raise RuntimeError("Unknown symbol '%s'." % self)

    def __repr__(self):
        return str(self)


#######################
# physical entities
#######################


class Cheese(mesa.Agent):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)

    def step(self):
        pass

    def show(self):
        return str(Symbol.CHEESE)

    def destroy(self):
        self.model.remove_entity(self)


class AgentBody(mesa.Agent):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        self.mental_init()

    @staticmethod
    def get_directions():
        directions = []
        for i in range(-1, 2):
            for j in range (-1, 2):
                directions.append((i, j))
        return directions

    def get_percepts(self, radius=1):
        if self.pos is None: # case in which the agent has been destroyed # TODO, get percepts should not be called in this case!
            return []

        relevant_entities = self._get_relevant_entities()

        if relevant_entities is None:
            raise RuntimeWarning("An agent has been initialized not paying attention to any entity.")

        percepts_about = {}
        for entity_type in relevant_entities:
            percepts_about[entity_type] = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                absx, absy = self.pos
                x = (absx + dx) % self.model.width
                y = (absy + dy) % self.model.height

                strength_concerning = {}
                for entity_type in relevant_entities:
                    strength_concerning[entity_type] = 0

                for item in self.model.grid.get_cell_list_contents((x, y)):
                    entity_type = type(item)
                    if entity_type in relevant_entities:
                        strength_concerning[entity_type] += 1

                for entity_type in relevant_entities:
                    percepts_about[entity_type].append(strength_concerning[entity_type])

        percepts = []
        for entity_type in relevant_entities:
            percepts += percepts_about[entity_type]

        return percepts

    def react(self):
        elems = self.model.grid.get_cell_list_contents([self.pos])
        for elem in elems:
            if elem != self:
                pass

    def move(self, direction):
        dx, dy = direction
        absx, absy = self.pos
        x = absx + dx
        y = absy + dy
        if mesa.Agent in self.model.disabilities:
            disabilities = self.model.disabilities[mesa.Agent]["move"]
        else:
            disabilities = []
        if (x, y) not in [disabled() for disabled in disabilities]:
            self.model.grid.move_agent(self, (x, y))
        else:
            self.model.events.append((self, False))
            self.trace("action 'move' failed")

    def step(self):
        self.mental_step()
        self.react()

    def trace(self, text):
        self.model.console.append(f"{type(self).__name__} {self.unique_id} > {text}                                    ")

    def destroy(self):
        pass

    def mental_init(self):
        self.next_action = None

    def _get_relevant_entities(self):
        return None

    def mental_step(self):
        if self.next_action is None:
            self.next_action = self.random.choice(self.get_directions())
        self.trace(f"next action: {self.next_action}")
        self.move(self.next_action)
        self.next_action = None

    def show(self):
        raise RuntimeError("Not defined symbol.")


#######################
# entities with minds
#######################

class Mouse(AgentBody):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model, position)

    def _get_relevant_entities(self):
        return [Cheese]

    def show(self):
        return str(Symbol.MOUSE)

    def destroy(self):
        self.model.remove_entity(self)

    def react(self):
        elems = self.model.grid.get_cell_list_contents([self.pos])
        for elem in elems:
            if elem != self:
                if type(elem) == Cheese:
                    self.model.events.append((self, elem))
                    self.trace("I've found the cheese!")
                    elem.destroy()
                    self.model.end = True

#######################
# world
#######################

class WorldModel(mesa.Model):

    def __init__(self, width, height):
        self.entities = []
        self.disabilities = {}
        self.width = width
        self.height = height
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.end = False
        self.console = []

    def step(self):
        self.events = []
        self.schedule.step()
        return self.end, self.events

    def get_positions(self):
        positions = {}
        size = self.width * self.height
        for entity in self.entities:
            entity_type = str(type(entity))
            if entity_type not in positions:
                positions[entity_type] = []
                for _ in range(size):
                    positions[entity_type].append(0)

            if entity.pos is not None:
                x, y = entity.pos
                positions[entity_type][y * self.width + x] += 1
                # print(f"position of {entity_type} is {(x, y)}")

        flat_positions = []
        for entity_type in sorted([str(k) for k in positions.keys()]):
            flat_positions += positions[entity_type]

        return flat_positions

    def add_entity(self, entity_type, x, y):
        entity = entity_type(len(self.entities), self, (x, y))
        self.grid.place_agent(entity, (x, y))
        self.schedule.add(entity)
        self.entities.append(entity)

    def add_disability(self, entity_type, action, callable_for_value):
        if entity_type not in self.disabilities:
            self.disabilities[entity_type] = {}
        if action not in self.disabilities[entity_type]:
            self.disabilities[entity_type][action] = []
        self.disabilities[entity_type][action].append(callable_for_value)

    def remove_entity(self, entity):
        self.grid.remove_agent(entity)
        self.schedule.remove(entity)

    def remove_disability(self, entity_type, action, callable_for_value):
        self.disabilities[entity_type][action].remove(callable_for_value)


#######################
# viewer
#######################

import time


def move(x, y):
    print("\033[%d;%dH" % (y, x))


class WorldView:
    def __init__(self, world_model, fps=25):
        self.world = world_model

        # from frame per second (fps) toseconds per frame (ms)
        # eg. 25 f/s = 1/25 s/f = 1000/25 ms/f = 40 ms/f
        self.delay = 1/fps

    def init(self):
        print('\x1b[2J')

    def header(self):
        move(0, 0)
        print("mesagym -- minimal gym built on top on mesa\n")

    def show(self, reverse_order=False):
        self.header()
        string = ""

        string += "|"
        for i in range(0, self.world.height):
            string += "-"
        string += "|\n"

        # for how mesa furnish the coordinate
        # we have to print the transpose of the world
        for cell in self.world.grid.coord_iter():
            cell_content, (x, y) = cell

            if y == 0:
                string += "|"
            if len(cell_content) > 0:
                found = False
                for entity in cell_content:
                    if isinstance(entity, AgentBody):
                        string += entity.show()
                        found = True
                        break
                if not found:
                    string += cell_content[0].show()
            else:
                string += " "
            if y == self.world.height - 1:
                string += "|\n"

        string += "|"
        for i in range(0, self.world.height):
            string += "-"
        string += "|\n"

        print(string)

        print(">>> console <<<")

        if reverse_order:
            console = reversed(self.world.console[-5:])
        else:
            console = self.world.console[-5:]
        for item in console:
            print(item)

        time.sleep(self.delay)


#######################
# helpers
#######################

def create_world(map):
    # remove trailing new line at the beginning
    if map[0] == "\n": map = map[1:]

    width = map.index("\n") - 2  # accounting for the borders
    if width == 0:
        raise ValueError("Unexpected dimensions of the map.")

    height = int(len(map) / (width + 3)) - 2  # accounting for the borders and newlines
    if height == 0 or len(map) % (width + 3) != 0:
        raise ValueError("Unexpected dimensions of the map.")

    # print(f"loading map on a {width}x{height} grid...")

    model = WorldModel(height, width)

    x = 0
    y = 0
    for z, ch in enumerate(map):
        if width + 3 < z < len(map) - width - 3:
            if 0 < z % (width + 3) <= width:
                if ch == " ":
                    pass
                else:
                    entity_type = Symbol.symbol_to_entity(ch)
                    model.add_entity(entity_type, x, y)
                y += 1
            if z % (width + 3) == 0:
                x += 1
                y = 0

    return model


#######################
# main
#######################

if __name__ == "__main__":

    map = """
|--------------------|
|                    |
|                    |
|                ☺   |
|                    |
|                    |
|                    |
|                    |
|                    |
|              ♠     |
|                    |
|--------------------|
"""

#     map = """
# |---|
# |  ☺|
# |   |
# | ♠ |
# |   |
# |---|
# """

    model = create_world(map)
    view = WorldView(model)
    view.init()
    view.show()

    while True:
        end, events = model.step()
        view.show()
        if end:
            break