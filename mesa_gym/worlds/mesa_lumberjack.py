# -*- coding: utf-8 -*-

import mesa
from enum import Enum

#######################
# physical entities
#######################

class Tree(mesa.Agent):

    def __init__(self, unique_id, model, strength=1):
        super().__init__(unique_id, model)
        self.strength = strength

    def get_state(self):
        return (self.strength)

    def destroy(self):
        self.model.remove_entity(self)

    def step(self):
        pass

    def show(self):
        return str(self.strength)

class Strength1Tree(Tree):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, strength=1)

class Strength2Tree(Tree):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, strength=2)

class Lumberjack(mesa.Agent):

    def __init__(self, unique_id, model, strength):
        super().__init__(unique_id, model)
        self.strength = strength
        self.mental_init()

    def get_state(self):
        return (self.strength, self.pos)

    @staticmethod
    def get_directions():
        directions = []
        for i in range(-1, 2):
            for j in range (-1, 2):
                directions.append((i, j))
        return directions

    def get_percepts(self):
        cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        percepts_about_trees = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        percepts_about_agents = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for pos in cells:
            absx, absy = self.pos
            x, y = pos; dx = x - absx; dy = y - absy
            if dx == self.model.width - 1: dx = -1
            elif dx == 1 - self.model.width: dx = 1
            if dy == self.model.height - 1: dy = -1
            elif dy == 1 - self.model.height: dy = 1
            dx += 1; dy += 1
            tree_total_strength = 0
            agent_total_strength = 0
            for item in self.model.grid.get_cell_list_contents([pos]):
                if isinstance(item, Tree):
                    tree_total_strength += item.strength
                elif isinstance(item, Lumberjack):
                    agent_total_strength += item.strength
            percepts_about_trees[dx*3 + dy] = tree_total_strength
            percepts_about_agents[dx*3 + dy] = agent_total_strength

        return  percepts_about_agents + percepts_about_trees

    def react(self):
        elems = self.model.grid.get_cell_list_contents([self.pos])
        for elem in elems:
            if elem != self:
                if elem != self:
                    if isinstance(elem, Tree):
                        if elem.strength <= self.strength:
                            self.model.events.append((self, elem))
                            self.trace("I've found a tree... I cut it!")
                            elem.destroy()
                        else:
                            self.model.events.append((self, False))
                            self.trace("I've found a tree... but that's too big for me!")

    def move(self, direction):
        dx, dy = direction
        absx, absy = self.pos
        x = absx + dx; y = absy + dy
        self.model.grid.move_agent(self, (x, y))

    def step(self):
        self.mental_step()
        self.react()

    def mental_init(self):
        self.next_action = None

    def mental_step(self):
        percepts = self.get_percepts()
        self.trace(f"percepts: {percepts}")
        if self.next_action is None:
            self.next_action = self.random.choice(self.get_directions())
        self.trace(f"next action: {self.next_action}")
        self.move(self.next_action)
        self.next_action = None

    def destroy(self):
        self.model.remove_entity(self)

    def trace(self, text):
        self.model.console.append(f"{type(self).__name__} {self.unique_id} > {text}".ljust(90, " "))

    def show(self):
        raise RuntimeError("Not defined symbol.")


#######################
# entities with minds
#######################

class StrongLumberjack(Lumberjack):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, strength=2)

    def show(self):
        return str(Symbol.STRONG_LUMBERJACK)

class WeakLumberjack(Lumberjack):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, strength=1)

    def show(self):
        return str(Symbol.WEAK_LUMBERJACK)


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

        self.ntrees = 0

    def step(self):
        self.events = []
        self.schedule.step()
        return self.end, self.events

    def add_entity(self, entity_type, x, y):
        entity = entity_type(len(self.entities), self)
        self.grid.place_agent(entity, (x, y))
        self.schedule.add(entity)
        self.entities.append(entity)

        # MOD for lumberjack
        if isinstance(entity, Tree):
            self.ntrees += 1

    def add_disability(self, entity_type, action, callable_for_value):
        if entity_type not in self.disabilities:
            self.disabilities[entity_type] = {}
        if action not in self.disabilities[entity_type]:
            self.disabilities[entity_type][action] = []
        self.disabilities[entity_type][action].append(callable_for_value)

    def remove_entity(self, entity):
        self.grid.remove_agent(entity)
        self.schedule.remove(entity)

        # MOD for lumberjack
        if isinstance(entity, Tree):
            self.ntrees -= 1
            if self.ntrees == 0:
                self.trace("All trees have been cut! End of game.")
                self.end = True

    def remove_disability(self, entity_type, action, callable_for_value):
        self.disabilities[entity_type][action].remove(callable_for_value)

    def trace(self, text):
        self.console.append(f">>>>>>> {text}".ljust(90, " "))


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

    def show(self, reversed=False):
        self.header()
        string = ""

        string += "|"
        for i in range(0, self.world.height):
            string += "-"
        string += "|\n"

        # for how mesa furnish the coordinate
        # we have to print the transpose of the world
        for cell in self.world.grid.coord_iter():
            cell_content, x, y = cell
            if y == 0:
                string += "|"
            if len(cell_content) > 0:
                found = False
                for entity in cell_content:
                    if isinstance(entity, Lumberjack):
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

        if reversed:
            console = reversed(self.world.console[-5:])
        else:
            console = self.world.console[-5:]
        for item in console:
            print(item)

        time.sleep(self.delay)


#######################
# helpers
#######################

# symbols for map stored as enum

class Symbol(Enum):
    STRONG_LUMBERJACK = 1
    WEAK_LUMBERJACK = 2
    TREE = 3

    def symbol_to_entity(symbol):
        if symbol == str(Symbol.STRONG_LUMBERJACK):
            return StrongLumberjack
        elif symbol == str(Symbol.WEAK_LUMBERJACK):
            return WeakLumberjack
        elif symbol == "1":
            return Strength1Tree
        elif symbol == "2":
            return Strength2Tree
        else:
            raise RuntimeError("Unknown symbol '%s'." % (symbol))

    def __str__(self):
        if self == Symbol.STRONG_LUMBERJACK:
            return "☻"
        elif self == Symbol.WEAK_LUMBERJACK:
            return "☺"
        else:
            raise RuntimeError("Unknown symbol '%s'." % self)

    def __repr__(self):
        return str(self)


def create_world(map):
    # remove trailing new line at the beginning
    if map[0] == "\n": map = map[1:]

    width = map.index("\n") - 2  # accounting for the borders
    if width == 0:
        raise ValueError("Unexpected dimensions of the map.")

    height = int(len(map) / (width + 3)) - 2  # accounting for the borders and newlines
    if height == 0 or len(map) % (width + 3) != 0:
        raise ValueError("Unexpected dimensions of the map.")

    # print(f"loading a {width}x{height} grid map...")
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
|-----|
|2  2 |
|1   2|
|  ☺  |
|1 ☻  |
| 1   |
|-----|
"""

    model = create_world(map)
    view = WorldView(model)
    view.init()

    MAX = 100
    n = 0
    while n < MAX:
        end, events = model.step()
        view.show()
        if end:
            break
        n += 1
    if n == MAX:
        print("Max number of steps reached. <<<<<<")