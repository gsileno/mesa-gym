# -*- coding: utf-8 -*-

import mesa
from enum import Enum


#######################
# symbols for map
#######################

class Symbol(Enum):
    ALICE = 1
    BOB = 2
    LION = 3
    WALL = 4
    SOMEGRASS = 5
    GRASS = 6
    MOREGRASS = 7
    DIAMOND = 8

    def symbol_to_entity(symbol):
        if symbol == str(Symbol.ALICE):
            return RangerAgent
        elif symbol == str(Symbol.BOB):
            return RangerAgent
        elif symbol == str(Symbol.LION):
            return LionAgent
        elif symbol == str(Symbol.WALL):
            return Wall
        elif symbol == str(Symbol.SOMEGRASS):
            return Grass
        elif symbol == str(Symbol.GRASS):
            return Grass
        elif symbol == str(Symbol.MOREGRASS):
            return Grass
        elif symbol == str(Symbol.DIAMOND):
            return Diamond
        else:
            raise RuntimeError("Unknown symbol '%s'." % (symbol))

    def __str__(self):
        if self == Symbol.ALICE:
            return "☻"
        elif self == Symbol.BOB:
            return "☺"
        elif self == Symbol.LION:
            return "Ω"
        elif self == Symbol.WALL:
            return "█"
        elif self == Symbol.SOMEGRASS:
            return "░"
        elif self == Symbol.GRASS:
            return "▒"
        elif self == Symbol.MOREGRASS:
            return "▓"
        elif self == Symbol.DIAMOND:
            return "♦"
        else:
            raise RuntimeError("Unknown symbol '%s'." % self)

    def __repr__(self):
        return str(self)


#######################
# physical entities
#######################

class Wall(mesa.Agent):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        self.model.add_disability(mesa.Agent, "move", self.get_pos)

    def get_pos(self):
        return self.pos

    def step(self):
        pass

    def show(self):
        return str(Symbol.WALL)


class Grass(mesa.Agent):

    def __init__(self, unique_id, model, position, amount=1):
        super().__init__(unique_id, model)
        self.amount = amount

    def get_state(self):
        return self.amount

    def step(self):
        if self.amount <= 30:
            self.amount += 1

    def show(self):
        if self.amount == 0:
            return " "
        elif self.amount > 20:
            return str(Symbol.SOMEGRASS)
        elif self.amount > 10:
            return str(Symbol.GRASS)
        elif self.amount > 0:
            return str(Symbol.MOREGRASS)


class Diamond(mesa.Agent):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)

    def step(self):
        pass

    def show(self):
        return str(Symbol.DIAMOND)

    def destroy(self):
        self.model.remove_entity(self)


class AgentBody(mesa.Agent):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        self.energy = 999
        self.mental_init()

    def get_pos(self):
        return self.pos

    def get_state(self):
        return self.energy

    @staticmethod
    def get_directions():
        directions = []
        for i in range(-1, 2):
            for j in range (-1, 2):
                directions.append((i, j))
        return directions

    def get_percepts(self):
        cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        percepts = {}
        for pos in cells:
            absx, absy = self.pos
            x, y = pos
            dx = x - absx
            dy = y - absy
            if dx == self.model.width - 1:
                dx = -1
            elif dx == 1 - self.model.width:
                dx = 1
            if dy == self.model.height - 1:
                dy = -1
            elif dy == 1 - self.model.height:
                dy = 1
            percepts[(dx, dy)] = self.model.grid.get_cell_list_contents([pos])
        return percepts

    def react(self):
        elems = self.model.grid.get_cell_list_contents([self.pos])
        for elem in elems:
            if elem != self:
                if type(elem) == Grass:
                    elem.amount -= 5
                    if elem.amount < 0:
                        elem.amount = 0
                elif type(elem) == Diamond:
                    if type(self) is RangerAgent:
                        self.model.events.append((self, elem))
                        self.trace("I've found the diamond!")
                        elem.destroy()
                        self.model.end = True
                elif type(elem) == RangerAgent:
                    if type(self) is LionAgent:
                        self.model.events.append((self, elem))
                        self.trace("I've found a ranger... GNAM!")
                        elem.destroy()
                        self.model.end = True

    def move(self, direction):
        dx, dy = direction
        absx, absy = self.pos
        x = absx + dx
        y = absy + dy
        disabilities = self.model.disabilities[mesa.Agent]["move"]
        if (x, y) not in [disabled() for disabled in disabilities]:
            self.model.grid.move_agent(self, (x, y))
        else:
            self.model.events.append((self, False))
            self.trace("action 'move' failed")
        self.energy -= 1

    def step(self):
        if self.energy == 0:
            self.model.end = True
            return
        else:
            self.trace(f"energy: {self.energy}")
            self.mental_step()

        self.react()

    def trace(self, text):
        self.model.console.append(f"entity {self.unique_id} > {text}                                    ")

    def destroy(self):
        pass

    def mental_init(self):
        self.next_action = None

    def mental_step(self):
        # percepts = self.get_percepts()
        if self.next_action is None:
            self.next_action = self.random.choice(self.get_directions())
        self.trace(f"next action {self.next_action}")
        self.move(self.next_action)
        self.next_action = None

    def show(self):
        raise RuntimeError("Not defined symbol.")


#######################
# entities with minds
#######################

class RangerAgent(AgentBody):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model, position)
        self.model.add_disability(RangerAgent, "move", self.get_pos)

    def show(self):
        return str(Symbol.ALICE)

    def destroy(self):
        self.model.remove_disability(RangerAgent, "move", self.get_pos)
        self.model.remove_entity(self)


class LionAgent(AgentBody):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model, position)
        self.model.add_disability(LionAgent, "move", self.get_pos)

    def show(self):
        return str(Symbol.LION)

    def destroy(self):
        self.model.remove_disability(LionAgent, "move", self.get_pos)
        self.model.remove_entity(self)

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

    def show(self):
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

        for item in reversed(self.world.console[-5:]):
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
            if 1 < z % (width + 3) < width + 1:
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
|  ██████            |
|  █             ☺   |
|  █                 |
|  ██████            |
|  ░░░░░█    ████████|
|  ░░░░░█         Ω  |
|  ██████            |
|              ♦     |
|                    |
|--------------------|
"""

    model = create_world(map)
    view = WorldView(model)
    view.init()

    while True:
        end, events = model.step()
        view.show()
        if end:
            break