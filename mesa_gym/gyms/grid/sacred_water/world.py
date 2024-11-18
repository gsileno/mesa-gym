# -*- coding: utf-8 -*-
import random

from common import *

#######################
# symbols for map
#######################

class Symbol(GenericSymbol):
    GATHERER = 1
    FRUIT = 2
    CLEAN_WATER = 3
    POISONED_WATER = 4
    VERY_POISONED_WATER = 5

    def symbol_to_entity(symbol):
        if symbol == str(Symbol.GATHERER):
            return Gatherer
        elif symbol == str(Symbol.FRUIT):
            return Fruit
        elif symbol == str(Symbol.CLEAN_WATER):
            return Water
        elif symbol == str(Symbol.POISONED_WATER):
            return Water(poison=10)
        elif symbol == str(Symbol.VERY_POISONED_WATER):
            return Water(poison=20)
        else:
            raise RuntimeError("Unknown symbol '%s'." % (symbol))

    def __str__(self):
        if self == Symbol.GATHERER:
            return "☺"
        elif self == Symbol.FRUIT:
            return "σ"
        elif self == Symbol.CLEAN_WATER:
            return "░"
        elif self == Symbol.POISONED_WATER:
            return "▒"
        elif self == Symbol.VERY_POISONED_WATER:
            return "▓"
        else:
            raise RuntimeError("Unknown symbol '%s'." % self)




#######################
# physical entities
#######################

class Fruit(mesa.Agent):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)

    def step(self):
        fruits = 1
        water = 0
        poison = 0
        empty_cells = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx != 0 or dy != 0:
                    if self.pos is not None:
                        absx, absy = self.pos
                        x = (absx + dx) % self.model.width
                        y = (absy + dy) % self.model.height

                        items = self.model.grid.get_cell_list_contents((x, y))
                        if len(items) == 0:
                            empty_cells.append((x, y))
                        for item in items:
                            if type(item) == Fruit:
                                fruits += 1
                            elif type(item) == Water:
                                water +=  1
                                poison += item.poison
                    # else:
                    # TOCHECK: THIS IS A BUG OCCURRING NOW


        # almost arbitrary formula to govern the probability of new fruits to be generated
        poison = round(poison/5)
        prob_new_fruit = (max(water - poison, 0)) * (fruits ** 2) / (250 * (4 ** 2))

        # print(f"Probability of growth: {prob_new_fruit}")
        for cell in empty_cells:
            if random.random() <= prob_new_fruit:
                self.model.add_entity(Fruit, cell[0], cell[1])

    def show(self):
        return str(Symbol.FRUIT)

    def destroy(self):
        self.model.remove_entity(self)


class Water(mesa.Agent):

    def __init__(self, unique_id, model, position, poison=0):
        super().__init__(unique_id, model)
        self.poison = poison

    def get_state(self):
        return self.poison

    def step(self):
        if self.poison > 0:
            self.poison -= 0.1

    def show(self):
        if self.poison < 5:
            return str(Symbol.CLEAN_WATER)
        elif self.poison < 15:
            return str(Symbol.POISONED_WATER)
        else:
            return str(Symbol.VERY_POISONED_WATER)


#######################
# entities with minds
#######################

class Gatherer(AgentBody):

    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model, position)
        self.water = 0
        self.food = 0

    def _get_relevant_entities(self):
        return [Fruit]

    def show(self):
        return str(Symbol.GATHERER)

    def destroy(self):
        self.model.remove_entity(self)

    def react(self):
        self.water -= 0.1
        self.food -= 0.1
        elems = self.model.grid.get_cell_list_contents([self.pos])
        for elem in elems:
            if elem != self:
                if type(elem) == Fruit:
                    self.food += 1
                    self.model.events.append((self, elem))
                    self.trace("I've found a fruit!")
                    elem.destroy()
                elif type(elem) == Water:
                    self.water += 1
                    elem.poison += 10
                    self.model.events.append((self, elem))
                    self.trace("I've found water!")

        if self.water <= 0:
            self.trace("I'm thirsty")
            if self.water < -5:
                self.model.events.append((self, (Water, None)))
                self.model.end = True
                self.trace("I am too thirsty (end session).")
        if self.food <= 0:
            self.trace("I'm hungry")
            if self.food < -5:
                self.model.events.append((self, (Fruit, None)))
                self.model.end = True
                self.trace("I am too hungry (end session).")

#######################
# main
#######################

# default_map = """
# |--------------------|
# |        σ           |
# |                    |
# | ░░░             ☺  |
# |  ░░░               |
# |          σ     σ   |
# | σ                  |
# |                 σ  |
# |                    |
# |     σσ░░░░         |
# |    σ░░░░░░░░       |
# |--------------------|
# """

default_map = """
|--------------------|
|        σ           |
|                    |
| ░               ☺  |
|                    |
|                    |
|                    |
|                    |
|                    |
|                    |
|    σ░              |
|--------------------|
"""

if __name__ == "__main__":


    model = create_world(default_map, symbol_type=Symbol)
    view = WorldView(model, name="sacred water")
    view.init()
    view.show()

    while True:
        end, events = model.step()
        view.show()
        if end:
            break