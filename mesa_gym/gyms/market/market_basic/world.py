# TODO: not yet complete

import mesa
from enum import Enum

###########################
# language between agents
###########################

class Message:

    def __init__(self, agent, action, refinement):
        self.agent = agent
        self.action = action
        self.refinement = refinement 


#######################
# physical entities
#######################


class OrderBook:

    def __init__(self):
        self.offers = {}
        self.demands = {}
        self.deals = []
        self.messages = []
        self.time = 0

    def add_offer(self, seller, price):
        self.offers[seller] = price # previous offers are overridden

    def get_offers(self):
        public_offers = []
        for seller in self.offers.keys():
            public_offers.append(Message(seller, "sell", self.offers[seller]))
        return public_offers

    def remove_seller(self, seller):
        self.offers.pop(seller, None)

    def add_demand(self, buyer, price):
        self.demands[buyer] = price # previous demands are overridden

    def remove_buyer(self, buyer):
        self.offers.pop(buyer, None)

    def add_deal(self, agent, proposal):        
        self.deals.append((agent, proposal, self.time))

    def acknowledge_offeror(self, offeree, proposal):
        offeror = proposal.agent
        offeror.messages.append(Message(offeree, "accepted", proposal))

    def step(self):
        for msg in self.messages:
            if msg.action == "buy":
                self.add_demand(msg.agent, msg.refinement)
            elif msg.action == "sell":
                self.add_offer(msg.agent, msg.refinement)
            elif msg.action == "accept":
                self.add_deal(msg.agent, msg.refinement)
                self.acknowledge_offeror(msg.agent, msg.refinement)

        self.time += 1


class AgentBody(mesa.Agent):

    def __init__(self, unique_id, model, params=None):
        super().__init__(unique_id, model)
        if params is not None and "capital" in params:
            self.capital = params["capital"]
        else:
            self.capital = 0
        if params is not None and "assets" in params:
            self.assets = params["assets"]
        else:
            self.assets = 0

        self.messages = []
        self.mental_init()

    def buy(self, amount):
        self.model.orderbook.messages.append(Message(self, "buy", amount))

    def sell(self, amount):
        self.model.orderbook.messages.append(Message(self, "sell", amount))

    def accept(self, proposal):
        self.model.orderbook.messages.append(Message(self, "accept", proposal))

    def step(self):
        self.mental_step()

    def trace(self, text):
        self.model.console.append(f"{type(self).__name__} {self.unique_id} > {text}                                    ")

    def mental_init(self):
        pass

    def mental_step(self):
        pass

    def __str__(self):
        return f"agent{self.unique_id}"


#######################
# entities with minds
#######################

import random

class ZeroIntelligentBuyer(AgentBody):

    def mental_init(self):
        self.listening = False

    def mental_step(self):

        # passive loop
        if not self.listening:
            self.myprice = random.randint(0, self.capital)
            self.listening = True
            self.timeout = 10
            self.trace(f"waiting for offer on asset (my price is: {self.myprice})")

        offer_found = False
        for msg in self.model.orderbook.get_offers():
            if msg.action == "sell":
                if msg.refinement < self.myprice:
                    self.trace(f"accepting relevant offer by {msg.agent}")
                    self.accept(msg)
                    self.listening = False
                    break

        if self.listening and not offer_found:
            self.timeout -= 1
            if self.timeout == 0:
                self.listening = False


class ZeroIntelligentPlusBuyer(AgentBody):

    def mental_init(self):
        self.myprice = random.randint(0, self.capital)
        self.listening = False

    def mental_step(self):

        if not self.listening:
            self.listening = True
            self.timeout = 10
            self.trace(f"waiting for offer on asset (my price is: {self.myprice})")

        # passive loop
        offer_found = False
        for msg in self.model.orderbook.get_offers():
            if msg.action == "offer":
                if msg.refinement < self.myprice:
                    self.trace(f"accepting relevant offer by {msg.agent})")
                    self.accept(msg)
                    self.myprice -= 1
                    offer_found = True
                    self.listening = False
                    break

        if self.listening and not offer_found:
            self.timeout -= 1
            if self.timeout == 0:
                self.myprice += 1
                self.listening = False


class ZeroIntelligentSeller(AgentBody):

    def mental_init(self):
        self.offering = False

    def mental_step(self):
        # active loop
        if not self.offering:
            self.myprice = random.randint(0, self.capital)
            self.offering = True
            self.timeout = 10
            self.trace(f"offering asset for {self.myprice}")
            self.sell(self.myprice)
        else:
            for msg in self.messages:
                if msg.action == "accepted":
                    self.trace(f"offer accepted by {msg.agent}")
                    self.offering = False
            if self.offering:
                self.timeout -= 1
                if self.timeout == 0:
                    self.offering = False


class ZeroIntelligentPlusSeller(AgentBody):

    def mental_init(self):
        self.myprice = random.randint(0, self.capital)
        self.offering = False

    def mental_step(self):

        # active loop
        if self.offering is False:
            self.offering = True
            self.timeout = 10
            self.trace(f"offering asset for {self.myprice}")
            self.sell(self.myprice)
        else:
            for msg in self.messages:
                if msg.action == "accepted":
                    self.trace(f"offer accepted by {msg.agent}")
                    self.offering = False
                    self.myprice += 1
            if self.offering:
                self.timeout -= 1
                if self.timeout == 0:
                    self.myprice -= 1
                    self.offering = False


#######################
# world
#######################

class WorldModel(mesa.Model):

    def __init__(self, orderbook):
        self.entities = []
        self.orderbook = orderbook
        self.disabilities = {}
        self.schedule = mesa.time.RandomActivation(self)        
        self.end = False
        self.console = []

    def step(self):
        self.orderbook.step()
        self.schedule.step()    
        return self.end

    def add_entity(self, Entity_type, params=None):
        entity = Entity_type(len(self.entities), self, params)    
        self.schedule.add(entity)
        self.entities.append(entity)

    def add_disability(self, Entity_type, action, value):
        if Entity_type not in self.disabilities:
            self.disabilities[Entity_type] = {}
        if action not in self.disabilities[Entity_type]:
            self.disabilities[Entity_type][action] = []        
        self.disabilities[Entity_type][action].append(value)

    def remove_entity(self, entity):
        self.schedule.remove(entity)

    def remove_disability(self, Entity_type, action, value):
        self.disabilities[Entity_type][action].remove(value)


#######################
# viewer
#######################

import time

def move (x, y):
    print("\033[%d;%dH" % (y, x))

class WorldView:
    def __init__(self, world_model):
        self.world = world_model

    def init(self):
        print('\x1b[2J')

    def header(self):
        move(0, 0)
        print("mesagym -- minimal gym built on top on mesa -- market emulator\n")

    def show(self):
        self.header()

        print(">>> console <<<")

        for item in reversed(self.world.console[-20:]):
            print(item)

        time.sleep(0.1)


#######################
# main
#######################

if __name__ == "__main__":
    orderbook = OrderBook()
    model = WorldModel(orderbook)

    N = 1
    for i in range(N):
        model.add_entity(ZeroIntelligentBuyer, {"capital": 1000})
        model.add_entity(ZeroIntelligentSeller, {"capital": 1000})

    view = WorldView(model)
    view.init()

    T = 100
    while T > 0:
        T -= 1
        end = model.step()
        view.show()
        if end:
            break

