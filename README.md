# mesa-gym

Minimal gym for AI experiments (RL, ML, planning, etc.) based on the Mesa agent library for Python (https://mesa.readthedocs.io/en/latest/).
Characters are inspired by the old classic ZZT (https://museumofzzt.com/).

### Components

#### World definition

The map of the target 2D environment is specified as a multi-line string.

For instance in the current code we have:

```
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
```

The map shows:

- a player `☺`, 
- a lion `Ω`, which ends the game if it touches the player, 
- some grass `░`, that grows in time, but is destroyed to some degree when someone walks on it
- walls `█` (that blocks movement), a gem `♦` (the end goal).

All components of the scene are agents handled by mesa.

#### Setup and execution

The map is used to create the computational model of the envinroment, placing elements in the correct position. A `view` class can be used to generate a visual representation to follow the execution. The execution is discrete, and follows the convention of Mesa. 

```
model = create_world(map)
view = WorldView(model)
view.init()

while True:  
    end = model.step()
    view.show() 
    if end:
        break
```

#### Body and mental elements

To facilitate problem decomposition, players are defined by body (dealing with mechanicistic interventions on and by the environment), and mind (dealing with sensory/motor decision mechanisms). As a good practice, given a certain domain, intervention of modelers should be only at mental level. 
So, in practice, the body eg. `AgentBody(mesa.Agent)` class that comes with the environment, which is extended by a mental element.

For instance in this case, the agent takes the perceptions (only a limited portion of the environment), identify the positions by the relative coordinates coming from the perception, and move randomly in any of those. 

```
class RandomWalkingAgent(AgentBody):

    def mental_init(self):
        pass  

    def mental_step(self):
        percepts = self.get_percepts()
        potential_positions = list(percepts.keys())
        self.move(self.random.choice(potential_positions))
```

### Dependencies

```
pip install mesa
```
