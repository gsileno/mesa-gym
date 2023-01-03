# mesa-gym

Minimal gym infrastructure for AI experiments (RL, ML, planning, BDI, multi-agent scenarios, ...) based on the Mesa agent library for Python (https://mesa.readthedocs.io/en/latest/). 

Available worlds:
- `mesa_zzt`: 2D grid world, characters are inspired by the old classic ZZT (https://museumofzzt.com/).
- `mesa_market`: market with various sale transactions 

Those worlds can be run standalone, or connected as environments for the `gymnasium`/`pettingzoo` library

## Available worlds  

### ZZT-like 2D grid world

The map of the target 2D environment is specified as a multi-line string.
For instance:

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

#### setup and execution

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

#### separation of 'body' and 'mental' elements

To facilitate problem decomposition, agents may be defined as body (dealing with mechanicistic interventions on and by the environment) and mind (dealing with sensory/motor decision mechanisms). As a good practice, given a certain domain, intervention of modelers should be only at mental level, that is, there is a body eg. `AgentBody(mesa.Agent)` class that comes with the environment, which is then extended by a mental element.

In the current code, for instance, the agent takes the perceptions (only a limited portion of the environment), identify the positions by the relative coordinates coming from the perception, and move randomly in any of those. 

```
class RandomWalkingAgent(AgentBody):

    def mental_init(self):
        pass  

    def mental_step(self):
        percepts = self.get_percepts()
        potential_positions = list(percepts.keys())
        self.move(self.random.choice(potential_positions))
```

### market world

[..] todo 



## extensions

In principle, one can use this framework to:
- setup reinforcement learning algorithms for creating policies
- setup classificatory algorithms for creating abstractions of perceptual data
- setup hard-coded rules to regulate behaviour
- setup a full-fledged BDI decision-making cycle,
- ... 

## to do

- this is a super-minimal self-contained script at the moment, should be refactored to be more modular
- create interfaces towards gymnasium and pettingzoo.. 

## dependencies

The fastest way to install all dependencies is to install `mesa_gym` as a package:
```
pip install -e .
```

If you want to install packages manually, for the worlds to run:
```
pip install mesa
```

To use gymnasium for RL:
```
pip install gymnasium
```

For worlds requiring graphics:
```
pip install pygame
```
