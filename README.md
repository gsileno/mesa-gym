# mesa-gym

Minimal gym infrastructure for AI experiments (RL, ML, planning, BDI, multi-agent scenarios, ...) based on the Mesa agent library for Python (https://mesa.readthedocs.io/en/latest/), and providing custom environments for gymnasium (https://github.com/Farama-Foundation/Gymnasium).

## Main components

The infrastructure consists of different modules:
- the `mesa` part is about programming the environment and possibly hard-coding the agents: actions (and disabilities), what happens at performance, etc. When executed it makes change on the environment and returns relevant events:
- the `gymnasium` environment collects the states and compute the rewards (typically from the events). 
- training methods work on top of the `gymnasium` environment

### Project structure

structure of `mesa-gym`:
- `/worlds`: worlds that runs on `mesa`. they can be executed standalone
  - `mesa_zzt.py`: 2D grid world, characters are inspired by the old classic ZZT (https://museumofzzt.com/).
  - `mesa_market.py`: a simple market with sale transactions (still in development, not working now)
- `/envs`: custom environments for `gymnasium` relying on `mesa` worlds 
  - `mesa_zzt_env.py`
- `/scripts`: simple scripts that runs the worlds, possibly reusing trained models
  - `mesa_zzt_script.py`
- `/trainees`: RL methods to create agent models
  - `/trained_models`: pre-trained models
    - `mesa_zzt_trainee.py`: training batch for agents in `mesa_zzt` 
    - `qlearning.py`: tabular q-learning agent
  - `/training_data`: training data and visualizationn helpers
    - `data_viz.py`: visualize variation of performance during training

### Starting scripts

To start, you can run (from within the `mesa_gym` directory)

- to execute a multi-agent world, using only `mesa`:
```
python worlds/mesa_zzt.py
``` 
- to execute a multi-agent world, using a `gymnasium` custom environment on top of `mesa`, and possibly pre-trained models
```
python scripts/mesa_zzt_script.py
``` 
- to train lion(s) and ranger(s) in `mesa_zzt` using tabular q-learning within `gymnasium` 
```
python trainees/mesa_zzt_trainee.py
```
- to visualize (meta-)data produced during training
```
python trainees/training_data/data_viz.py
````

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

The map is used to create the computational model of the envinroment, placing elements in the correct position. A `view` class can be used to generate a visual representation to follow the execution. The execution is discrete, and follows the convention of Mesa: 

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

For instance, the agent may take the perceptions (only a limited portion of the environment), identify the positions by the relative coordinates coming from the perception, and move randomly in any of those:

```
class RandomWalkingAgent(AgentBody):

    def mental_init(self):
        pass  

    def mental_step(self):
        percepts = self.get_percepts()
        potential_positions = list(percepts.keys())
        self.move(self.random.choice(potential_positions))
```

## extensions

In principle, one can use this framework to:
- setup reinforcement learning algorithms for creating policies automatically
- setup classificatory algorithms for creating abstractions of perceptual data
- setup hard-coded rules to regulate behaviour
- setup a full-fledged BDI decision-making cycle,
- ... 

## to do

- create interfaces towards pettingzoo.. 

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

If you require to visualize diagrams
```
pip install matplotlib
```
