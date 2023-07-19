# mesa-gym

Infrastructure for AI experiments (RL, ML, planning, BDI, multi-agent scenarios, ...). It relies on the multi-agent library [Mesa](https://mesa.readthedocs.io/en/latest/) to design gym environments, to be embedded eg. in custom environments for [gymnasium](https://gymnasium.farama.org/).

## Main components

The infrastructure consists of different modules:
- the `mesa` part is about programming the environment and possibly hard-coding the agents: actions (and disabilities), what happens at performance, etc. When executed it maintains state, produces changes, and returns relevant events;
- custom `gymnasium` environments collect the states and compute the rewards (typically from events);
- training methods work on top of the `gymnasium` environments.

### Project structure

Structure of `mesa-gym` project:
- `/gyms` contains various gyms built on top of MESA
  - `/grid` contains grid-based symbolic worlds
  - `/market` contains communication-centered (eg. markets) worlds
  - `...`
  - `/test` clones a simple environment (no MESA) from gym to test the integration with gymnasium
- `/trainees` contains reusable agent learning modules
- `/common` contains helpers, eg. for data visualization

Conventionally, each `mesa-gym` world contains:
- `world.py`: agent and environmental models that run on `mesa`. they can be executed standalone
- `env.py`: custom environments for `gymnasium` building upon the world model
- `script.py`: simple script that runs the world, possibly reusing trained models
- `training.py`: learning script, creating behavioural models 

### Starting scripts

To start, you can run, from within a directory in `gyms`, eg. `grid/zzt_basic`

- to execute a multi-agent world, using only `mesa`:
```
python world.py 
``` 
- to execute a multi-agent world, using a `gymnasium` custom environment on top of `mesa`, and possibly pre-trained models
```
python script.py
``` 
- to train agents within `gymnasium` 
```
python training.py
```
- to visualize (meta-)data produced during training
```
python data_viz.py
````

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
