# zzt-like gridworld as environment for gymnasium
import copy

import numpy as np

import mesa_gym.worlds.mesa_zzt as mesa_zzt

import gymnasium as gym
from gymnasium import spaces

class MesaZZTEnv(gym.Env):
    """
        MesaZZTEnv involves a ZZT-like world populated by several entities.

        ### Action Space
        The agent takes a 1-element vector for actions.
        All actions are just about moving; directions are provided by the MESA grid world

        ### Observation Space
        The observation space consists of:
        - a map with the idle objects
        - for each moving agent, a value representing the agent's current position as current_row * n_cols + current_col.
        - for each agent with state: an array of possible states
        Therefore, the number of possible observations depends on the size of the map, the number of agents and of states.

        ### Arguments
        ```
        gym.make('MesaZZTEnv-v0', map: string = None)
        ```
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, map=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.fps = self.metadata["render_fps"]

        if map is None:
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

        self.map = map
        self.model = self._get_world()
        self.booting = True

        n_positions = self.model.width * self.model.height

        self.potential_actions = mesa_zzt.AgentBody.get_directions()
        n_actions = len(self.potential_actions)

        self.entities = self._get_entities()
        self.agents = self._get_agents()

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        self.events = {}

        for entity in self.entities:
            self.observation_space[entity.unique_id] = spaces.Discrete(n_positions)

        for agent in self.agents:
            self.action_space[agent.unique_id] = spaces.Discrete(n_actions)

    def _get_world(self):
        return mesa_zzt.create_world(self.map)

    def _get_entities(self):
        return self.model.entities

    def _get_agents(self):
        lions = []
        grass_zones = []
        rangers = []

        for entity in self.model.entities:
            if type(entity) is mesa_zzt.LionAgent:
                lions.append(entity)
            elif type(entity) is mesa_zzt.Grass:
                grass_zones.append(entity)
            elif type(entity) is mesa_zzt.RangerAgent:
                rangers.append(entity)

        return lions + rangers

    def _get_obs(self):
        return { entity.unique_id: entity.pos for entity in self.entities }

    def _get_info(self):
        info = {}
        for agent in self.agents:
            if agent.unique_id not in info:
                info[agent.unique_id] = {}
            info[agent.unique_id]["energy"] = agent.energy
            if agent in self.events:
                for event in self.events[agent]:
                    info[agent.unique_id][event] = self.events[agent][event]
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.booting:
            self.booting = False
        else:
            self.model = self._get_world()

        if self.render_mode == "human":
            self.view = mesa_zzt.WorldView(self.model, self.fps)
            self.view.init()
            self.view.show()

        self.agents = self._get_agents()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, actions):

        for agent in self.agents:
            agent.next_action = self.potential_actions[actions[agent.unique_id]]

        self.events = {}
        terminated, events = self.model.step()

        if self.render_mode == "human":
            self._render_frame()

        rewards = {}
        for agent, reached_entity in events:
            self.events[agent] = {}
            if reached_entity is False:
                rewards[agent.unique_id] = -1
                self.events[agent]["collided"] = 1
            else:
                if type(agent) is mesa_zzt.LionAgent:
                    self.events[reached_entity] = {}

                    if type(reached_entity) is mesa_zzt.RangerAgent:
                        rewards[agent.unique_id] = 10
                        self.events[agent]["success"] = 1
                        self.events[reached_entity]["failure"] = 1

                elif type(agent) is mesa_zzt.RangerAgent:
                    if type(reached_entity) is mesa_zzt.Diamond:
                        rewards[agent.unique_id] = 10
                        self.events[agent]["success"] = 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, rewards, terminated, False, info

    def render(self):
        self.view.show()

    def _render_frame(self):
        self.view.show()



