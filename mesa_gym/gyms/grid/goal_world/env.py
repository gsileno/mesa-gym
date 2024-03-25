# basic gridworld as environment for gymnasium

import mesa_gym.gyms.grid.goal_world.world as w
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class MesaGoalEnv(gym.Env):
    """
        MesaGoalEnv involves a minimal grid world.

        ### Action Space
        The agent takes a 1-element vector for actions.
        All actions are just about moving; directions are provided by the MESA grid world

        ### Observation Space
        The observation space consists of a matrix with all positions of entities

        ### Arguments
        ```
        gym.make('MesaGoalEnv-v0', map: string = None)
        ```
    """

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, map=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.fps = self.metadata["render_fps"]

        if map is None:
#             map = """
# |---|
# |  ☺|
# |   |
# | ♠ |
# |   |
# |---|
# """
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
        self.map = map
        self.model = self._get_world()
        self.booting = True

        self.potential_actions = w.AgentBody.get_directions()
        n_actions = len(self.potential_actions)

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        self.entities = self._get_entities()
        self.agents = self._get_agents()
        self.events = {}

        size = self.model.width * self.model.height

        for agent in self.agents:
            self.action_space[agent.unique_id] = spaces.Discrete(n_actions)

        MIN = 0; MAX = 1
        n_features = size * (MAX - MIN) * len(self.entities)
        features_high = np.array([MAX] * n_features, dtype=np.float32)
        features_low = np.array([MIN] * n_features, dtype=np.float32)
        self.observation_space = spaces.Box(features_low, features_high)

    def _get_world(self):
        return w.create_world(self.map)

    def _get_entities(self):
        return self.model.entities

    def _get_agents(self):
        agents = []
        for entity in self.model.entities:
            if type(entity) is w.Mouse:
                agents.append(entity)
        return agents

    def _get_obs(self):
        return self.model.get_positions()

    def _get_info(self):
        info = {}
        for agent in self._get_agents():
            if agent.unique_id not in info:
                info[agent.unique_id] = {}
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
            self.view = w.WorldView(self.model, self.fps)
            self.view.init()
            self.view.show()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_rewards(self, events):
        rewards = {}

        # for agent in self._get_agents():
        #     rewards[agent.unique_id] = -1               # losing energy for each time step

        for agent, reached_entity in events:
            self.events[agent] = {}
            # if reached_entity is False:                 # collision with wall
            #     rewards[agent.unique_id] = -5
            #     self.events[agent]["collided"] = 1
            # else:
            if type(agent) is w.Mouse: # mouse finds cheese
                if type(reached_entity) is w.Cheese:
                    rewards[agent.unique_id] = -10
                    self.events[agent]["success"] = 0
                # else:
                #     rewards[agent.unique_id] = 1

        return rewards


    def step(self, actions):

        for agent in self._get_agents():
            agent.next_action = self.potential_actions[actions[agent.unique_id]]

        self.events = {}
        terminated, events = self.model.step()

        if self.render_mode == "human":
            self._render_frame()

        rewards = self._get_rewards(events)
        observation = self._get_obs()
        info = self._get_info()

        return observation, rewards, terminated, False, info

    def render(self):
        self.view.show()

    def _render_frame(self):
        self.view.show()



