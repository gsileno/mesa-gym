# environment for gymnasium

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mesa_gym.gyms.grid.sacred_water.world as w

class MesaSacredWaterEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, map=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.fps = self.metadata["render_fps"]

        if map is None:
            map = w.default_map 

        self.map = map
        self.model = self._get_world()
        self.booting = True

        # the potential actions are just movements in this environment
        self.potential_actions = w.AgentBody.get_directions()
        n_actions = len(self.potential_actions)

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        self.entities = self._get_entities()
        self.agents = self._get_agents()
        self.events = {} # necessary for the first call to info

        size = self.model.width * self.model.height

        for agent in self.agents:
            self.action_space[agent.unique_id] = spaces.Discrete(n_actions)

        # the observation space is made by the cartesian product
        # of all grid with the position of each entity
        # there is only one state for each coordinate, standing for presence
        MIN = 0; MAX = 1
        n_features = size * (MAX - MIN) * len(self.entities)
        features_high = np.array([MAX] * n_features, dtype=np.float32)
        features_low = np.array([MIN] * n_features, dtype=np.float32)
        self.observation_space = spaces.Box(features_low, features_high)

    def _get_world(self):
        return w.create_world(self.map, w.Symbol)

    def _get_entities(self):
        return self.model.entities

    def _get_agents(self):
        agents = []
        for entity in self.model.entities:
            if type(entity) is w.Gatherer:
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
            self.view = w.WorldView(self.model, name="sacred water", fps=self.fps)
            self.view.init()
            self.view.show()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_rewards(self, events):
        rewards = {}

        for agent, event_type in events:
            self.events[agent] = {}
            if type(agent) is w.Gatherer: # mouse finds cheese
                rewards[agent.unique_id] = 0
                if type(event_type) is tuple:
                    if event_type[1] is not None:
                        rewards[agent.unique_id] -= 1
                        self.events[agent]["moving"] = 1
                    else: 
                        rewards[agent.unique_id] -= 0
                        if type(event_type[0]) is w.Fruit:
                            self.events[agent]["starved"] = 1
                        else:
                            self.events[agent]["shrivelled"] = 1

                if type(event_type) is w.Fruit:
                    rewards[agent.unique_id] += 100
                    self.events[agent]["eating"] = 1
                elif type(event_type) is w.Water:
                    rewards[agent.unique_id] += 0
                    self.events[agent]["drinking"] = 1
                # elif event_type is False:  # movement has failed
                #     rewards[agent.unique_id] = -5
                #     self.events[agent]["collided"] = 1

        return rewards


    def step(self, actions):

        self.events = {}

        for agent in self._get_agents():
            agent.next_action = self.potential_actions[actions[agent.unique_id]]

        # there are two components of event in one step:
        # 1. the action performed by the agents
        # 2. the outcomes consequent to the actions
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



