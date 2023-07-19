import mesa_gym.gyms.grid.lumberjack.world as mesa_lumberjack
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MesaLumberjackEnv(gym.Env):
    """
        MesaLumberjack involves a grid world populated by lumberjacks that cut trees

        ### Action Space
        The agent takes a 1-element vector for actions.
        All actions are just about moving; directions are provided by the MESA grid world

        ### Observation Space

        For each tree, the observation space consists of:
        - a state: strength, position

        For each lumberjack, the observation space consists of:
        - a state: strength, position
        - relative perceptions:
            - a vector of 9 items for each neighbour cell with sum of trees' strengths if present, 0 otherwise
            - a vector of 9 items for each neighbour cell with sum of agents' strengths if present, 0 otherwise

        ### Arguments
        ```
        gym.make('MesaLumberjack-v0', map: string = None)
        ```
    """

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, map=None, value_dimension=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.fps = self.metadata["render_fps"]

        if value_dimension is None:
            self.value_dimension = 1
        else:
            self.value_dimension = value_dimension

        self.map = map
        self.model = self._get_world()

        self.booting = True

        # for a lumberjack agent,
        # the number of possible actions are the directions of movements (9)
        self.potential_actions = mesa_lumberjack.Lumberjack.get_directions()
        n_actions = len(self.potential_actions)

        # for a lumberjack agent,
        # the number of possible (perceptual) states are entailed by the perception
        # a 3x3 grid with the sum of strengths of nearby trees and a 3x3 grid with sum of strengths of nearby agents
        # the two grids are encoded in a 9*2 vector
        # we assume a max value of 2
        n_features = 9 * 2
        features_high = np.array([2] * n_features, dtype=np.float32)
        features_low = np.array([0] * n_features, dtype=np.float32)

        self.entities = self._get_entities()
        self.agents = self._get_agents()
        self.events = {}

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        for agent in self.agents:
            self.action_space[agent.unique_id] = spaces.Discrete(n_actions)
            self.observation_space[agent.unique_id] = spaces.Box(features_high, features_low)

    def _get_world(self):
        if self.map is None:
            return mesa_lumberjack.create_random_world(5, 5,
                                                       {mesa_lumberjack.WeakLumberjack: 1,
                                                        mesa_lumberjack.StrongLumberjack: 1,
                                                        mesa_lumberjack.Strength1Tree: 3,
                                                        mesa_lumberjack.Strength2Tree: 3})

        else:
            # map = """
            # |-----|
            # |2  2 |
            # |1   2|
            # |  ☺  |
            # |1 ☻  |
            # | 1   |
            # |-----|
            # """
            return mesa_lumberjack.load_world(self.map)

    def _get_entities(self):
        return self.model.entities

    def _get_agents(self):
        lumberjacks = []
        for entity in self.model.entities:
            if isinstance(entity, mesa_lumberjack.Lumberjack):
                lumberjacks.append(entity)
        return lumberjacks

    def _get_obs(self):
        return {agent.unique_id: agent.get_percepts() for agent in self.agents}

    def _get_info(self):
        info = {}
        for agent in self.agents:
            if agent.unique_id not in info:
                info[agent.unique_id] = {}
            info[agent.unique_id]["strength"] = agent.strength
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
            self.view = mesa_lumberjack.WorldView(self.model, self.fps)
            self.view.init()
            self.view.show()

        self.agents = self._get_agents()
        observations = self._get_obs()
        infos = self._get_info()

        return observations, infos

    def step(self, actions):

        rewards = {}
        for agent in self.agents:
            agent.next_action = self.potential_actions[actions[agent.unique_id]]

        self.events = {}
        terminated, events = self.model.step()

        if self.render_mode == "human":
            self._render_frame()

        for agent, tree in events:
            self.events[agent] = {}
            if tree is False:
                self.events[agent]["failure"] = 1
            else:
                self.events[agent]["success"] = 1
                if self.value_dimension == 1:
                    rewards[agent.unique_id] = 1  # selfishness
                elif self.value_dimension == 2:
                    for other in self.agents:
                        if other != agent:
                            rewards[other.unique_id] = 1  # altruism
                elif self.value_dimension == 3:
                    if self.model.ntrees == 0:
                        rewards[agent.unique_id] = -1  # environmentalism

        observations = self._get_obs()
        infos = self._get_info()

        return observations, rewards, terminated, False, infos

    def render(self):
        self.view.show()

    def _render_frame(self):
        self.view.show()
