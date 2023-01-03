# load the target environment
import pandas as pd

import mesa_gym.envs.mesa_zzt_env

env = mesa_gym.envs.mesa_zzt_env.MesaZZTEnv(render_mode=None)

# the env returns a dict of observation spaces for each entity
# here we flat them

import gymnasium as gym

# env = gym.wrappers.FlattenObservation(env)

agents = []
for mesa_agent in env.agents:
    agents.append(mesa_agent.unique_id)

# take the number of dimensions
nb_states = gym.spaces.flatdim(env.observation_space)
print(f"number of states: {nb_states}")

# define a simple Q-learning agent

from collections import defaultdict
import numpy as np


class QLearningTrainee:
    def __init__(
            self,
            agent,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):

        self.agent = agent

        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space[self.agent].n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return env.action_space[self.agent].sample()
        else:
            return int(np.argmax(self.q_values[tuple(obs)]))

    def update(
            self,
            obs: tuple,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[tuple(obs)][action]
        )

        self.q_values[tuple(obs)][action] = (
                self.q_values[tuple(obs)][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)


# target of training

n_episodes = 1000  # 100_000

# create the trainee agent instances

from tqdm import tqdm

data = {}
data_fields = []

# # Vanilla environment with random actions
#
# for episode in tqdm(range(n_episodes)):
#     obs, info = env.reset()
#     done = False
#
#     data[episode] = {}
#
#     step = 0
#     while not done:
#         actions = env.action_space.sample()
#         obs, rewards, terminated, truncated, info = env.step(actions)
#
#         # collect data
#         data[episode][step] = {}
#         for agent in agents:
#             data[episode][step][agent] = {}
#             data[episode][step][agent]["reward"] = rewards[agent] if agent in rewards else 0
#             if agent in info:
#                 for key in info[agent]:
#                     if key not in data_fields:
#                         data_fields.append(key)
#                     data[episode][step][agent][key] = info[agent][key]
#
#         step += 1
#         done = terminated or truncated

learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

trainees = {}
for agent in agents:
    trainees[agent] = QLearningTrainee(agent=agent, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    data[episode] = {}

    step = 0
    while not done:
        actions = {}
        for agent in agents:
            actions[agent] = trainees[agent].get_action(obs)
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # collect data
        data[episode][step] = {}
        for agent in agents:
            data[episode][step][agent] = {}
            data[episode][step][agent]["reward"] = rewards[agent] if agent in rewards else 0
            if agent in info:
                for key in info[agent]:
                    if key not in data_fields:
                        data_fields.append(key)
                    data[episode][step][agent][key] = info[agent][key]

        # update the agent
        for agent in agents:
            reward = rewards[agent] if agent in rewards else 0
            trainees[agent].update(obs, actions[agent], reward, terminated, next_obs)
        obs = next_obs

        # update if the environment is done and the current obs
        done = terminated or truncated
        step += 1

    for agent in agents:
        trainees[agent].decay_epsilon()

rows = []
for episode in data.keys():
    for step in data[episode].keys():
        for agent in data[episode][step]:
            row = {}
            row["agent"] = agent
            row["episode"] = episode
            row["step"] = step
            row["reward"] = data[episode][step][agent]["reward"]
            for key in data_fields:
                if key in data[episode][step][agent]:
                    row[key] = data[episode][step][agent][key]
                else:
                    row[key] = 0
            rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)

import matplotlib.pyplot as plt

episodes = df.episode.unique()
agents = df.agent.unique()

for agent in agents:
    x = []
    y = {}
    y[0] = []
    y[1] = []
    y[2] = []
    y[3] = []

    for episode in episodes:
        x.append(episode)
        Y = df[(df["agent"] == agent) & (df["episode"] == episode)] \
            [["reward", "collided", "success", "failure"]].sum()
        y[0].append(Y.reward)
        y[1].append(Y.collided)
        y[2].append(Y.success)
        y[3].append(Y.failure)

    plt.title(f"agent {agent}")
    plt.xlabel("episode")
    plt.ylabel("amount")
    plt.plot(x, y[0], color="blue")
    # plt.plot(x, y[1], color="black")
    # plt.plot(x, y[2], color="green")
    # plt.plot(x, y[3], color="red")

    plt.show()