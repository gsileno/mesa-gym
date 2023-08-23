# simple script to run the zzt-like 2d grid world from gymnasium

import mesa_gym.gyms.grid.goal_world.env as w
env = w.MesaGoalEnv(render_mode="human")

import numpy as np
import pickle

import os
path = os.path.dirname(os.path.abspath(__file__))

# add files here if you want to use a trained model
trained_models = {}
trained_models[0] = "models/Mouse_0_goal_world-qlearning_10000_0.05_1.0_0.0002_0.1.pickle"

# load q_tables to use them
q_tables = {}
for id in trained_models.keys():
    with open(trained_models[id], "rb") as f:
        q_tables[id] = pickle.load(f)

# load q_tables to use them
q_tables = {}
for id in trained_models.keys():
    with open(trained_models[id], "rb") as f:
        q_tables[id] = pickle.load(f)

obs, info = env.reset()
n_games = 0

for _ in range(100):

    actions = {}
    for agent in env._get_agents():
        id = agent.unique_id
        if id in trained_models:
            if tuple(obs) not in q_tables[id]:
                raise RuntimeError("Warning: state not present in the Q-table, have you loaded the correct one?")
            action = int(np.argmax(q_tables[id][tuple(obs)]))
        else:
            action = env.action_space[id].sample()
        actions[id] = action

    obs, reward, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        obs, info = env.reset()
        n_games += 1

env.close()

print(f"Number of games played: {n_games}")

