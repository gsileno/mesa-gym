# simple script to run the lumberjack 2d grid world from mesa-gym

import mesa_gym.envs.mesa_lumberjack_env
env = mesa_gym.envs.mesa_lumberjack_env.MesaLumberjackEnv(render_mode="human")

import os
path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pickle

# add filenames to this dict if you want to use a trained model for some entities
# eg. trained_models[id] = "<FILENAME.pickle>"
trained_models = {}
trained_models[6] = f"{path}/../training/models/StrongLumberjack_6_lumberjack-qlearning_1000_0.05_1.0_0.002_0.1.pickle"
trained_models[4] = f"{path}/../training/models/WeakLumberjack_4_lumberjack-qlearning_1000_0.05_1.0_0.002_0.1.pickle"

def load_trained_models(trained_models):
    q_tables = {}
    for id in trained_models.keys():
        with open(trained_models[id], "rb") as f:
            q_tables[id] = pickle.load(f)
    return q_tables

# load q_tables from trained models
q_tables = load_trained_models(trained_models)

# start environment
observations, info = env.reset()

for _ in range(1000):
    actions = {}
    for agent in env.agents:
        id = agent.unique_id
        observation = tuple(observations[id])
        if id in trained_models:
            action = int(np.argmax(q_tables[id][observation]))
        else:
            action = env.action_space[id].sample()
        actions[id] = action

    observations, rewards, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        exit(1)
        observations, info = env.reset()

env.close()