# simple script to run the zzt-like 2d grid world from gymnasium

import mesa_gym.gyms.grid.zzt_basic.env as w
env = w.MesaZZTEnv(render_mode="human")

import numpy as np
import pickle

import os
path = os.path.dirname(os.path.abspath(__file__))

# add files here if you want to use a trained model
trained_models = {}
trained_models[7] = "models/RangerAgent_7_zzt-qlearning_1000_0.05_1.0_0.002_0.1.pickle"
trained_models[35] = "models/LionAgent_35_zzt-qlearning_1000_0.05_1.0_0.002_0.1.pickle"

# load q_tables to use them
q_tables = {}
for id in trained_models.keys():
    with open(trained_models[id], "rb") as f:
        q_tables[id] = pickle.load(f)

obs, info = env.reset()

for _ in range(1000):

    actions = {}
    for agent in env._get_agents():
        id = agent.unique_id
        if id in trained_models:
            action = int(np.argmax(q_tables[id][tuple(obs)]))
        else:
            action = env.action_space[id].sample()
        actions[id] = action

    obs, reward, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        observation, info = env.reset()

env.close()


