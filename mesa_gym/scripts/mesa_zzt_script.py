# simple script to run the zzt-like 2d grid world from gymnasium

import mesa_gym.envs.mesa_zzt_env
env = mesa_gym.envs.mesa_zzt_env.MesaZZTEnv(render_mode="human")

import numpy as np
import pickle

import os
path = os.path.dirname(os.path.abspath(__file__))

# add files here if you want to use a trained model # NOT WORKING NOW
trained_models = {}
trained_models[35] = f"{path}/../trainees/trained_models/LionAgent_35_zzt-qlearning_100000_0.05_1.0_2e-05_0.1.pickle"
trained_models[7] = f"{path}/../trainees/trained_models/RangerAgent_7_zzt-qlearning_100000_0.05_1.0_2e-05_0.1.pickle"

# load q_tables to use them
q_tables = {}
for id in trained_models.keys():
    with open(trained_models[id], "rb") as f:
        q_tables[id] = pickle.load(f)

obs, info = env.reset()

for _ in range(1000):

    actions = {}
    for agent in env.agents:
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


