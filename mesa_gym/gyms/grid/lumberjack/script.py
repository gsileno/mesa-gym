# simple script to run the lumberjack 2d grid world from mesa-gym

# add filenames to this dict if you want to use a trained model for some entities
# eg. trained_models[id] = "<FILENAME.pickle>"
trained_models = {}
trained_models["StrongLumberjack"] = "models/StrongLumberjack_lumberjack-qlearning_selfishness_500_0.05_1.0_0.004_0.1.pickle"
trained_models["WeakLumberjack"] = "models/WeakLumberjack_lumberjack-qlearning_selfishness_500_0.05_1.0_0.004_0.1.pickle"

import mesa_gym.gyms.grid.lumberjack.env as e
env = e.MesaLumberjackEnv(render_mode="human")

import os
path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pickle

def load_trained_models(trained_models):
    q_tables = {}
    for agent_type in trained_models.keys():
        with open(trained_models[agent_type], "rb") as f:
            q_tables[agent_type] = pickle.load(f)
    return q_tables

# load q_tables from trained models
q_tables = load_trained_models(trained_models)

# start environment
observations, info = env.reset()

for _ in range(100):
    actions = {}
    for agent in env.agents:
        id = agent.unique_id
        agent_type = type(agent).__name__
        observation = tuple(observations[id])
        if agent_type in trained_models:
            if observation not in q_tables[agent_type]:
                action = env.action_space[id].sample()
                agent.trace("UNKNOWN CONDITION in my Qtable: random extraction")
            else:
                action = int(np.argmax(q_tables[agent_type][observation]))
        else:
            action = env.action_space[id].sample()
        actions[id] = action

    observations, rewards, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        exit(1)
        observations, info = env.reset()

env.close()