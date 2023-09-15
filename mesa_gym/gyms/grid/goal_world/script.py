# simple script to run the zzt-like 2d grid world from gymnasium

import mesa_gym.gyms.grid.goal_world.env as w
env = w.MesaGoalEnv(render_mode="human")

import numpy as np
import pickle

import os
path = os.path.dirname(os.path.abspath(__file__))

q_trained_models = {}
dqn_trained_models = {}

# add files here if you want to use a trained model
# q_trained_models[0] = "models/Mouse_0_goal_world-qlearning_10000_0.05_0.95_1.0_0.0002_0.1.pickle"
# dqn_trained_models[0] = "models/Mouse_0_goal_world-DQNlearning_10000_128_0.005_0.0001_0.99_0.9_0.00018_0.05.pt"
dqn_trained_models[0] = "models/Mouse_0_goal_world-DQNlearning_10000_128_0.005_0.0001_0.99_0.9_0.00018_0.05.pt"

# load q_tables
q_tables = {}
for id in q_trained_models.keys():
    with open(q_trained_models[id], "rb") as f:
        q_tables[id] = pickle.load(f)

# load dqn_models
dqn_models = {}
for id in dqn_trained_models.keys():

    import gymnasium as gym
    import torch
    from mesa_gym.trainers.DQN import DQN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_actions = gym.spaces.flatdim(env.action_space)
    nb_states = gym.spaces.flatdim(env.observation_space)

    dqn_models[id] = DQN(nb_states, nb_actions)
    dqn_models[id].load_state_dict(torch.load(dqn_trained_models[id]))
    dqn_models[id].eval()

# playing loop

obs, info = env.reset()
n_games = 0

for _ in range(100):

    actions = {}
    for agent in env._get_agents():
        id = agent.unique_id
        if id in q_trained_models:
            if tuple(obs) not in q_tables[id]:
                raise RuntimeError("Warning: state not present in the Q-table, have you loaded the correct one?")
            action = int(np.argmax(q_tables[id][tuple(obs)]))
        elif id in dqn_trained_models:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = dqn_models[id](state).max(1)[1].view(1, 1)
        else:
            action = env.action_space[id].sample()
        actions[id] = action

    obs, reward, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        obs, info = env.reset()
        n_games += 1

env.close()

print(f"Number of games played: {n_games}")

