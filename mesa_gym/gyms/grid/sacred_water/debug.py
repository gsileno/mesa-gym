import numpy as np
import pickle

import os
path = os.path.dirname(os.path.abspath(__file__))

import mesa_gym.gyms.grid.sacred_water.env as w
env = w.MesaSacredWaterEnv()

empty_symbol = "█"

action2symbol = {
    0: "↖",
    1: "↑",
    2: "↗",
    3: "←",
    4: "·",
    5: "→",
    6: "↙",
    7: "↓",
    8: "↘"
}

# add files here if you want to use a trained model
q_trained_models = {}
q_trained_models[2] = "models/Gatherer_2_goal_world-qlearning_1000_0.05_0.95_1.0_0.002_0.1.pickle"

dqn_trained_models = {}

########################################
## Load models                         #
########################################

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

########################################
## Exploration by execution #
########################################

print("------------------------------------------------")
print("Exploration by execution")
print("------------------------------------------------")
print(">>>> we run a single game and record the choices")
print("")

obs, info = env.reset()

charts_best_actions = {}
charts_pos_values = {}

for _ in range(100):
    actions = {}
    for agent in env._get_agents():
        id = agent.unique_id

        if id in q_tables:
            if tuple(obs) not in q_tables[id]:
                raise RuntimeError("Warning: state not present in the Q-table, have you loaded the correct one?")

        # create empty charts for each agent
        if id not in charts_best_actions:
            charts_best_actions[id] = {}
            charts_pos_values[id] = {}

            for i in range(env.model.width):
                charts_best_actions[id][i] = {}
                charts_pos_values[id][i] = {}

                for j in range(env.model.height):
                   charts_best_actions[id][i][j] = empty_symbol
                   charts_pos_values[id][i][j] = None

        if id in q_tables:
            best_action = int(np.argmax(q_tables[id][tuple(obs)]))
            best_action_value = np.max(q_tables[id][tuple(obs)])
        elif id in dqn_models:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            best_action = int(dqn_models[id](state).max(1)[1].view(1, 1))
            best_action_value = dqn_models[id](state).tolist()[0][best_action]
        else:
            raise RuntimeError(f"No model available for agent id {id}")

        x, y = agent.pos

        best_action_symbol = action2symbol[best_action]
        current_symbol = charts_best_actions[id][x][y]
        if current_symbol == empty_symbol:
            charts_best_actions[id][x][y] = action2symbol[best_action]
        elif current_symbol != action2symbol[best_action]:
            raise RuntimeError(f"Unexpected different value {current_symbol} vs {action2symbol[best_action]}. Perhaps the environment is not deterministic?")

        charts_pos_values[id][x][y] = best_action_value
        actions[id] = best_action

    obs, reward, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

for id in charts_best_actions:
    print(f"==== Agent with id {id}")
    print()

    best_action_grid = ""
    pos_value_grid = ""
    for x in charts_best_actions[id].keys():
        for y in charts_best_actions[id][x].keys():
            best_action_grid += charts_best_actions[id][x][y]
            if charts_pos_values[id][x][y] is not None:
                pos_value_grid += "%.2f " % charts_pos_values[id][x][y]
            else:
                pos_value_grid += "---- "
        best_action_grid += "\n"
        pos_value_grid += "\n"

    print("chart with best action depending on position")
    print(best_action_grid)

    print("chart with best action value depending on position")
    print(pos_value_grid)
    print()

#######################
## Static exploration #
#######################

print("------------------------------------------------")
print("Static exploration")
print("------------------------------------------------")
print(">>>> we modify the agent starting position")
print("")

obs, info = env.reset()

size = env.model.width * env.model.height

# here we reverse engineer obs, by cutting the views of the different agents.
# then we identify the view of a specific agent by using its initial position
views = []
view = []
for i, c in enumerate(obs):
    view.append(c)
    j = int((i+1)/size)
    if j > 0 and (i+1) % size == 0:
        views.append(view)
        view = []

charts_best_actions = {}
charts_pos_values = {}

for agent in env._get_agents():
    id = agent.unique_id

    x, y = agent.pos
    pos = x + y * env.model.width

    agent_scope = None
    for i, view in enumerate(views):
        if view[pos] == 1:
            agent_scope = i
    if agent_scope is None:
        raise RuntimeError("Wrong reverse engineering of obs")

    # create empty charts for each agent
    if id not in charts_best_actions:
        charts_best_actions[id] = {}
        charts_pos_values[id] = {}

        for i in range(env.model.width):
            charts_best_actions[id][i] = {}
            charts_pos_values[id][i] = {}

            for j in range(env.model.height):
               charts_best_actions[id][i][j] = empty_symbol
               charts_pos_values[id][i][j] = None

    mod_pos = x + y * env.model.width + size * agent_scope

    for x in range(env.model.width):
        for y in range(env.model.height):
            obs[mod_pos] = 0
            mod_pos = x + y * env.model.width + size * agent_scope
            obs[mod_pos] = 1

            if id in q_tables:
                if tuple(obs) in q_tables[id]:
                    best_action = int(np.argmax(q_tables[id][tuple(obs)]))
                    best_action_value = np.max(q_tables[id][tuple(obs)])
#                else:
#                    raise RuntimeError("Warning: state not present in the Q-table, have you loaded the correct one?")
            elif id in dqn_models:
                state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                best_action = int(dqn_models[id](state).max(1)[1].view(1, 1))
                best_action_value = dqn_models[id](state).tolist()[0][best_action]
            else:
                raise RuntimeError("Unknown agent with id {id}")

            if best_action_value >= 0.01:
                charts_best_actions[id][x][y] = action2symbol[best_action]
                charts_pos_values[id][x][y] = best_action_value
            else:
                charts_best_actions[id][x][y] = "."
                charts_pos_values[id][x][y] = best_action_value

for id in charts_best_actions:
    print(f"==== Agent with id {id}")
    print()

    best_action_grid = ""
    pos_value_grid = ""
    for x in charts_best_actions[id].keys():
        for y in charts_best_actions[id][x].keys():
            best_action_grid += charts_best_actions[id][x][y]
            if charts_pos_values[id][x][y] is not None:
                pos_value_grid += "%.2f " % charts_pos_values[id][x][y]
            else:
                pos_value_grid += "---- "
        best_action_grid += "\n"
        pos_value_grid += "\n"

    print("chart with best action depending on starting position")
    print(best_action_grid)

    print("chart with best action value depending on starting position")
    print(pos_value_grid)
    print()