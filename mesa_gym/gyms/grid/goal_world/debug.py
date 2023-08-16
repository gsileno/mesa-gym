# simple script to run the zzt-like 2d grid world from gymnasium

import mesa_gym.gyms.grid.goal_world.env as w

env = w.MesaGoalEnv()

import numpy as np
import pickle

import os
path = os.path.dirname(os.path.abspath(__file__))

empty_symbol = "█"

action2symbol = {
    0: "↖",
    1: "↑",
    2: "↗",
    3: "←",
    4: " ",
    5: "→",
    6: "↙",
    7: "↓",
    8: "↘"
}

def q_table_cell(cell):
    actions_to_value = {}
    output = ""
    for i, elem in enumerate(cell):
        actions_to_value[i] = elem
        output += "i: %.2f; " % elem
    return output

# add files here if you want to use a trained model
trained_models = {}
trained_models[0] = "models/Mouse_0_goal_world-qlearning_1000_0.05_1.0_0.002_0.1.pickle"

# load q_tables to use them
q_tables = {}
for id in trained_models.keys():
    with open(trained_models[id], "rb") as f:
        q_tables[id] = pickle.load(f)

########################################
## Exploration of q-table by execution #
########################################

obs, info = env.reset()

charts_best_actions = {}
charts_pos_values = {}

for _ in range(100):
    actions = {}
    for agent in env._get_agents():
        id = agent.unique_id

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

        best_action = int(np.argmax(q_tables[id][tuple(obs)]))
        x, y = agent.pos

        best_action_symbol = action2symbol[best_action]
        current_symbol = charts_best_actions[id][x][y]
        if current_symbol == empty_symbol:
            charts_best_actions[id][x][y] = action2symbol[best_action]
            charts_pos_values[id][x][y] = np.max(q_tables[id][tuple(obs)])
        elif current_symbol != action2symbol[best_action]:
            raise RuntimeError(f"Unexpected different value {current_symbol} vs {action2symbol[best_action]}. Perhaps the environment is not deterministic?")

        actions[id] = best_action

    obs, reward, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

for id in charts_best_actions:
    print("====================")
    print(f"Agent with id {id}")
    print("====================")
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

    print(best_action_grid)
    print(pos_value_grid)
    print()