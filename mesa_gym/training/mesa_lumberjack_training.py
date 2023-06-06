import os
path = os.path.dirname(os.path.abspath(__file__))

# load the target environment
import mesa_gym.envs.mesa_lumberjack_env
env = mesa_gym.envs.mesa_lumberjack_env.MesaLumberjackEnv(render_mode=None)

import gymnasium as gym

agent_ids = []
agent_id_to_type = {}
for mesa_agent in env.agents:
    agent_ids.append(mesa_agent.unique_id)
    agent_id_to_type[mesa_agent.unique_id] = type(mesa_agent).__name__

# # take the number of dimensions
# print(env.observation_space)
# nb_states = gym.spaces.flatdim(env.observation_space)
# print(f"number of states: {nb_states}")

# target of training

n_episodes = 1_000  # 100_000

# create the trainee agent instances

from tqdm import tqdm

data = {}
data["fields"] = []

# ######################################
# # vanilla (random actions)
# ######################################
#
# experiment_name = f"lumberjack-random_{n_episodes}"
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
#                     if key not in data["fields"]:
#                         data["fields"].append(key)
#                     data[episode][step][agent][key] = info[agent][key]
#
#         step += 1
#         done = terminated or truncated
#
# import pickle
# filename = f"training_data/{experiment_name}.pickle"
# with open(f"{path}/{filename}", "wb") as f:
#     pickle.dump(data, f)
#     print(f"training data saved in {filename}")

#####################################
# q-learning
#####################################

from qlearning import QLearningTrainee

learning_rate = 0.05
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

experiment_name = f"lumberjack-qlearning_{n_episodes}_{learning_rate}_{start_epsilon}_{epsilon_decay}_{final_epsilon}"

trainees = {}
for id in agent_ids:
    trainees[id] = QLearningTrainee(agent=id, action_space=env.action_space[id], learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

for episode in tqdm(range(n_episodes)):
    observations, info = env.reset()
    done = False

    data[episode] = {}

    step = 0
    while not done:
        actions = {}
        for id in agent_ids:
            actions[id] = trainees[id].get_action(observations[id])

        next_observations, rewards, terminated, truncated, info = env.step(actions)

        # collect data
        data[episode][step] = {}
        for id in agent_ids:
            data[episode][step][id] = {}
            data[episode][step][id]["reward"] = rewards[id] if id in rewards else 0
            if id in info:
                for key in info[id]:
                    if key not in data["fields"]:
                        data["fields"].append(key)
                    data[episode][step][id][key] = info[id][key]

        # update the agent
        for id in agent_ids:
            reward = rewards[id] if id in rewards else 0
            trainees[id].update(observations[id], actions[id], reward, terminated, next_observations[id])
        observations = next_observations

        # update if the environment is done and the current obs
        done = terminated or truncated
        step += 1

    for id in agent_ids:
        trainees[id].decay_epsilon()

import pickle
for trainee in trainees:
    filename = f"models/{agent_id_to_type[trainee]}_{trainee}_{experiment_name}.pickle"
    with open(f"{path}/{filename}", "wb") as f:
        pickle.dump(trainees[trainee].q_table(), f)
        print(f"trained model saved in {filename}")

import pickle
filename = f"data/{experiment_name}.pickle"
with open(f"{path}/{filename}", "wb") as f:
    pickle.dump(data, f)
    print(f"training data saved in {filename}")
