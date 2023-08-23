import os
path = os.path.dirname(os.path.abspath(__file__))

# load the target environment
import mesa_gym.gyms.grid.goal_world.env as w
env = w.MesaGoalEnv(render_mode=None)

import gymnasium as gym

agents = []
type_agent = {}
for mesa_agent in env._get_agents():
    agents.append(mesa_agent.unique_id)
    type_agent[mesa_agent.unique_id] = type(mesa_agent).__name__

# take the number of dimensions
nb_states = gym.spaces.flatdim(env.observation_space)
print(f"number of states: {nb_states}")

# target of training
n_episodes = 10_000  # 100_000

# create the trainee agent instances

from tqdm import tqdm

data = {}
data["fields"] = []

######################################
# q-learning
######################################

def q_learning():
    from mesa_gym.trainees.qlearning import QLearningTrainee

    learning_rate = 0.05
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1
    discount_factor = 0.95

    experiment_name = f"goal_world-qlearning_{n_episodes}_{learning_rate}_{discount_factor}_{start_epsilon}_{epsilon_decay}_{final_epsilon}"

    trainees = {}
    for agent in agents:
        trainees[agent] = QLearningTrainee(agent=agent, action_space=env.action_space[agent], learning_rate=learning_rate, initial_epsilon=start_epsilon, discount_factor=discount_factor, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

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
                        if key not in data["fields"]:
                            data["fields"].append(key)
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

    return experiment_name, trainees


experiment_name, trainees = q_learning()

import pickle
for trainee in trainees:
    filename = f"models/{type_agent[trainee]}_{trainee}_{experiment_name}.pickle"
    with open(f"{path}/{filename}", "wb") as f:
        pickle.dump(trainees[trainee].q_table(), f)
        print(f"trained model saved in {filename}")

import pickle
filename = f"data/{experiment_name}.pickle"
with open(f"{path}/{filename}", "wb") as f:
    pickle.dump(data, f)
    print(f"training data saved in {filename}")
