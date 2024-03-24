import os

path = os.path.dirname(os.path.abspath(__file__))

# load the target environment
import mesa_gym.gyms.grid.goal_world.env as w

env = w.MesaSacredWaterEnv(render_mode=None)

agents = []
type_agent = {}
for mesa_agent in env._get_agents():
    agents.append(mesa_agent.unique_id)
    type_agent[mesa_agent.unique_id] = type(mesa_agent).__name__

# target of training
n_episodes = 1000 # 10_000  # 100_000

# create the trainer instances

from tqdm import tqdm

data = {}
data["fields"] = []


def dqn_learning():
    from mesa_gym.trainers.DQN import DQNTrainer, device
    import torch
    from itertools import count

    replay_batch_size = 32 # 128
    learning_rate = 0.001  # 1e-4
    discount_factor = 0.95 # 0.99
    start_epsilon = 1.0    # 0.9
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1    # 0.05
    update_rate = 0.005

    experiment_name = f"goal_world-DQNlearning_{n_episodes}_{replay_batch_size}_{update_rate}_{learning_rate}_{discount_factor}_{start_epsilon}_{epsilon_decay}_{final_epsilon}"

    trainers = {}
    for agent in agents:
        trainers[agent] = DQNTrainer(agent=agent,
                                     observation_space=env.observation_space,
                                     action_space=env.action_space[agent],
                                     replay_batch_size=replay_batch_size,
                                     discount_factor=discount_factor,
                                     initial_epsilon=start_epsilon,
                                     final_epsilon=final_epsilon,
                                     epsilon_decay=epsilon_decay,
                                     update_rate=update_rate,
                                     learning_rate=learning_rate
                                     )

    for _ in tqdm(range(n_episodes)):
        observation, info = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        for _ in count():
            actions = {}
            reward_tensors = {}

            for agent in agents:
                actions[agent] = trainers[agent].select_action(state)

            observation, rewards, terminated, truncated, _ = env.step(actions)

            for agent in agents:
                reward = rewards[agent] if agent in rewards else 0
                reward_tensors[agent] = torch.tensor([reward], device=device)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            for agent in agents:
                trainers[agent].memory.push(state, actions[agent], next_state, reward_tensors[agent])

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            for agent in agents:
                trainers[agent].optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                trainers[agent].target_net_state_dict = trainers[agent].target_net.state_dict()
                policy_net_state_dict = trainers[agent].policy_net.state_dict()
                for key in policy_net_state_dict:
                    trainers[agent].target_net_state_dict[key] = policy_net_state_dict[key] * update_rate + \
                                                       trainers[agent].target_net_state_dict[key] * (1 - update_rate)
                trainers[agent].target_net.load_state_dict(trainers[agent].target_net_state_dict)

            if done:
                break

    # save models

    for agent in agents:
        trainer = trainers[agent]
        filename = f"models/{type_agent[agent]}_{agent}_{experiment_name}.pt"
        torch.save(trainer.policy_net.state_dict(), filename)
        print(f"trained model saved in {filename}")

    return experiment_name, trainers


######################################
# q-learning
######################################

def q_learning():
    from mesa_gym.trainers.qlearning import QLearningTrainer

    learning_rate = 0.05
    discount_factor = 0.95
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    experiment_name = f"goal_world-qlearning_{n_episodes}_{learning_rate}_{discount_factor}_{start_epsilon}_{epsilon_decay}_{final_epsilon}"

    trainers = {}
    for agent in agents:
        trainers[agent] = QLearningTrainer(agent=agent, action_space=env.action_space[agent],
                                           learning_rate=learning_rate, initial_epsilon=start_epsilon,
                                           discount_factor=discount_factor, epsilon_decay=epsilon_decay,
                                           final_epsilon=final_epsilon)

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        data[episode] = {}

        step = 0
        while not done:
            actions = {}
            for agent in agents:
                actions[agent] = trainers[agent].select_action(obs)
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
                trainers[agent].update(obs, actions[agent], reward, terminated, next_obs)
            obs = next_obs

            # update if the environment is done and the current obs
            done = terminated or truncated
            step += 1

        for agent in agents:
            trainers[agent].decay_epsilon()

    # save the Q-table
    import pickle

    for trainer in trainers:
        filename = f"models/{type_agent[trainer]}_{trainer}_{experiment_name}.pickle"
        with open(f"{path}/{filename}", "wb") as f:
            pickle.dump(trainers[trainer].q_table(), f)
            print(f"trained model saved in {filename}")

    return experiment_name, trainers

# experiment_name, trainers = dqn_learning()
experiment_name, trainers = q_learning()

# save data

import pickle

filename = f"data/{experiment_name}.pickle"
with open(f"{path}/{filename}", "wb") as f:
    pickle.dump(data, f)
    print(f"training data saved in {filename}")
