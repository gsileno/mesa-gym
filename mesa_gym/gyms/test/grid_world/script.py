# simple script to run the gridworld example from gymnasium

# import gymnasium as gym
# env = gym.make('mesa_gym/GridWorld-v0', render_mode="human")

import mesa_gym.gyms.test.grid_world.world as w
env = w.GridWorldEnv(render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()


