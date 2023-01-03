# simple script to run the zzt-like 2d grid world from gymnasium

import mesa_gym.envs.mesa_zzt_env
env = mesa_gym.envs.mesa_zzt_env.MesaZZTEnv(render_mode="human")

observation, info = env.reset()

for _ in range(1000):

    actions = {}
    for agent in env.agents:
        action = env.action_space[agent.unique_id].sample()
        actions[agent.unique_id] = action

    observation, reward, terminated, truncated, info = env.step(actions=actions)

    if terminated or truncated:
        observation, info = env.reset()

env.close()


