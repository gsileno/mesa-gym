from gymnasium.envs.registration import register

register(
     id="mesa_gym/GridWorld-v0",
     entry_point="mesa_gym.envs:GridWorldEnv",
     max_episode_steps=300,
)

register(
     id="mesa_gym/MesaZZT-v0",
     entry_point="mesa_gym.envs:MesaZZTEnv",
     max_episode_steps=1000,
)