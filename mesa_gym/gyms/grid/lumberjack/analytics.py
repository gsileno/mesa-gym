from mesa_gym.common.data_viz import DataViz

# viz = DataViz("data/lumberjack-qlearning_selfishness_1000_0.05_1.0_0.002_0.1.pickle", "per_step")
# viz.show(["reward", "failure"])

viz = DataViz("data/lumberjack-qlearning_selfishness_1000_0.05_1.0_0.002_0.1.pickle", "per_episode")
viz.show(["reward", "n_steps"])

# viz = DataViz("data/lumberjack-qlearning_selfishness_1000_0.05_1.0_0.002_0.1.pickle", "per_episode_group")
# viz.show(["reward", "n_steps"])

