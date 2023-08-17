from mesa_gym.common.data_viz import DataViz

viz = DataViz("data/lumberjack-qlearning_selfishness_1000_0.05_1.0_0.002_0.1.pickle")
viz.show(["reward", "failure"])


