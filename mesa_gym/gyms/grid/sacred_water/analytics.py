from mesa_gym.common.data_viz import DataViz

viz = DataViz("data/goal-qlearning_1000_0.05_1.0_0.002_0.1.pickle")
# viz = DataViz("data/zzt-random_1000.pickle")
viz.show(["reward", "success"])


