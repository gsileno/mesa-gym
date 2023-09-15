from collections import defaultdict
import numpy as np

class QLearningTrainer:

    def __init__(self, agent, action_space, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon."""

        self.agent = agent
        self.action_space = action_space

        self.q_values = defaultdict(lambda: np.zeros(self.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def select_action(self, obs) -> int:
        """Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration."""

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.q_values[tuple(obs)]))

    def update(self, obs: tuple, action: int, reward: float, terminated: bool, next_obs: tuple):
        """Updates the Q-value of an action."""

        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[tuple(obs)][action]
        )

        self.q_values[tuple(obs)][action] = (
                self.q_values[tuple(obs)][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def q_table(self):
        return dict(self.q_values)