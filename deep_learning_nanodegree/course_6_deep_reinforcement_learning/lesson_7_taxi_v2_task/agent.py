import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0.005, alpha=0.05, gamma=.90):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action_uniformly(self, state):
        """ Given the state, select an action uniformly.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA)

    def get_epsilon_greedy_probabilities(self, state):
        """ returns the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1.0 - self.epsilon + (self.epsilon / self.nA)
        return policy_s


    def select_action(self, state):
        """ Given the state, select an action
        using the epsilon greedy probabilities.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA,
                                p=self.get_epsilon_greedy_probabilities(state))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Using the Expetected SARMA algorithm.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        probs = self.get_epsilon_greedy_probabilities(next_state)
        self.Q[state][action] = ((1.0 - self.alpha) * self.Q[state][action]
                                  + self.alpha * (reward
                                                  + self.gamma * np.dot(probs, self.Q[next_state])))