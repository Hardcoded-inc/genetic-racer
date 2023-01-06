from game import Game
from neural_network import NeuralNetwork
import math
import numpy
import random


class QLearning:
    def __init__(self, game):

        self.game = Game()
        self.game.new_episode()


        # ------------------------ #
        #       Model params       #
        # ------------------------ #
        possible_actions = np.identity(game.actions_count, dtype=int)
        self.states_size = [game.states_size]
        self.actions_size = game.actions_count
        self.learning_rate = 0.00025

        # ------------------------ #
        #      Training params     #
        # ------------------------ #
        self.total_episodes = 100000
        self.max_steps = 5000           # for episode
        self.max_tau = 10000            # Tau is the step where we update our target network

        self.batch_size = 64
        self.memory_size = 100000       # Number of experiences the ReplayMemory can keep

        self.pretrain_length = 64       # Number of experiences collected before training


        # ------------------------ #
        #     Q-learning params    #
        # ------------------------ #
        self.gamma = 0.9                # Discounting rate
        self.training = True


        # -------------------------------- #
        #  Epsilon greedy strategy params  #
        # -------------------------------- #
        self.max_epsilon = 1            # Max Exploration rate
        self.min_epsilon = 0.01         # Min Exploration_rate
        self.decay_rate = 0.00001
        self.decay_step = 0




        # ------------------------ #
        #          Setup           #
        # ------------------------ #
        self.nn_architecture = [
            {"input_dim": states_size, "output_dim": 256, "activation": "relu"},
            {"input_dim": 256, "output_dim": 256, "activation": "relu"},
            {"input_dim": 256, "output_dim": 256, "activation": "relu"},
            {"input_dim": 256, "output_dim": 256, "activation": "relu"},
            {"input_dim": 256, "output_dim": actions_size, "activation": "sigmoid"},
        ]

        self.dq_network = NeuralNetwork(self.learning_rate, self.nn_architecture)
        self.target_network = NeuralNetwork(self.learning_rate, self.nn_architecture)

        self.replay_memory = ReplaydMemory(self.memory_size)




class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]

