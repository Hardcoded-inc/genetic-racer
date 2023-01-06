from game import Game
from neural_network import NeuralNetwork
from replay_memory import ReplayMemory
import math
import numpy
import random

game = Game()

# Model params
possible_actions = np.identity(game.actions_count, dtype=int).tolist()
states_size = [game.states_size]
actions_size = game.actions_count
learning_rate = 0.00025  # alpha

# Training params
total_episodes = 50000
max_steps = 5000 # for episode
batch_size = 64 # TODO: describe
max_tau = 10000  # Tau is the step where we update our target network

# Epsilon greedy strategy params
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Q-learning params
gamma = 0.95  # Discounting rate

# ReplayMemory params
memory_size = 100000  # Number of experiences the ReplayMemory can keep

# TODO: huh? vvvvv
pretrain_length = memory_size  # Number of experiences stored in the ReplayMemory when initialized for the first time


# WATCH THE TRAINED AGENT
# ---
# training =  False


# RENDER THE ENVIRONMENT
# ---
# episode_render = True
# load = True
# starting_episode = 0
# load_traing_model = False
# load_training_model_number = 300


nn_architecture = [
    {"input_dim": states_size, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": actions_size, "activation": "sigmoid"},
]

# Create NN instances
DQNetwork = NeuralNetwork(learning_rate, nn_architecture)
TargetNetwork = NeuralNetwork(learning_rate, nn_architecture)

# Create replay memory
memory = ReplayMemory(memory_size)

# Render the environment
game.new_episode()


