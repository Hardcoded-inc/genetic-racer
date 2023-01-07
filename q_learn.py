from neural_network import NeuralNetwork
from collections import deque
from time import sleep
import numpy as np
import math
import random
import os

class QLAgent:
    def __init__(self, game):

        self.game = game
        self.pretrained = False

        # ------------------------ #
        #       Model params       #
        # ------------------------ #
        self.possible_actions = np.identity(game.actions_count, dtype=int)
        self.state_size = [game.state_size]
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

        self.pretrain_length = 50      # Number of experiences collected before training

        self.autosave_freq = 1000
        self.save_dir_path = "./models"

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
        #        Setup NNs         #
        # ------------------------ #
        self.nn_architecture = [
            {"input_dim": self.state_size, "output_dim": 24, "activation": "relu"},
            {"input_dim": 24, "output_dim": 24, "activation": "relu"},
            {"input_dim": 24, "output_dim": self.actions_size, "activation": "sigmoid"},
        ]

        self.dq_network = NeuralNetwork(self.learning_rate, self.nn_architecture)
        self.target_network = NeuralNetwork(self.learning_rate, self.nn_architecture)

        self.replay_memory = ReplayMemory(self.memory_size)


        print("QL Agent initilized")

    def update_target_network_params(self):
        # Copy NN params from dq_n to target_n
        for fieldname in ["cost_history", "accuracy_history", "params_values"]:
            buff = getattr(dq_network, fieldname)
            setattr(target_network, fieldname, buff)

    def pretrain(self):
        print("Start pretraining...")

        state = []
        new_episode = False
        step = 0

        def step_function():
        # for step in range(self.pretrain_length):
            nonlocal step
            nonlocal state
            nonlocal new_episode

            print(f"step {step}")
            if step == 0:
                state = self.game.get_state()

            # Pick a random movement and do it to populate the memory thing
            action = random.choice(self.possible_actions)
            action_no = np.argmax(action)

            # Get next
            reward = self.game.make_action(action_no)
            next_state = self.game.get_state()

            if self.game.is_episode_finished():
                reward = -100
                new_episode = True
                self.replay_memory.store((state, action, reward, next_state, True))
                self.game.new_episode()
                state = self.game.get_state()
            else:
                self.replay_memory.store((state, action, reward, next_state, False))
                state = next_state

            step += 1
            self.game.clock.tick()
            return step < self.pretrain_length


        self.game.run_for_agent(step_function)
        self.update_target_network_params()
        print("Pre-Training finished!")




    def train(self):
        tau = 0
        state = []
        step_no = 0
        training_step_no = 0
        episode_no = 0
        new_episode = False

        def step_function():
            if training_step_no == 0:
                state = self.game.get_state()

            if new_episode:
                state = self.game.get_state()

            if step_no < self.max_steps:
                step_no += 1
                self.decay_step += 1
                training_step_no += 1
                tau += 1

                # choose best action if not exploring choose random otherwise

                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                    -self.decay_rate * self.decay_step)

                if np.random.rand() < epsilon:
                    choice = random.randint(1, len(self.possible_actions)) - 1
                    action = self.possible_actions[choice]

                else:
                    # TODO
                    q_values = self.sess.run(self.dq_network.output,
                                            feed_dict={self.dq_network.inputs_: np.array([state])})
                    choice = np.argmax(q_values)
                    action = self.possible_actions[choice]

                action_no = np.argmax(action)
                # now we need to get next state
                reward = self.game.make_action(action_no)

                next_state = self.game.get_state()

                if (reward > 0):
                    print(f"Hell YEAH, Reward = {reward}")
                # if car is dead then finish episode
                if self.game.is_episode_finished():
                    reward = -100
                    step_no = self.max_steps
                    print("DEAD!! Reward =  -100")

                # print("Episode {} Step {} Action {} reward {} epsilon {} experiences stored {}"
                #       .format(episode_no, step_no, action_no, reward, epsilon, training_step_no))

                # add the experience to the memory buffer
                self.replay_memory.store((state, action, reward, next_state, self.game.is_episode_finished()))

                state = next_state


            if tau > self.max_tau:
                self.update_target_network_params()
                print("Target Network Updated")
                tau = 0

            if step_no >= self.max_steps:
                episode_no += 1
                step_no = 0
                new_episode = True
                self.game.new_episode()
                if episode_no >= self.total_episodes:
                    self.training = False

                if episode_no % self.autosave_freq == 0:
                    self.save_model()
                    print("Model Saved")

            self.game.clock.tick()


        self.game.run_for_agent(step_function)
        print("Training finished!")


def save_model(self):
    directory = f"{self.save_dir_path}/model{episode_no}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # TODO
    # What to save?
    # self.target_network.params_values
    # save(f"{self.save_dir_path}/model{episode_no}/model.ckpt")


        print("Pre-Training finished!")


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]

