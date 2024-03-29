from neural_network import NeuralNetwork
from collections import deque
from time import sleep
import numpy as np
import math
import random
import os

class QLAgent:
    def __init__(self, game, name):

        self.name = name
        self.game = game
        self.pretrained = False

        # ------------------------ #
        #       Model params       #
        # ------------------------ #
        self.possible_actions = np.identity(game.actions_count, dtype=int)
        self.state_size = game.state_size
        self.actions_size = game.actions_count
        self.learning_rate = 0.00025

        # ------------------------ #
        #      Training params     #
        # ------------------------ #
        self.current_episode = 0
        self.total_episodes = 100000
        self.max_steps = 5000           # for episode
        self.max_tau = 10000            # Tau is the step where we update our target network

        self.batch_size = 64
        self.memory_size = 100000       # Number of experiences the ReplayMemory can keep

        self.pretrain_length = 5 * self.batch_size      # Number of experiences collected before training

        self.autosave_freq = 500
        self.save_dir_path = "./models"

        # ------------------------ #
        #     Q-learning params    #
        # ------------------------ #
        self.gamma = 0.95                # Discounting rate
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
            {"input_dim": self.state_size, "output_dim": 36, "activation": "relu"},
            {"input_dim": 36, "output_dim": 36, "activation": "relu"},
            {"input_dim": 36, "output_dim": 36, "activation": "relu"},
            {"input_dim": 36, "output_dim": self.actions_size, "activation": "sigmoid"},
        ]

        self.dq_network = NeuralNetwork(self.learning_rate, self.nn_architecture)
        self.target_network = NeuralNetwork(self.learning_rate, self.nn_architecture)

        self.replay_memory = ReplayMemory(self.memory_size)


        print("QL Agent initilized")

    def update_target_network_params(self):
        # Copy NN params from DQ-Network to Target Network
            buff = getattr(self.dq_network, "params_values")
            setattr(self.target_network, "params_values", buff)

    def pretrain(self):
        print("Start pretraining...")

        state = []
        new_episode = False
        step = 0

        def step_function():
        # for step in range(self.pretrain_length):
            nonlocal step
            nonlocal state

            if step % 10 == 0:
                print(f"[Pre-Training] Step {step}")

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
        print("✅ Pre-Training finished!")




    def train(self):
        if self.current_episode == 0:
            print("Starting training...")
        else:
            print("Resuming training...")

        tau = 0
        state = []
        step = 0
        training_step = 0

        def step_function():
            nonlocal tau
            nonlocal state
            nonlocal step
            nonlocal training_step

            if training_step == 0:
                state = self.game.get_state()

            if step == 0:
                percentage = self.current_episode / self.total_episodes * 100
                print(f"\n[Training] Episode: {self.current_episode}, {percentage}%")


            if step < self.max_steps:
                step += 1
                self.decay_step += 1
                training_step += 1
                tau += 1

                # choose best action if not exploring choose random otherwise

                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                    -self.decay_rate * self.decay_step)

                if(step == 1): print(f"Epsilon: {epsilon}")

                action = None
                if np.random.rand() < epsilon:
                    action = random.choice(self.possible_actions)
                    action_no = np.argmax(action)

                else:
                    computing_input_data = np.transpose(np.array([state]))
                    action_q_values = self.dq_network.full_forward_propagation(computing_input_data)
                    action = action_q_values
                    action_no = np.argmax(action)

                # now we need to get next state
                reward = self.game.make_action(action_no)

                next_state = self.game.get_state()

                if (reward > 0):
                    print(f"   > ⛩️  Gate  |  Reward: {reward}")
                    # print(f"\nstate: {state}")
                    # print(f"\naction: {action}")

                # if car is dead then finish episode
                if self.game.is_episode_finished():
                    reward = -100
                    step = self.max_steps
                    print(f"   > 💀 DEAD  |  Reward: {reward}\n")


                # add the experience to the memory buffer
                self.replay_memory.store((state, action, reward, next_state, self.game.is_episode_finished()))

                state = next_state

                # ---------
                # Learning!
                # first we are gonna need to grab a random batch of experiences from out memory

                replay_memory_samples = np.array(self.replay_memory.sample(self.batch_size), dtype=object)

                exp = np.transpose(replay_memory_samples)

                states_batch = exp[0]
                actions_batch = exp[1]
                rewards_batch = exp[2]
                next_states_batch = exp[3]
                dies_batch = exp[4]

                target_qs_from_batch = []


                # compute q values for current state of each experience in the batch

                input_data = np.transpose(np.stack(states_batch, axis=0))
                action_q_values_current_state = self.dq_network.full_forward_propagation(input_data)
                action_q_values_current_state = np.transpose(action_q_values_current_state)


                # predict the q values of the next state for each experience in the batch

                input_data = np.transpose(np.stack(next_states_batch, axis=0))
                action_q_values_next_states = self.target_network.full_forward_propagation(input_data)
                action_q_values_next_states = np.transpose(action_q_values_next_states)


                for i in range(self.batch_size):
                    action_no = np.argmax(action_q_values_next_states[i])  # double DQN
                    terminal_state = dies_batch[i]
                    if terminal_state:
                        target_qs_from_batch.append(rewards_batch[i])
                    else:
                        # The Bellman equation
                        target = rewards_batch[i] + self.gamma * action_q_values_next_states[i][action_no]  # double DQN
                        target_qs_from_batch.append(target)

                targets_for_batch = np.array([t for t in target_qs_from_batch])

                output_q_values = []
                for i in range(self.batch_size):
                    action_no = np.argmax(actions_batch[i])
                    value = action_q_values_current_state[i][action_no]
                    output_q_values.append(value)


                # if(step == self.max_steps):
                #     loss = self.dq_network.mse_loss(np.array([targets_for_batch]), np.array([output_q_values]))
                #     print(f"Current loss: {loss}")

                # step backward - calculating gradient
                grads_values = self.dq_network.full_backward_propagation(np.array([targets_for_batch]), np.array([output_q_values]))
#
                # updating model state
                self.dq_network.update(grads_values)


            if tau > self.max_tau:
                self.update_target_network_params()
                print("🗄️ Target Network Updated")
                tau = 0

            if step >= self.max_steps:
                self.replay_memory.store((state, action, reward, next_state, True))

                step = 0
                self.current_episode += 1
                self.game.new_episode()
                state = self.game.get_state()

                if self.current_episode >= self.total_episodes:
                    self.training = False

                if self.current_episode % self.autosave_freq == 0:
                    self.save_model(self.current_episode)
                    print("💽 Model Saved")
            else:
                self.replay_memory.store((state, action, reward, next_state, False))
                state = next_state

            self.game.clock.tick()
            return self.training



        self.game.run_for_agent(step_function)
        print("✅ Training finished!")



    def save_model(self, episode_no):
        directory = f"{self.save_dir_path}/{self.name}/model{episode_no}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(f"{directory}/agent_params.txt", 'w') as file:
            content = self.decay_step
            file.write(str(content))

        self.dq_network.save(directory)


    def load_model(self, model_name, episode_no):
        directory = f"{self.save_dir_path}/{self.name}/model{episode_no}"
        if os.path.exists(directory):
            self.dq_network.load(directory)
            self.target_network.load(directory)
            self.current_episode = episode_no

            with open(f"{directory}/agent_params.txt", 'r') as file:
                content = file.read()
                self.decay_step = int(content)

            print(f"✅ {model_name} model [Episde: {episode_no}] loaded")
        else:
            print("⛔️ No such model:episode checkpoint")


#     def test(self):
#
#         state = self.game.get_state()
#
#         QValues = self.sess.run(self.dq_network.output,
#                                 feed_dict={self.dq_network.inputs_: np.array([state])})
#         choice = np.argmax(QValues)
#         action = self.possible_actions[choice]
#
#         action_no = np.argmax(action)
#         # now we need to get next state
#         self.game.make_action(action_no)
#
#         if self.game.is_episode_finished():
#             self.game.new_episode()


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
