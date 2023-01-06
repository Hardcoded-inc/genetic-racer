from utils import sigmoid, relu, sigmoid_backward, relu_backward, convert_prob_into_class
import numpy as np

DEFAULT_SEED = 99
DEFAULT_EPOCHS = 10000
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

class NeuralNetwork:
    def __init__(self, learning_rate=DEFAULT_LEARNING_RATE, nn_architecture=DEFAULT_NN_ARCHITECTURE):
        self.verbose = True
        self.callback = None

        self.seed = DEFAULT_SEED
        self.nn_architecture = nn_architecture

        self.cost_history = []
        self.accuracy_history = []
        self.params_values = {}

        self.X = None
        self.Y = None

        self.epochs = DEFAULT_EPOCHS
        self.learning_rate = learning_rate



    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        # calculation of the input value for the activation function
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            activation_func = relu
        elif activation == "sigmoid":
            activation_func = sigmoid
        else:
            raise Exception('Non-supported activation function')

        # return of calculated activation A and the intermediate Z matrix
        return activation_func(Z_curr), Z_curr


    def full_forward_propagation(self):
        memory = {}
        A_curr = self.X

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory




    def calculate_loss(self):
        # To implement
        # Calculate loss between output Q-values and target Q-values.
        # Requires a second pass to the network for the next state

        # (Can use second NN for the stbility of the optimization)

        pass



    def get_cost_value(self, Y_hat):
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(self.Y, np.log(Y_hat).T) + np.dot(1 - self.Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    def get_accuracy_value(self, Y_hat):
        Y_hat_ = convert_prob_into_class(Y_hat)
        return (Y_hat_ == self.Y).all(axis=0).mean()




    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]

        if activation == "relu":
            backward_activation_func = relu_backward
        elif activation == "sigmoid":
            backward_activation_func = sigmoid_backward
        else:
            raise Exception('Non-supported activation function')

        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr




    def full_backward_propagation(self, Y_hat, memory):
        grads_values = {}
        m = self.Y.shape[1]
        self.Y = self.Y.reshape(Y_hat.shape)

        dA_prev = - (np.divide(self.Y, Y_hat) - np.divide(1 - self.Y, 1 - Y_hat));

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = self.params_values["W" + str(layer_idx_curr)]
            b_curr = self.params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values



    def update(self, grads_values):
        for layer_idx, layer in enumerate(self.nn_architecture, 1):
            self.params_values["W" + str(layer_idx)] -= self.learning_rate * grads_values["dW" + str(layer_idx)]
            self.params_values["b" + str(layer_idx)] -= self.learning_rate * grads_values["db" + str(layer_idx)]

    def init_layers(self):
        np.random.seed(self.seed)
        self.params_values = {}

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            self.params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1

    def train(self):
        self.cost_history = []
        self.accuracy_history = []


        for i in range(self.epochs):
            # step forward
            Y_hat, cashe = self.full_forward_propagation()

            # calculating metrics and saving them in history
            cost = self.get_cost_value(Y_hat)
            self.cost_history.append(cost)

            accuracy = self.get_accuracy_value(Y_hat)
            self.accuracy_history.append(accuracy)

            # step backward - calculating gradient
            grads_values = self.full_backward_propagation(Y_hat, cashe)

            # updating model state
            self.update(grads_values)

            if(i % 50 == 0):
                if(self.verbose):
                    print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
                if(self.callback != None):
                    callback(i, params_values)

        # return self.params_values, self.cost_history, self.accuracy_history




# ======================== #
#       Example usage      #
# ======================== #
#
# nn = NeuralNetwork()
#
# nn.seed = 2
# nn.nn_architecture = DEFAULT_NN_ARCHITECTURE
# nn.init_layers()
#
#
#
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
#
# # number of samples in the data set
# N_SAMPLES = 1000
# # ratio between training and test sets
# TEST_SIZE = 0.1
#
# X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
#
# nn.X = np.transpose(X_train)
# nn.Y = np.transpose(y_train.reshape((y_train.shape[0], 1)))
#
# res = nn.train()


