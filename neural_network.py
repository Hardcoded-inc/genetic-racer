from utils import sigmoid, relu, sigmoid_backward, relu_backward, convert_prob_into_class
import numpy as np
import pickle

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

        self.params_values = {}
        self.memory = {}

        self.X = None
        self.Y = None

        self.epochs = DEFAULT_EPOCHS
        self.learning_rate = learning_rate

        self.init_layers()




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


    def full_forward_propagation(self, X):
        self.memory = {}
        A_curr = X

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            self.memory["A" + str(idx)] = A_prev
            self.memory["Z" + str(layer_idx)] = Z_curr


        return A_curr



    def mse_loss(self, targets, y_pred):
        """
        Computes Mean Squared error/loss between targets
        and predictions.
        Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
            targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
        Returns: scalar
        Note: The averaging is only done over the output nodes and not over the samples in a batch.
        Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
        """
        return np.sum((y_pred - targets)**2) / y_pred.shape[1]

    def mse_loss_grad(self, targets, y_pred):
        """
        Computes mean squared error gradient between targets
        and predictions.
        Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
            targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
        Returns: (N,k) ndarray
        Note: The averaging is only done over the output nodes and not over the samples in a batch.
        Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
        """

        return 2 * (y_pred - targets) / y_pred.shape[1]


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




    def full_backward_propagation(self, Y, Y_hat):
        grads_values = {}

        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        # Replaced Cress-Entropy Loss with MSE_Loss
        # dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        dA_prev = self.mse_loss_grad(Y, Y_hat)

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = self.memory["A" + str(layer_idx_prev)]
            Z_curr = self.memory["Z" + str(layer_idx_curr)]
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
        for i in range(self.epochs):
            # step forward
            Y_hat, cashe = self.full_forward_propagation(self.X)

            # calculating metrics
            cost = self.get_cost_value(Y_hat)
            accuracy = self.get_accuracy_value(Y_hat)

            # step backward - calculating gradient
            grads_values = self.full_backward_propagation(self.Y, Y_hat, cashe)

            # updating model state
            self.update(grads_values)

            if(i % 50 == 0):
                if(self.verbose):
                    print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
                if(self.callback != None):
                    callback(i, params_values)


    def save(self, path):
        for field_name in [ "params_values", "memory", "nn_architecture"]:
            with open(f'{path}/{field_name}.pkl', 'wb') as outfile:
                data = getattr(self, field_name)
                pickle.dump(data, outfile)

    def load(self, path):
        for field_name in [ "params_values", "memory", "nn_architecture"]:
            with open(f'{path}/{field_name}.pkl', 'rb') as infile:
                data = pickle.load(infile)
                setattr(self, field_name, data)




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


