import numpy as np
from random import random


# save activation and derivatives
# implement backpropagation
# implement gradient descent
# implement train
# train our net wit some dummy dataset
# make some prediction


class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_lnputs=3, num_hidden=[3, 3], num_output=1):

        """" Constructor for the MLP. Takes the number of input,
        variable of hidden layers, and number of outputs

        Args:
            Number of inputs: the value is integers

            num_lnputs (int) : Number of inputs
            hidden layer (list) : A list of interger for hidden layer
            output layer(int): number of output layer
        """

        self.num_inputs = num_lnputs
        self.num_hidden = num_hidden
        self.num_output = num_output

        # creating generic representation of layers

        layers = [num_lnputs] + num_hidden + [num_output]

        # crating the random weights

        weight = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weight.append(w)
        self.weight = weight

        # activation representation
        activation = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation.append(a)

        self.activation = activation

        # derivative represation

        derivative = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivative.append(d)

        self.derivative = derivative

    def forward_propagation(self, inputs):
        """ compute the forward propagation based on input

        Args:
            inputs(ndarray) : Input Array

        Return:
            activation(ndarray) : Output Array
        """

        # the inputs layer activation is just input layer itself
        activation = inputs

        # save activation for the backpropagation
        self.activation[0] = inputs

        # iterative to network of layer

        for i, w in enumerate(self.weight):
            # calculting matrix matipulation of between previous activation and weight matrix
            net_inputs = np.dot(activation, w)

            # applying the sigmoid activation function
            activation = self.sigmoid(net_inputs)

            # save activation for backpropagation
            self.activation[i + 1] = activation

        # return output layer activation

        return activation

    def back_propagate(self, error, verbose=False):

        # dE/dW_i = (y-a_[i+1]) s'(h_[i+1)) a_i
        # s'(h_[i+1] = s(h_i+1](1-s(h_[i+1]))
        # s(h_[i+1)) = a_[i+1]

        # dE/dW_[i-1] = (y-a_[i+1]) s'(h_[i+1])) w_i s'(h_i) a_[i-1]

        # Iterate back through the network layers
        for i in reversed(range(len(self.derivative))):

            # get the actiation for previous layers
            activation = self.activation[i + 1]

            # apply the sigmoid the derivative function
            delta = error * self.sigmoid_derivative(activation)

            delta_reshaped = delta.reshape(delta.shape[0], -1).T  # --> ndarray ([0.1,0.2]) into ([[0.1,0.2]])

            current_activation = self.activation[i]
            current_activation = current_activation.reshape(current_activation.shape[0], -1)
            # current_activation = current_activation.reshape(-1,1)

            # save the derivative after applying the matrix multiplication
            self.derivative[i] = np.dot(current_activation, delta_reshaped)

            # backprogate to next error
            error = np.dot(delta, self.weight[i].T)

            if verbose:
                print("Derivatives for W{} : {}".format(i, self.derivative[i]))

        # return error

    def gradient_desent(self, learning_rate=1):
        for i in range(len(self.weight)):
            weight = self.weight[i]
            # print("Original W{} {}".format(i,weight))
            derivative = self.derivative[i]
            weight += derivative * learning_rate
            # print("Updated W{} {}".format(i,weight))

    def train(self, inputs, target, epochs, learning_rate):

        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """

        # now entering the training loop
        for i in range(epochs):
            sum_error = 0

            for data_input, data_target in zip(inputs, target):
                # forward propagation
                output = self.forward_propagation(data_input)

                # calculate error

                error = data_target - output

                # back propagation

                self.back_propagate(error)

                # applying gradient decent

                self.gradient_desent(learning_rate=0.1)

                sum_error += self.mse(data_target, output)

            # report error

            print("Error: {} at epoch {}".format(sum_error / len(inputs), i + 1))
        print()
        print("Training Completed")
        print("===========")

    def mse(self, target, output):
        return np.average((target - output) ** 2)

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def sigmoid(self, x):
        y = 1.0 / (1.0 + np.exp(-x))

        return y


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])  # array ([[0.1,0.2], [0.3,0.4]])
    target = np.array([[i[0] + i[1]] for i in inputs])  # array ([[0.3], [0.7]])

    # create an mlp
    mlp = MLP(2, [5], 1)

    # create dummy data
    # inputs = np.array([0.1, 0.2])
    # target = np.array([0.3])

    # train our mlp
    mlp.train(inputs, target, 50, 0.1)

    # create dummy data

    inputs_ck = np.array([0.3, 0.1])
    target_ck = np.array([0.4])

    model_output = mlp.forward_propagation(inputs_ck)
    print()
    print()

    print("Our network believes that {} + {} is equal to {}".format(inputs_ck[0], inputs_ck[1], model_output[0]))
