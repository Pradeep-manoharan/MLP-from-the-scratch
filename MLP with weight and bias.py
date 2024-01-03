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

    def __init__(self, Num_input=3, num_hidden=[3, 3], num_output=1):

        """" Constructor for the MLP. Takes the number of input,
        variable of hidden layers, and number of outputs

        Args:
            Number of inputs: the value is integers

            Num_input (int) : Number of inputs
            hidden layer (list) : A list of integer for hidden layer
            output layer(int): number of output layer
        """

        self.num_inputs = Num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        # creating generic representation of layers

        layers = [Num_input] + num_hidden + [num_output]

        # crating the random weights

        weight = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weight.append(w)
        self.weight = weight

        # crating the random bias

        bias = []



        for i in range(len(layers) - 1):
            b = np.random.rand(1, layers[i + 1])
            bias.append(b)
        self.bias = bias

        # activation representation
        activation = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation.append(a)

        self.activation = activation

        # derivative representation of b

        derivative_b = []
        for i in range(len(layers) - 1):
            db = np.zeros((1, layers[i + 1]))
            derivative_b.append(db)

        self.derivative_b = derivative_b

        # derivative representation of w

        derivative_w = []
        for i in range(len(layers) - 1):
            dw = np.zeros((layers[i], layers[i + 1]))
            derivative_w.append(dw)

        self.derivative_w = derivative_w

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

        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            # calculating matrix manipulation of between previous activation and weight matrix

            net_inputs = np.dot(activation, w)
            net_inputs_b = net_inputs + b
            net_input_b_reshape = np.ravel(net_inputs_b)

            # applying the sigmoid activation function
            activation = self.sigmoid(net_input_b_reshape)

            # save activation for backpropagation
            self.activation[i + 1] = activation

        # return output layer activation

        return activation

    def back_propagate(self, error, verbose=False):

         # dE/dW_i = (y-a_[i+1]) s'(h_[i+1)) a_i
        # s'(h_[i+1] = s(h_i+1](1-s(h_[i+1]))
        # s(h_[i+1)) = a_[i+1]

        # dE/dW_[i-1] = (y-a_[i+1]) s'(h_[i+1]) w_i s'(h_i) a_[i-1]

        # Iterate back through the network layers
        for i in reversed(range(len(self.derivative_w))):
            # get the activation for previous layers
            activation = self.activation[i + 1]

            # apply the sigmoid the derivative function
            delta = error * self.sigmoid_derivatives(activation)

            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            # --> ndarray ([0.1,0.2]) into ([[0.1,0.2]])

            current_activation = self.activation[i]
            current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1)


            # save the derivative after applying the matrix multiplication

            self.derivative_w[i] = np.dot(current_activation_reshaped, delta_reshaped)

            self.derivative_b[i] = delta
            # backpropagation to next error
            error = np.dot(delta, self.weight[i].T)
        return error


        # for i in reversed(range(len(self.bias))):
        #
        #     # dE/dW_i = (y-a_[i+1]) s'(h_[i+1))
        #     # s'(h_[i+1] = s(h_i+1](1-s(h_[i+1]))
        #     # s(h_[i+1)) = a_[i+1]
        #
        #     # dE/dW_[i-1] = (y-a_[i+1]) s'(h_[i+1]) w_i s'(h_i)
        #
        #     # get the activation for previous layers
        #     activation = self.activation[i + 1]
        #
        #     # apply the sigmoid the derivative function
        #     delta = error * self.sigmoid_derivatives(activation)
        #
        #     # save the derivative after applying the matrix multiplication
        #     self.derivative_b[i] = delta
        #
        #     # backpropagation to next error
        #     error = np.dot(delta, self.weight[i].T)

        if verbose:
                print("Derivatives for W{} : {}".format(i, self.derivative[i]))

        # return error

    def gradient_decent(self, learning_rate=1):
        for i in range(len(self.weight)):
            weight = self.weight[i]
            # print("Original W{} {}".format(i,weight))
            derivative = self.derivative_w[i]
            weight += derivative * learning_rate
            # print("Updated W{} {}".format(i,weight))
        for i in range(len(self.bias)):
            bias = self.bias[i]
            # print("Original W{} {}".format(i,weight))
            derivative = self.derivative_b[i]
            bias += derivative * learning_rate
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

                self.gradient_decent(learning_rate=0.1)

                sum_error += self.mse(data_target, output)

            # report error

            print("Error: {} at epoch {}".format(sum_error / len(inputs), i + 1))
        print()
        print("Training Completed")
        print("===========")

    def mse(self, target, output):
        return np.average((target - output) ** 2)

    def sigmoid_derivatives(self, x):
        return x * (1.0 - x)

    def sigmoid(self, x):
        y = 1.0 / (1.0 + np.exp(-x))

        return y


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    print("MLP with weight and bias")
    print()
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])  # array ([[0.1,0.2], [0.3,0.4]])
    target = np.array([[i[0] + i[1]] for i in inputs])  # array ([[0.3], [0.7]])

      # create a mlp
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
