# Import Libray

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transform

# Hyper-parameters

epochs = 2
batch_size = 100
Learning_rate = 0.01


class MLP(object):

    def __init__(self, num_inputs, number_hidden_layers, num_output):
        self.num_inputs = num_inputs
        self.hidden_layers = number_hidden_layers
        self.num_output = num_output

        layers = [num_inputs] + number_hidden_layers + [num_output]
        print(layers)

        # Creating the random weights and bias

        weight = []
        bias = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            b = np.random.rand(1,layers[i+1])

            weight.append(w)
            bias.append(b)

        self.weight = weight
        self.bias = bias

        # Creating the Activation

        activation = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation.append(a)

        self.activation = activation

        # Derivative Representation

        derivative = []

        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivative.append(d)

        self.derivative = derivative

    def forward_propagation(self, inputs):

        activation = inputs

        self.activation[0] = inputs

        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            net_input = np.dot(activation,w)
            net_input_bias = net_input + b

            # Sigmoid derivative

            activation = self.sigmoid(net_input_bias)

            # save Activation

            self.activation[i + 1] = activation
        return activation

    def backward_propagation(self, error,Learning_rate = 1, verbose=False):

        # Iterate back through the network layers

        for i in reversed(range(len(self.derivative))):
            # Get the activation for the previous layers
            activation = self.activation[i + 1]

            # Apply the sigmoid derivative function
            sigmoid_derivative = self.sigmoid_derivative(activation)

            # calculate the delta

            delta = error * sigmoid_derivative

            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activation = self.activation[i]
            current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1).T

            # save derivative after the matric multiplication

            self.derivative[i] = np.dot(current_activation_reshaped, delta)

            # Update the weight and bias

            self.weight[i] += self.derivative[i] * Learning_rate

            if isinstance(delta,torch.Tensor):
                delta = delta.detach().numpy()

            #delta_mean = delta.mean(axis= 0)

            self.bias[i]  += delta.mean() * Learning_rate

            # Backpropagation to next error

            error = np.dot(delta,self.weight[i].T)

            if verbose:
                print("Derivatives for W{} :{}".format(i,self.derivative[i]))

    def train(self,data,epochs,learning_rate):

        for i in range(epochs):

            sum_error = 0

            for j, (inputs,target) in enumerate(data):

                image = inputs.reshape(-1, 28*28)

                output = self.forward_propagation(image)

                target = target.reshape(target.shape[0],-1)

                error = target - output

                self.backward_propagation(error)

                sum_error += self.mse(target,output)

                print("Error :  {} at epochs {}".format(sum_error / len(inputs),j+1))
        print()
        print("Training Completed")


    def mse(self,target,output):
        return np.average((target-output) **2)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)


if __name__ == "__main__":
    print("MLP with MNIST dataset")
    print(type(epochs))

    # Data Preparation
    train_data = torchvision.datasets.MNIST(root="\data", train=True,
                                            transform=transform.ToTensor(),
                                            download=True)

    test_data = torchvision.datasets.MNIST(root="\data",
                                           train=False,
                                           transform=transform.ToTensor())

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data
                                                   , batch_size=batch_size,
                                                   shuffle=True)

Model = MLP(784, [2], 1)

Model.train(train_data_loader, epochs,Learning_rate)
