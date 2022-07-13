
import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):

    """
    A simple convolutional neural network
    Here you build implement a 3 layer CNN architecture
    You need to specify the detailed architecture in function "get_cnn_model" below
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        We can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------
        # self.convolutional_layers (list Conv1D) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------

        print('\n--------------------Neels tests---------------------\n')
        print('input_width', input_width)
        print('num_input_channels', num_input_channels)
        print('num_channels', num_channels)
        print('kernel_sizes', kernel_sizes)
        print('strides', strides)
        print('num_linear_neurons', num_linear_neurons)
        print('activations', activations)
        print('conv_weight_init_fn', conv_weight_init_fn)
        print('bias_init_fn', bias_init_fn)
        print('linear_weight_init_fn', linear_weight_init_fn)
        print('criterion', criterion)
        print('lr', lr, '\n------------------------------------------------')

        in_width = input_width #copy before modifying in loop
        for i in range(self.nlayers):
            out_width = (in_width - kernel_sizes[i]) // strides[i] + 1
            in_width = out_width

        channels = [num_input_channels] + num_channels

        self.convolutional_layers = [Conv1D(in_channel=channels[i], out_channel=channels[i+1], kernel_size=kernel_sizes[i], stride=strides[i], weight_init_fn=conv_weight_init_fn, bias_init_fn=bias_init_fn) for i in range(self.nlayers)]
        self.flatten = Flatten()
        self.linear_layer = Linear(in_feature = out_width*channels[-1], out_feature = num_linear_neurons, weight_init_fn=linear_weight_init_fn, bias_init_fn=bias_init_fn)


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """

        # Iterate through each layer
        # <---------------------
        input = np.copy(x) #can use x directly too

        # convolutions: affine combination -> activation ->
        for cl in range(self.nlayers):
            affine_combination = self.convolutional_layers[cl].forward(input)
            input = self.activations[cl].forward(affine_combination)
        # flattening
        input = self.flatten.forward(input)
        # linear layer
        # Save output (necessary for error and loss)
        self.output = self.linear_layer.forward(input)
        return self.output

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        #loss = criterion(actual, predicted), summed over resulting array
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()

        # Iterate through each layer in reverse order
        # <---------------------

        #linear layer
        grad = self.linear_layer.backward(grad)
        #flatten
        grad = self.flatten.backward(grad)
        #convolutional layers
        for i in reversed(range(self.nlayers)):
            grad_activation = self.activations[i].derivative()
            grad = grad * grad_activation
            grad = self.convolutional_layers[i].backward(grad)

        return grad


    def zero_grads(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
