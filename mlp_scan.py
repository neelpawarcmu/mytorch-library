import numpy as np
import os
import sys

from numpy.core.fromnumeric import transpose

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        # scanning over one conv filter
        self.conv1 = Conv1D(in_channel=24, out_channel=8, kernel_size=8, stride=4)
        self.conv2 = Conv1D(in_channel=8, out_channel=16, kernel_size=1, stride=1)
        self.conv3 = Conv1D(in_channel=16, out_channel=4, kernel_size=1, stride=1)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.flatten = Flatten()

        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.flatten]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights

        def convert_weight_shapes(w, shape):
            transposed_weights = w.T
            reshaped_weights = transposed_weights.reshape(shape)
            transposed_weights = reshaped_weights.transpose(0,2,1)
            return transposed_weights

        shapes = (
            (8,8,24), #only conv layer 1
            (16,1,8), #no conv layer 2
            (4,1,16) #no conv layer 3
            )
        self.conv1.W = convert_weight_shapes(w1, shapes[0])
        self.conv2.W = convert_weight_shapes(w2, shapes[1])
        self.conv3.W = convert_weight_shapes(w3, shapes[2])

    def forward(self, x):
        """

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        #distributed scanning over layers
        self.conv1 = Conv1D(in_channel=24, out_channel=2, kernel_size=2, stride=2)
        self.conv2 = Conv1D(in_channel=2, out_channel=8, kernel_size=2, stride=2)
        self.conv3 = Conv1D(in_channel=8, out_channel=4, kernel_size=2, stride=1)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.flatten = Flatten()
        
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.flatten]

    def __call__(self, x):
        return self.forward(x)


    def init_weights(self, weights):
        # Load the weights for CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        #function to transpose reshape transpose weights into 
        def convert_weight_shapes(w, shape, crop):
            transposed_weights = w.T
            reshaped_weights = transposed_weights.reshape(shape) if not crop else w.T.reshape(shape)[:,:crop,:]
            transposed_weights = reshaped_weights.transpose(0,2,1)
            return transposed_weights

        w1,w2,w3 = weights
        shapes = (
            (2,8,24), #follow same shapes as original matrix but in different order
            (8,4,2), 
            (4,2,8)
            )
        self.conv1.W = convert_weight_shapes(w1[:,:2], shapes[0], crop=2)
        self.conv2.W = convert_weight_shapes(w2[:,:8], shapes[1], crop=2)
        self.conv3.W = convert_weight_shapes(w3[:,:], shapes[2], crop=None)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta