"""
# Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        # Initialize and add all your linear layers into the list 'self.linear_layers'

        #create single array of all node counts for linear layers (linear layers => i/p, hidden, o/p)
        layerWise_nodes = []
        layerWise_nodes.append(input_size)
        layerWise_nodes.extend(hiddens)
        layerWise_nodes.append(output_size)

        #create array of layer objects and use that throughout 
        self.linear_layers = [Linear(layerWise_nodes[idx], layerWise_nodes[idx+1], weight_init_fn, bias_init_fn) 
                                for idx in range(len(layerWise_nodes)-1)] #-1:account for idx+1, #didn't use zip but can use it here

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn: 
            self.bn_layers = [BatchNorm(hiddens[idx]) 
                                for idx in range(self.num_bn_layers)]

                                
    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP. 
        # loop over layers till all linear layers are done
        for i in range(self.nlayers):
            #affine combination, linear layer
            z = self.linear_layers[i].forward(x) #zlin

            #batchnorm, bn layer
            if self.bn and i < self.num_bn_layers:
                eval = not self.train_mode
                z = self.bn_layers[i].forward(z, eval) #znorm
            
            #activations, activation layer
            if self.activations:
                y = self.activations[i].forward(z)
            
            x = y #prepare input for next stage (next loop of lin-bn-activ)

        return y

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(self.nlayers):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
        
        for i in range(self.num_bn_layers):
            self.bn_layers[i].dgamma.fill(0.0)
            self.bn_layers[i].dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)
        # Update weights and biases here

        # step size is either the negative of gradient or the positive momentum
        for i in range(len(self.linear_layers)):
            #default steps used for grad descent rule
            step_W = -self.lr * self.linear_layers[i].dW
            step_b = -self.lr * self.linear_layers[i].db
            if self.momentum !=0:
                # update momentum and make steps = momentum update rule
                step_W = self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W + step_W
                step_b = self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b + step_b
            
            #step forward with whatever step was generated
            self.linear_layers[i].W += step_W
            self.linear_layers[i].b += step_b

        # Do gradient descent update for batchnorm layers 
        for i in range(self.num_bn_layers):
            self.bn_layers[i].gamma -= self.lr * self.bn_layers[i].dgamma
            self.bn_layers[i].beta -= self.lr * self.bn_layers[i].dbeta
        


    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        # input_linear -> (bn/no bn) -> activation -> hidden_linear -> (bn/ no bn) -> activation -> output_linear -> loss_fn

        # final layer: one forward pass to get activation state
        outputLayer = self.activations[-1]
        _ = self.criterion.forward(outputLayer.state, labels)

        # derivative at output layer
        delta = self.criterion.derivative()
        for i in reversed(range(self.nlayers)): # nlayers is for all layers upto output layer
            # derivative at activation
            der_activation = self.activations[i].derivative()
            delta = delta * der_activation

            # derivative at batchnorm
            # no derivative if bn layer absent or if testing is on
            if i < self.num_bn_layers and self.train_mode:
                delta = self.bn_layers[i].backward(delta)
            
            # derivative at linear
            delta = self.linear_layers[i].backward(delta)
        

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

#You can complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup 

    for e in range(nepochs):

        # Per epoch setup 

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train 

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val 

        # Accumulate data

    # Cleanup 

    # Return results 

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented