import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_prime: (batch_size, hidden_size)
            hidden state at the current time step and current layer

        """
        # W_ih shape => hidden_size * input_size, x => batch_size * input_size
        # b_ih shape => hidden_size
        # W_hh shape => hidden_size * hidden_size, h => batch_size * hidden_size
        # b_hh shape => hidden_size
        # h_prime: (batch_size, hidden_size)

        affine_term_1 = np.dot(x, self.W_ih.T) + self.b_ih
        affine_term_2 = np.dot(h, self.W_hh.T) + self.b_hh
        h_prime = self.activation(affine_term_1 + affine_term_2)
        return h_prime


    def backward(self, delta, h, h_prev_l, h_prev_t):
        """RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer
        (basically analogous to x from forward) 

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer
        (basically analogous to h from forward) 

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]

        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optional input.
        dz = self.activation.derivative(state=h) * delta # => 

        # backprop formulae
        # h_prime = activation(W_ih * x + W_hh * h)
        # dh_prime / dactivation = dz
        # dh_prime / dW_ih = (dh_prime / dactivation) * (dactivation / dW_ih) = dz * x , where x is h_prev_l
        # dh_prime / dW_hh = (dh_prime / dactivation) * (dactivation / dW_hh) = dz * h , where h is h_prev_t
        # dh_prime / db_ih = (dh_prime / dactivation) * 1                     = dz
        # dh_prime / db_hh = (dh_prime / dactivation) * 1                     = dz
        # dh_prime / dx    = (dh_prime / dactivation) * (dactivation / dx)    = dz * W_ih
        # dh_prime / dh    = (dh_prime / dactivation) * (dactivation / dh)    = dz * W_hh

        '''
        backprop shapes based on formulae
        #Note1: We take averages of actual derivatives shown below, as we are accumulating gradients over entire time vector
        dz : (batch_size * hidden_size)
        delta : (batch_size, hidden_size)
        dW_ih = dz * h_prev_l => (hidden_size * input_size) = (batch_size * hidden_size).T * (batch_size, input_size)
        dW_hh = dz * h_prev_t => (hidden_size * hidden_size) = (batch_size * hidden_size).T * (batch_size, hidden_size)
        db_ih = dz            => (hidden_size) = (batch_size * hidden_size) , sum over batch_size axis
        
        dx    = dz * W_ih    => (batch_size, input_size) = (batch_size * hidden_size) * (hidden_size * input_size)
        dh    = dz * W_hh    => (batch_size, hidden_size) = (batch_size * hidden_size) * (hidden_size * hidden_size)
        '''

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size # division by batch_size: Note1
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size 
        self.db_ih += np.sum(dz, axis=0) / batch_size 
        self.db_hh += np.sum(dz, axis=0) / batch_size 

        '''
        #Note2: No storing or averaging done for dx, dh because @@@@@?? 
        '''
        # 2) Compute dx, dh
        dx = np.dot(dz, self.W_ih) #no storing done here
        dh = np.dot(dz, self.W_hh)

        # 3) Return dx, dh
        return dx, dh