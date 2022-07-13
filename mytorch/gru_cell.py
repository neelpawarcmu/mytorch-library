import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        #IDL lec15 slide22 + hw3p1 writeup pg 9 

        # # #slides
        # # #block 1
        term1 = np.dot(self.Wrh, h) + self.bir            # (h,h) dot (h,) + (h,) = (h,)
        term2 = np.dot(self.Wrx, x) + self.bhr            # (h,d) dot (d,) + (h,) = (h,)
        self.r = self.r_act(term1 + term2)                # activation(h,) = (h,)

        #block2
        term1 = np.dot(self.Wzh, h) + self.biz            # (h,h) dot (h,) + (h,) = (h,)
        term2 = np.dot(self.Wzx, x) + self.bhz            # (h,d) dot (d,) + (h,) = (h,)
        self.z = self.r_act(term1 + term2)                # activation(h,) = (h,)

        #block3
        term1 = np.dot(self.Wnx, x) + self.bin            # (h,h) dot (h,) + (h,) = (h,)
        term2 = self.r * (np.dot(self.Wnh, h) + self.bhn) # (h,) * (h,h) dot (h,) + (h,) = (h,)
        # save inner state to compute derivative in backprop easily
        self.n_state = np.dot(self.Wnh, h) + self.bhn                       
        self.n = self.h_act(term1 + term2)                # activation(h,) = (h,)

        #block4 
        term1 = (1 - self.z) * self.n                     # (h,) * (h,) = (h,)
        term2 = self.z * h                                # (h,) * (h,) = (h,)
        self.h_t = term1 + term2                          # (h,) + (h,) = (h,)  


        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert self.h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return self.h_t


    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim) #### this is basically dh_t, derivative of an h_t from forward
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly


        input_dim, = self.x.shape
        hidden_dim, = self.hidden.shape

        #input = 5, hidden = 2
        # derivatives are row vectors and actuals are column vectors. 
        # to begin with, delta shape is good to go as given by them
        # some changes to pdf for gru but the small subparts only, overall it is the same
        # calculate derivatives as given in pdf and transpose in the end to get shape of der = shape of actual
        # all things should be vectors like (1,5)
        # we get around 26 equations for fwd
        # Follow the derivatives from the saved values in these equations
        #delta is same as dh_t

        for elem in ('x','hidden','n','z','r','Wnx','bin','Wnh','bhn'):
            elemname = elem
            elem = eval('self.'+elem)
            print(f'{elemname} shape: {elem.shape}', end="\t")
        
        self.x = self.x.reshape(1,-1)
        self.hidden = self.hidden.reshape(1,-1)
        self.r = self.r.reshape(1,-1)
        self.z = self.z.reshape(1,-1)
        self.n = self.n.reshape(1,-1)

        # create 5 derivatives here itself for ease of troubleshooting
        dx = np.zeros_like(self.x) 
        dh = np.zeros_like(self.hidden)
        dn = np.zeros_like(self.n) 
        dz = np.zeros_like(self.z)
        dr = np.zeros_like(self.z)

        #block4          
        dz += delta * (-self.n + self.hidden)      # (1,h) * (1,h) = (1,h)
        dn += delta * (1 - self.z)                 # (1,h) * (1,h) = (1,h)
        dh += delta * self.z                       # (1,h) * (1,h) = (1,h)

        #block3
        grad_activ_n   = dn * (1-self.n**2)         # (1,h)
        r_grad_activ_n = grad_activ_n * self.r      # (1,h)

        self.dWnx += np.dot(grad_activ_n.T, self.x) # (h,1) dot (1,d) = (h,d)
        dx        += np.dot(grad_activ_n, self.Wnx) # (1,h) dot (h,d) = (1,d)
        self.dbin += np.sum(grad_activ_n, axis=0)   # (1,h)
        dr        += grad_activ_n * self.n_state.T  # (1,h)


        self.dWnh += np.dot(r_grad_activ_n.T, self.hidden) # (h,1) dot (1,h) = (h,d)
        dh        += np.dot(r_grad_activ_n, self.Wnh)      # (h,1) dot (1,h) = (h,d)
        self.dbhn += np.sum(r_grad_activ_n, axis=0)        # (1,h)

        #block2
        grad_activ_z = dz * self.z * (1-self.z)             # (1,h) * (1,h) * (1,h) = (1,h)
        
        dx        += np.dot(grad_activ_z, self.Wzx)         # (1,h) dot (h,d) = (1,d)
        self.dWzx += np.dot(grad_activ_z.T, self.x)         # (h,1) dot (1,d) = (h,d)
        self.dWzh += np.dot(grad_activ_z.T, self.hidden)    # (h,1) dot (1,d) = (h,d)
        dh        += np.dot(grad_activ_z, self.Wzh)         # (1,h) dot (h,d) = (1,d)
        self.dbiz += np.sum(grad_activ_z, axis=0)           # (1,h)
        self.dbhz += np.sum(grad_activ_z, axis=0)           # (1,h)


        #block1
        grad_activ_r = dr * self.r * (1-self.r)          # (1,h) * (1,h) * (1,h) = (1,h)
        dx        += np.dot(grad_activ_r, self.Wrx)      # (1,h) dot (h,d) = (1,d)
        self.dWrx += np.dot(grad_activ_r.T, self.x)      # (h,1) dot (1,d) = (h,d)
        self.dWrh += np.dot(grad_activ_r.T, self.hidden) # (h,1) dot (1,h) = (h,h)
        dh        += np.dot(grad_activ_r, self.Wrh)      # (1,h) dot (h,d) = (1,d)
        self.dbir += np.sum(grad_activ_r, axis=0)        # (1,h)
        self.dbhr += np.sum(grad_activ_r, axis=0)        # (1,h)

        return dx, dh
        