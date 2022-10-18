import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        """

        self.x = x

        # training: compute mean and use it; compute running mean
        # testing: use running mean 
        # norm = (batch, in), gamma = (1, in) => out = (1, in)

        if not eval:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            # step 1. normalize to unit size and center
            norm = (x - mean) / (np.sqrt(var + self.eps))  
            # step 2. affine transformation
            out = (self.gamma * norm) + self.beta    
        else:
            mean = self.running_mean #mean = (1, in)
            var = self.running_var #var = (1, in)
            norm = (x - mean) / (np.sqrt(var + self.eps)) #norm = (batch, in) #step 1.
            out = (self.gamma * norm) + self.beta #out = (batch, in) #step 2. 
            return out
            

        #store as class variables for backprop
        self.mean = mean 
        self.var = var
        self.norm = norm
        self.out = out

        # Update running batch statistics 
        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        # TODO: improve notations like first term etc. 
        batch_size = delta.shape[0] 
        
        #repetitive term saved for ease
        sqrt_var_eps = np.sqrt(self.var + self.eps)

        #update gradients
        self.dgamma = np.sum(delta * self.norm, axis = 0, keepdims=True)
        self.dbeta = np.sum(delta, axis = 0, keepdims=True)
        
        #calculate dnorm and dvar
        gradNorm = self.gamma * delta
        gradVar = -0.5*(np.sum((gradNorm * (self.x - self.mean) / (sqrt_var_eps**3)), axis = 0))

        #calculate dmu
        first_term_dmu = - np.sum(gradNorm/sqrt_var_eps, axis = 0)
        second_term_dmu = - (2/batch_size)*(gradVar)*(np.sum(self.x-self.mean, axis = 0))
        gradMu = first_term_dmu + second_term_dmu

        #calculate dx = f(dnorm) + g(dvar) + h(dmu)
        first_term_dx = gradNorm / sqrt_var_eps
        second_term_dx = gradVar * (2/batch_size) * (self.x-self.mean)
        third_term_dx = gradMu * (1/batch_size)
        dx = first_term_dx + second_term_dx + third_term_dx

        return dx
