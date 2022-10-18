import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        #
        batch_size, in_channel, input_size = x.shape
        output_size = ((input_size - self.kernel_size) // self.stride) + 1

        #store x to use later in backprop calculations
        self.x = x

        Z = np.zeros((batch_size, self.out_channel, output_size))


        #iterate over batches
        for i in range(batch_size):
            batch =  x[i,:,:] # shape => (in_channel, input_size)
            
            #iterate over channels
            for j in range(self.out_channel):
                W = self.W[j,:,:] # shape => (in_channel, input_size)
                b = self.b[j] # shape => (1) ie. scalar

                #iterate over image width 
                for k in range(output_size): # (iterating over output size ensures we dont run out of image for edge cases where filter is longer)
                    start, end = k * self.stride, k * self.stride + self.kernel_size  # indexing for kernel placement over image
                    segment = batch[:, start:end] # shape => (in_channel, kernel_size)
                    affineCombination = np.sum(segment * W) + b # shape => (1) ie. scalar
                    Z[i,j,k] = affineCombination
        return Z

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, out_channel, output_size = delta.shape

        dx = np.zeros_like(self.x)
        for i in range(batch_size):
            batch =  self.x[i,:,:] # shape => (in_channel, input_size)
            for j in range(self.out_channel):
                W = self.W[j,:,:] # shape => (in_channel, kernel_size)
                b = self.b[j] # shape => (), ie. scalar
                for k in range(output_size):
                    start, end = k * self.stride, k * self.stride + self.kernel_size
                    segment = batch[:, start:end] # shape => (in_channel, kernel_size)
                    delta_local = delta[i,j,k] # shape => (1) ie. scalar

                    # (make sure gradient dimensions match) -> expected dims:
                    # dx[i,:,start:end]: (in_channel, kernel_size) 
                    # dW[j,:,:]: (in_channel, kernel_size)
                    # db[j]: (1) ie. scalar

                    dx[i,:,start:end] += W * delta_local
                    self.dW[j,:,:] += segment * delta_local
                    self.db[j] += delta_local
        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        
        batch_size, in_channel, input_width, input_height = x.shape
        output_width = ((input_width - self.kernel_size) // self.stride) + 1
        output_height = ((input_height - self.kernel_size) // self.stride) + 1

        #store x to use later in backprop
        self.x = x

        Z = np.zeros((batch_size, self.out_channel, output_width, output_height))

        for i in range(batch_size):
            batch =  x[i,:,:,:] # shape => (in_channel, input_width, input_height)
            for j in range(self.out_channel):
                W = self.W[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                b = self.b[j] # shape => (1), ie. scalar
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_size
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_size
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_size, kernel_size)
                        affineCombination = np.sum(segment * W) + b # shape => (1), ie. scalar
                        Z[i,j,k,l] = affineCombination
        return Z

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, out_channel, output_width, output_height = delta.shape

        dx = np.zeros_like(self.x)

        for i in range(batch_size):
            batch =  self.x[i,:,:,:] # shape => (in_channel, input_width, input_height)
            for j in range(self.out_channel):
                W = self.W[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                b = self.b[j] # shape => (), ie. scalar
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_size
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_size
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_size, kernel_size)
                        delta_local = delta[i,j,k,l] # shape => (1) ie. scalar

                        # (make sure gradient dimensions match) -> expected size:
                        # dx[i,:,start:end]: (in_channel, kernel_size) 
                        # dW[j,:,:]: (in_channel, kernel_size)
                        # db[j]: (1) ie. scalar

                        dx[i,:,startWidth:endWidth,startHeight:endHeight] += W * delta_local
                        self.dW[j,:,:,:] += segment * delta_local
                        self.db[j] += delta_local
        return dx
        

# dilation => 
class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Takes two attributes into consideration: padding and dilation.
        the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (kernel_size-1) * (dilation-1) + kernel_size
        self.kernel_dilated = (kernel_size-1) * (dilation-1) + kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """

        batch_size, in_channel, input_width, input_height = x.shape

        input_width_padded = input_width + 2 * self.padding
        input_height_padded = input_height + 2 * self.padding

        x_padded = np.zeros((batch_size, in_channel, input_width_padded, input_height_padded))

        # padding x with self.padding parameter
        for b in range(batch_size):
            batch = x[b,:,:,:] 
            for c in range(in_channel): # shape => (in_channel, input_width, input_height)
                x_padded[b,c,:,:] = np.pad(batch[c,:,:], ((self.padding,self.padding),(self.padding,self.padding)), mode='constant', constant_values=0)
                
        # do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        # loop over channels to get self.W_dilated

        self.W_dilated = np.zeros((self.out_channel, in_channel, self.kernel_dilated, self.kernel_dilated))
        for ch_o in range(self.out_channel):
            for ch_i in range(self.in_channel):
                self.W_dilated[ch_o, ch_i, :, :][::self.dilation,::self.dilation] = self.W[ch_o, ch_i, :, :]


        # regular forward, just like Conv2d().forward()
        # store x to use later in backprop gradient calculations
        self.x = x
        self.x_padded = x_padded

        # output_size calculations
        output_width = (input_width_padded - self.kernel_dilated) // self.stride + 1
        output_height = (input_height_padded - self.kernel_dilated) // self.stride + 1

        Z = np.zeros((batch_size, self.out_channel, output_width, output_height))
        
        '''
        x_padded = (batch_size, in_channel, input_width_padded, input_height_padded)
        W_dilated = (out_channel, in_channel, kernel_dilated, kernel_dilated)
        b = (out_channel)
        Z = (batch_size, out_channel, output_width, output_height)
        '''

        for i in range(batch_size):
            batch =  x_padded[i,:,:,:] # shape => (in_channel, input_width_padded, input_height_padded)
            for j in range(self.out_channel):
                W = self.W_dilated[j,:,:,:] # shape => (in_channel, kernel_dilated, kernel_dilated)
                b = self.b[j] # shape => (1), ie. scalar
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_dilated
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_dilated
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_dilated, kernel_dilated)
                        affineCombination = np.sum(segment * W) + b # shape => (1), ie. scalar
                        Z[i,j,k,l] = affineCombination
        return Z


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # main part is like Conv2d().backward(). The only difference is: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        
        #we dilate delta, convolve using x_padded and dW_dilated and then downsample to preserve shapes changed by dilation
        batch_size, out_channel, output_width, output_height = delta.shape

        dilated_width = (output_width - 1)*(self.stride - 1) + output_width
        dilated_height = (output_height- 1)*(self.stride - 1) + output_height

        delta_dilated = np.zeros((batch_size, out_channel, dilated_width, dilated_height))
        for b in range(batch_size):
            for ch_o in range(self.out_channel):
                delta_dilated[b, ch_o, :, :][::self.stride,::self.stride] = delta[b, ch_o, :, :]

        dx = np.zeros_like(self.x_padded) # shape => (batch_size, in_channel, input_width, input_height)
        dW_dilated = np.zeros_like(self.W_dilated)

        for i in range(batch_size):
            batch =  self.x_padded[i,:,:,:] # shape => (in_channel, input_width, input_height)
            for j in range(self.out_channel):
                W = self.W_dilated[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                for k in range(dilated_width):
                    startWidth, endWidth = k, k + self.kernel_dilated
                    for l in range(dilated_height):
                        startHeight, endHeight = l, l + self.kernel_dilated
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_size, kernel_size)
                        delta_local = delta_dilated[i,j,k,l] # shape => (1) ie. scalar

                        # (make sure gradient dimensions match) -> 
                        # dx[i,:,start:end]:(in_channel, kernel_size) 
                        # dW[j,:,:]:(in_channel, kernel_size)
                        # db[j]: (1) ie. scalar

                        dx[i,:,startWidth:endWidth,startHeight:endHeight] += W * delta_local
                        dW_dilated[j,:,:,:] += segment * delta_local
                        self.db[j] += delta_local

        #reshape to orginal shape
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dW[:,:,i,j] = dW_dilated[:,:,i*self.dilation,j*self.dilation]

        return dx[:,:,self.padding:-self.padding, self.padding:-self.padding]



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        # flatten outputs, save original shape
        self.b, self.c, self.w = x.shape
        dx = np.reshape(x, (self.b, self.c * self.w))
        return dx

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        # unflatten using saved shape
        dx = np.reshape(delta, (self.b, self.c, self.w))
        return dx