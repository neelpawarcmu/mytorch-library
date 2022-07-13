import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []

        self.ctc = CTC()

    def __call__(self, logits, target, input_lengths, target_lengths):
        return self.forward(logits, target, input_lengths, target_lengths)


    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """

        # -------------------------------------------->
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        # B = batch_size
        B, _ = target.shape 
        totalLoss = np.zeros(B)


        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            

            # Truncate the target to target length and logits to input length
            target_truncated_slice = target[b, :target_lengths[b]]
            logits_truncated_slice = logits[:input_lengths[b], b, :]

            # Extend target sequence with blank
            extSymbols, skipConnect = self.ctc.targetWithBlank(target_truncated_slice)

            # Compute forward probabilities and backward probabilities
            alpha = self.ctc.forwardProb(logits_truncated_slice, extSymbols, skipConnect)
            beta = self.ctc.backwardProb(logits_truncated_slice, extSymbols, skipConnect)
            
            # Compute posteriors using total probability function
            gamma = self.ctc.postProb(alpha, beta)
            
            batch_loss = 0
            T, S = gamma.shape
            for t in range(T):
                for s in range(S):
                    loss_at_single_input = - gamma[t,s] * np.log(logits_truncated_slice[t, extSymbols[s]])
                    batch_loss += loss_at_single_input

            totalLoss[b] = batch_loss

        return np.mean(totalLoss)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        print('dY.shape', dY.shape)

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------
            
            
            # -------------------------------------------->
            # Truncate the target to target length and logits to input length
            target_truncated_slice = self.target[b, :self.target_lengths[b]]
            logits_truncated_slice = self.logits[:self.input_lengths[b], b, :]

            # Extend target sequence with blank
            extSymbols, skipConnect = self.ctc.targetWithBlank(target_truncated_slice)

            alpha = self.ctc.forwardProb(logits_truncated_slice, extSymbols, skipConnect)
            beta = self.ctc.backwardProb(logits_truncated_slice, extSymbols, skipConnect)
            gamma = self.ctc.postProb(alpha, beta)

            T, S = gamma.shape
            for t in range(T):
                for s in range(S):
                    dY[t, b, extSymbols[s]] -= gamma[t, s] / logits_truncated_slice[t, extSymbols[s]]
            # <---------------------------------------------
            
        return dY
