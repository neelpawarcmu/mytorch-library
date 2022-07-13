import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]
        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]

        """

        # -------------------------------------------->

        extSymbols = [self.BLANK] * (2*len(target) + 1)
        skipConnect = [0] * (2*len(target) + 1)

        s, f = 0, 0 # slow pointer for extSymbols and fast pointer for target
        while f < len(target):
                if s % 2 == 1:
                        # fill B, IY, IY, F
                        extSymbols[s] = target[f]
                        # check if skipping connection is okay i.e. consecutive disparate phonemes like B and IY
                        # thus IY will have 1 in skipconnect
                        if f>0 and target[f]!=target[f-1]:
                                skipConnect[s] = 1
                        f+=1
                s+=1
        return extSymbols, skipConnect
        # <---------------------------------------------

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # alpha = (time_stamps * target_length)
        alpha[0, 0] = logits[0, int(extSymbols[0])]
        alpha[0, 1] = logits[0, int(extSymbols[1])]
        for t in range(1,T):
                alpha[t, 0] = alpha[t-1, 0] * logits[t, int(extSymbols[0])]
                for i in range(1,S):
                        alpha[t, i] = alpha[t-1, i-1] + alpha[t-1, i]
                        if skipConnect[i] == 1:
                                alpha[t, i] += alpha[t-1, i-2]
                        alpha[t, i] *= logits[t, int(extSymbols[i])]

        return alpha
        # <---------------------------------------------
    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        beta[T-1, S-1] = 1
        beta[T-1, S-2] = 1
        for t in reversed(range(T-1)):
                beta[t, S-1] = beta[t+1, S-1] * logits[t+1, int(extSymbols[S-1])]
                for i in reversed(range(S-1)):
                        beta[t,i] = beta[t+1, i]*logits[t+1, int(extSymbols[i])] + beta[t+1, i+1]*logits[t+1, int(extSymbols[i+1])]
                        if i<S-2 and skipConnect[i+2] == 1:
                                beta[t, i] += (beta[t+1, i+2] * logits[t+1, extSymbols[i+2]])
        return beta
        # <---------------------------------------------
        

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1) = (T * S))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1) = (T * S))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1) = (T * S))
                posterior probability 

        """
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))

        # -------------------------------------------->
        for t in range(T):
                sumgamma = 0
                for i in range(S):
                        gamma[t, i] = alpha[t, i] * beta[t, i]
                        sumgamma += gamma[t, i]
                for i in range(S):
                        gamma[t, i] /= sumgamma
        return gamma
        # <---------------------------------------------