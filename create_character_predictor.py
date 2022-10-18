import numpy as np
import sys

sys.path.append("mytorch")
from gru_cell import *
from linear import *


class CharacterPredictor(object):
    """CharacterPredictor class.

    This is the neural net that will run one timestep of the input
    This is to test that our GRU Cell implementation is correct when used as a GRU.

    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        """The network consists of a GRU Cell and a linear layer."""
        self.rnn = GRUCell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)
        self.input_dim, self.hidden_dim, self.num_classes = input_dim, hidden_dim, num_classes

    def init_rnn_weights(
        self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn
    ):
        self.rnn.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn
        )

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        hnext = self.rnn.forward(x, h)
        logits = self.projection(hnext)
        return logits, hnext


def inference(net, inputs):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """
    seq_len, feature_dim = inputs.shape
    hnext = np.zeros(net.hidden_dim)
    logits = np.zeros((seq_len, net.num_classes))

    for i, input in enumerate(inputs):
        log_probs, hnext = net.forward(input, hnext)
        logits[i, :] = log_probs.copy()

    return logits
