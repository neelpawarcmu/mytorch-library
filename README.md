# Introduction

This repository contains a Torch library built using Python and NumPy as part of Deep Learning at CMU. As part of this course, this involves designing main parts of PyTorch from scratch. The main goal is to understand the complex mathematical computations, modularity, structure and usage of PyTorch.

Further use of PyTorch for solving big data problems by implementing research papers can be found here:
1. Frame Level Classification of Speech (to be linked)
2. [Face detection and verification with Resnet](https://github.com/neelpawarcmu/deep-learning-course-projects/blob/main/Face_detection_and_verification_with_Resnet_50_design.ipynb)
3. [Utterance to Phoneme Mapping](https://github.com/neelpawarcmu/deep-learning-course-projects/blob/main/Utterance_to_Phoneme_Mapping_using_Seq2Seq.ipynb)
4. Attention-based End-to-End Speech-to-Text Deep Neural Network (to be linked)


# Components
MyTorch includes the following components:

* `activation.py`: Contains activation functions (currently Tanh implemented) and logic for their backprop.
* `ctc.py`: implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch.
* `conv.py`: implements Conv1D, Conv2D, Conv2D with dilation and Flatten. (similar to torch.nn.Conv2d and torch.nn.Flatten)
CTC is an algorithm used to train deep neural networks in speech recognition, handwriting recognition and other sequence problems.
CTC is used when we don't know how the input aligns with the output (how the characters in the transcript align to the audio).
* `batchnorm.py`: Performs batch normalization (similar to torch.nn.BatchNorm1d).
* `ctc_loss.py`: Performs forward and backward propagation of CTC loss.
* `gru.py`: Implements the mathematical functions involved in GRU (Gated Recurrent Unit) class of MyTorch. Provides support for mytorch.GRU (similar to torch.nn.GRU).
* `linear.py`: Implements the mathematical functions involved in Linear class of MyTorch. Provides support for mytorch.Linear (similar to torch.nn.Linear)
* `rnn_cell.py`: Implements the mathematical functions involved in RNN class of MyTorch. Provides support for mytorch.RNN (similar to torch.nn.RNN).
* `loss.py`: Implements softmax cross entropy loss.
* `search.py`: Implements greedy and beam search. Greedy search greedily picks the label with maximum probability at each time step to compose the output sequence. Beam search is a more effective decoding technique to obtain a sub-optimal result out of sequential decisions, striking a balance between a greedy search and an exponential exhaustive search by keeping a beam of top-k scored sub-sequences at each time step (BeamWidth). In the context of CTC, we would also consider a blank symbol and repeated characters, and merge the scores for several equivalent sub-sequences.


# Library usage
Examples
* `create_character_predictor.py` - Character predictor using `gru_cell.py`,  `linear.py`.
* `create_cnn.py` - Create simple CNN using `conv.py`, `linear.py`.
* `create_mlp.py` - Create simple MLP using `linear.py`, `batchnorm.py`.
*  `rnn_classifier.py` Phoneme classifier using `rnn_cell.py`, `linear.py`.
*  `mlp_scan.py` - Create simple MLP using `linear.py` to scan image and contrast results and required computation with `create_cnn.py`.
(In addition, all of the above examples use activations and loss functions found in `activations.py` and `loss.py`)


# Feedback
Please feel free to leave feedback via a PR. Thank you!
