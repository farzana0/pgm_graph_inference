"""

Defines GNNInference objects: models that perform
inference, given a graphical model.
Authors: kkorovin@cs.cmu.edu, mark.cheung@cmu.edu

Options:
- Gated Graph neural network:
https://github.com/thaonguyen19/gated-graph-neural-network-pytorch/blob/master/model_pytorch.py
- TBA

"""

import torch
import torch.nn as nn

# local
from core import Inference
from ggnn_model import GGNN


class GatedGNNInference(GGNN, Inference):
    def __init__(self, mode):
        Inference.__init__(self, mode)  # self.mode set here
        GGNN.__init__(self)

    def forward(self):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode
        raise NotImplementedError

    def run(self, graphs):
        raise NotImplementedError
