"""

Defines GNNInference objects: models that perform
inference, given a graphical model.
Authors: kkorovin@cs.cmu.edu

Options:
- Gated Graph neural network
- TBA

"""

import torch
import torch.nn as nn
from core import Inference
from model_pytorch import GGNN

# class GatedGNN(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self):
#         pass

#     def infer(self, graphs):
        
#         This method should raise an error if self is
#         not initialized to trained model.

#         1) Returns inference results,
#         2) and also sets estimated marginals/MAPs in graphs.
#         pass


class GatedGNNInference(GGNN, Inference):
    def __init__(self, mode):
        Inference.__init__(self, mode)  # self.mode set here
        GGNN.__init__(self)

    def forward(self):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode
        raise NotImplementedError
