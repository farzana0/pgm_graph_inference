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
from inference.core import Inference


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


class GatedGNNInference(GatedGNN, Inference):
    def __init__(self, mode):
        Inference.__init__(self, mode)  # self.mode set here
        GatedGNN.__init__(self)

    def forward(self):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode
        raise NotImplementedError
