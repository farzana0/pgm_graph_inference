"""

Defines GNNInference objects: models that perform
inference, given a graphical model.
Authors: kkorovin@cs.cmu.edu, markcheu@andrew.cmu.edu

Options:
- Gated Graph neural network:
https://github.com/thaonguyen19/gated-graph-neural-network-pytorch/blob/master/model_pytorch.py
- TBA

"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

# local
from inference.core import Inference
from inference.ggnn_model import GGNN


class GatedGNNInference(Inference):
    def __init__(self, mode, n_nodes, state_dim=2, message_dim=2, n_steps=10, load_path=None):
        Inference.__init__(self, mode)  # self.mode set here        
        self.model = GGNN(n_nodes, state_dim, message_dim, n_steps)
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.message_dim = message_dim

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage,
                    loc: storage))
            self.model.eval()

    def forward(self, graph, criterion, device):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode
        self.model.eval()
        with torch.no_grad():
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            out = self.model(J,b)


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def run(self, dataset, optimizer, criterion, device):
        #TODO: exact probs need to be in dataset
        # for i, graph,probs in enumerate(dataset,0):
        self.model.train()
        self.model.zero_grad()
        batch_loss=[]
        epochs=10
        for epoch in range(epochs):
            for i, graph in enumerate(dataset):
                b = torch.from_numpy(graph.b).float().to(device)
                J = torch.from_numpy(graph.W).float().to(device)
                target =torch.from_numpy(graph.marginal).float().to(device)
                out = self.model(J,b)
                loss = criterion(out, target)
                batch_loss.append(loss)

                if((i+1)%50==0):
                    ll_mean = torch.stack(batch_loss).mean()
                    ll_mean.backward()
                    optimizer.step()
                    self.model.zero_grad()
                    batch_loss=[]
                    print('loss', ll_mean)