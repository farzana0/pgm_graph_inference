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
import numpy as np
import torch.optim as optim

# local
from inference.core import Inference
from inference.ggnn_model import GGNN


class GatedGNNInference(Inference):
    def __init__(self, mode, n_nodes, state_dim=2, annotation_dim=1,
            n_edge_types=1, n_steps=10):
        Inference.__init__(self, mode)  # self.mode set here        
        self.model=GGNN(state_dim, annotation_dim, n_edge_types, n_nodes, n_steps)
        self.n_nodes=n_nodes
        self.state_dim=state_dim
        self.annotation_dim=annotation_dim


    def forward(self, graph, criterion, device):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode
        self.model.eval()
        with torch.no_grad():
            graph.W=(graph.W+graph.W.T)/2 #TODO, for testing only


    def run(self, dataset, optimizer, criterion, device):
        #TODO: exact probs need to be in dataset
        # for i, graph,probs in enumerate(dataset,0):
        self.model.train()
        annotation=torch.zeros(1,self.n_nodes,self.annotation_dim).float()
        padding = torch.zeros(len(annotation), self.n_nodes, self.state_dim - self.annotation_dim).float()
        init_input = torch.cat((annotation, padding), 2).to(device)
        for i, graph in enumerate(dataset):
            self.model.zero_grad()
            probs=np.ones(self.n_nodes)/self.n_nodes #TODO, for testing only

            b = torch.from_numpy(graph.b).float().to(device) #Todo, unused
            target =torch.from_numpy(probs).float().to(device)

            adj = torch.from_numpy(self.create_adj(graph.W)).to(device)
            adj=adj.unsqueeze(0).float()
            output = self.model(init_input, annotation,adj)
            # print(output)
            loss = criterion(output.squeeze(1), target)
            loss.backward()
            optimizer.step()

    def create_adj(self, W):
        n_nodes=W.shape[0]
        A=np.zeros((n_nodes,n_nodes*2))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if(W[i][j]!=0):
                    A[i,j]=W[i][j]
                    A[j,i+n_nodes]=W[i][j]
        return A
