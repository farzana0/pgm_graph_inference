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
            annotation=torch.zeros(1,self.n_nodes,self.annotation_dim).float()
            padding = torch.zeros(len(annotation), self.n_nodes, self.state_dim - self.annotation_dim).float()
            init_input = torch.cat((annotation, padding), 2).to(device)

            probs=np.ones(self.n_nodes)/self.n_nodes #TODO, for testing only
            b = torch.from_numpy(graph.b).unsqueeze(0).float().to(device)
            adj = torch.from_numpy(np.concatenate((graph.W, graph.W.T),axis=1)).unsqueeze(0).float().to(device)
            
            output = self.model(init_input, annotation,adj,b)
            loss = criterion(output.squeeze(1), target)


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def run(self, dataset, optimizer, criterion, device):
        #TODO: exact probs need to be in dataset
        # for i, graph,probs in enumerate(dataset,0):
        self.model.train()
        self.model.zero_grad()
        batch_loss=[]
        for i, graph in enumerate(dataset):
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            target =torch.from_numpy(graph.marginal).float().to(device)
            out = self.model(J,b)
            loss = criterion(out, target)
            # batch_loss.append(loss)
            loss.backward()
            optimizer.step()
            # print(loss)
            if((i+1)%100==0):
                print('target: ',target)
                print('output: ',out)
                print('loss', loss)

            #     # ll = torch.stack(batch_loss).sum()
            #     ll_mean = torch.stack(batch_loss).mean()
            #     ll_mean.backward()
            #     optimizer.step()
            #     self.model.zero_grad()
            #     print(ll_mean)
