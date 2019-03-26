"""
Training of the GNNInference objects
Typical training loop, resulting in saved models
(in inference/pretrained)
Authors: kkorovin@cs.cmu.edu (main), markcheu@andrew.cmu.edu

TODO:
* think how to specify an experiment; 
    currently I think of train.py (with args)
    + a directory of inference experiments
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from experiments.exp_specs import get_dataset_by_name
from inference import get_algorithm
from time import time

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_name', default='debug', type=str,
                        help='name of training set (see experiments/train_specs.py)')
    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to train GNN to perform')
    parser.add_argument('--data_dir', default='./graphical_models/datasets/',
                        type=str, help='directory to load data from')
    parser.add_argument('--model_dir', default='./inference/pretrained',
                        type=str, help='directory to save a trained model')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display training statistics')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_train_args()
    dataset = get_dataset_by_name(args.train_set_name, args.data_dir)

    # GGNN parmeters
    n_nodes = dataset[0].W.shape[0]
    n_hidden_states = 5
    message_dim_P = 5
    hidden_unit_message_dim = 64 
    hidden_unit_readout_dim = 64
    T = 10
    learning_rate = 1e-2
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gnn_constructor = get_algorithm("gnn_inference")
    gnn_inference = gnn_constructor('marginal', n_nodes, n_hidden_states, 
        message_dim_P,hidden_unit_message_dim, hidden_unit_readout_dim, T)
    optimizer = Adam(gnn_inference.model.parameters(), lr=learning_rate)

    criterion = nn.KLDivLoss()
    # criterion = nn.MSELoss()

    for epoch in range(epochs):
        gnn_inference.train(dataset, optimizer, criterion, device)
        t = "_".join(str(time()).split("."))
        gnn_inference.save_model(t)

    # TODO: training loop, assuming the
    # map vs marginals part is handled in forward
