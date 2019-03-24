"""
Training of the GNNInference objects
Typical training loop, resulting in saved models
(in inference/pretrained)
Authors: kkorovin@cs.cmu.edu (main)

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

    # TODO: fit a GNN on these graphs
    P = 2 # message dim
    n_hidden_states = 2
    gnn_constructor = get_algorithm("gnn_inference")
    gnn_inference = gnn_constructor('marginal',dataset[0].W.shape[0],n_hidden_states, P)
    optimizer = Adam(gnn_inference.model.parameters(), lr=1e-2)
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_inference.run(dataset,optimizer,criterion,device)

    # TODO: training loop, assuming the
    # map vs marginals part is handled in forward
