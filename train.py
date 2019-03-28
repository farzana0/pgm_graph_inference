"""
Training of the GNNInference objects
Typical training loop, resulting in saved models
(in inference/pretrained)
Authors: kkorovin@cs.cmu.edu, markcheu@andrew.cmu.edu

# TODO: use validation set for tuning

"""

import os
import argparse
from time import time
import torch
import torch.nn as nn
from torch.optim import Adam

from experiments.exp_helpers import get_dataset_by_name
from inference import get_algorithm
from constants import *


def parse_train_args():
    parser = argparse.ArgumentParser()

    # critical arguments, change them
    parser.add_argument('--train_set_name', type=str,
                        help='name of training set (see experiments/exp_helpers.py)')
    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to train GNN to perform')

    # non-critical arguments, fine with default
    # model_name can be used for different hyperparameters later
    parser.add_argument('--model_name', default='default',
                        type=str, help='model name, defaults to the train_set_name')
    parser.add_argument('--data_dir', default='./graphical_models/datasets/train',
                        type=str, help='directory to load training data from')
    parser.add_argument('--model_dir', default='./inference/pretrained',
                        type=str, help='directory to save a trained model')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display training statistics')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_train_args()
    print("Training a model `{}` on training dataset `{}`".format(args.model_name,
                                                                  args.train_set_name))

    dataset = get_dataset_by_name(args.train_set_name, args.data_dir)[:50]

    # GGNN parmeters
    n_hidden_states = 5
    message_dim_P = 5
    hidden_unit_message_dim = 64 
    hidden_unit_readout_dim = 64
    T = 10
    learning_rate = 1e-2

    # number of epochs
    epochs = 1

    gnn_constructor = get_algorithm("gnn_inference")
    gnn_inference = gnn_constructor('marginal', n_hidden_states, 
                                    message_dim_P,hidden_unit_message_dim,
                                    hidden_unit_readout_dim, T)
    optimizer = Adam(gnn_inference.model.parameters(), lr=learning_rate)

    criterion = nn.KLDivLoss()
    # criterion = nn.MSELoss()

    for epoch in range(epochs):
        gnn_inference.train(dataset, optimizer, criterion, DEVICE)

        os.makedirs(args.model_dir, exist_ok=True)
        if args.model_name == "default":
            model_path = os.path.join(args.model_dir, args.train_set_name)
        else:
            model_path = os.path.join(args.model_dir, args.model_name)
        gnn_inference.save_model(model_path)

    print("Model saved in {}".format(model_path))

