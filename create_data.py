"""

Data creation helpers and action.
Authors: kkorovin@cs.cmu.edu (main)

TODO:
* add random seeds
"""

import argparse
import numpy as np

from graphical_models.data_structs import BinaryMRF
from inference import get_algorithm


def parse_dataset_args():
    parser = argparse.ArgumentParser()
    # TODO: options
    parser.add_argument('--graph_struct', default="star", type=str,
                        help='type of graph structure, such as star or fc')
    parser.add_argument('--size_range', default="1_1", type=str,
                        help='range of sizes, in the form "10_20"')
    parser.add_argument('--num', default=10, type=int,
                        help='number of graphs to generate')
    parser.add_argument('--algo', default='exact', type=str,
                        help='algorithm to use for labeling')
    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to perform')
    parser.add_argument('--save_to_dir', default='./graphical_models/datasets/',
                        type=str, help='directory to save a generated dataset')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display dataset statistics')
    args = parser.parse_args()
    return args


def generate_struct_mask(struct, n_nodes):
    mask = np.ones((n_nodes, n_nodes), dtype=int)
    if struct == "star":
        mask[0, 0] = 0
        mask[1:, 1:] = 0
    elif struc == "fc":
        mask[np.arange(n_nodes), np.arange(n_nodes)] = 0
    else:
        raise NotImplementedError("Other structures not implemented yet.")
    return mask


def construct_binary_mrf(struct, n_nodes):
    """Construct one binary MRF graphical model

    Arguments:
        struct {string} -- structure of the graph
        (on of "chain", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
    Returns:
        np.array -- BinaryMRF object
    """
    W = np.random.normal(0., 1., (n_nodes, n_nodes))
    b = np.random.normal(0., 0.25, n_nodes)
    mask = generate_struct_mask(struct, n_nodes)
    W *= mask
    return BinaryMRF(W, b)


if __name__=="__main__":
    ## parse arguments and dataset name
    args = parse_dataset_args()
    low, high = args.size_range.split("_")
    size_range = np.arange(int(low), int(high)+1)
    ## construct graphical models
    graphs = []
    for _ in range(args.num):
        # sample n_nodes from range
        n_nodes = np.random.choice(size_range)
        graphs.append(construct_binary_mrf(args.graph_struct, n_nodes))

    ## label them using a chosen algorithm
    algo_obj = get_algorithm(args.algo)(args.mode)
    list_of_res = algo_obj.run(graphs)

    ## save to graphical_models/datasets
    # TODO

