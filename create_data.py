"""

Data creation helpers and action.
Authors: kkorovin@cs.cmu.edu

If variable size range is supplied, each
generated graph has randomly chosen size in range.

TODO:
* add random seeds
"""

import os
import argparse
import numpy as np
from pprint import pprint
from time import time

from graphical_models import construct_binary_mrf
from inference import get_algorithm


def parse_dataset_args():
    parser = argparse.ArgumentParser()

    # crucial arguments
    parser.add_argument('--graph_struct', default="star", type=str,
                        help='type of graph structure, such as star or fc')
    parser.add_argument('--size_range', default="5_5", type=str,
                        help='range of sizes, in the form "10_20"')
    parser.add_argument('--num', default=1, type=int,
                        help='number of graphs to generate')
    # should be used for train-test split
    parser.add_argument('--data_mode', default='train',
                        type=str, help='use train/val/test subdirectory of base_data_dir')

    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to perform')
    parser.add_argument('--algo', default='exact', type=str,
                        help='algorithm to use for labeling')

    # no need to change the following arguments
    parser.add_argument('--base_data_dir', default='./graphical_models/datasets/',
                        type=str, help='directory to save a generated dataset')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display dataset statistics')
    args = parser.parse_args()
    return args


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
        if args.verbose: print(graphs[-1].W)

    ## label them using a chosen algorithm
    algo_obj = get_algorithm(args.algo)(args.mode)
    list_of_res = algo_obj.run(graphs)

    ## save to graphical_models/datasets
    for graph, res in zip(graphs, list_of_res):
        if args.mode == "marginal":
            res_marginal, res_map = res, None
        else:
            res_marginal, res_map = None, res

        directory = os.path.join(args.base_data_dir, args.data_mode,
                                 graph.struct, str(graph.n_nodes))
        os.makedirs(directory, exist_ok=True)
        data = {"W": graph.W, "b": graph.b,
                "marginal": res_marginal, "map": res_map}
        #pprint(data)

        t = "_".join(str(time()).split("."))
        path_to_graph = os.path.join(directory, t)
        np.save(path_to_graph, data)

