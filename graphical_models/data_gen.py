"""

Graphical model generators
Authors: kkorovin@cs.cmu.edu

"""

import numpy as np
from graphical_models.data_structs import BinaryMRF


def generate_struct_mask(struct, n_nodes):
    mask = np.ones((n_nodes, n_nodes), dtype=int)
    if struct == "star":
        mask[0, 0] = 0
        mask[1:, 1:] = 0
    elif struct == "fc":
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
    W = (W + W.T) / 2
    b = np.random.normal(0., 0.25, n_nodes)
    mask = generate_struct_mask(struct, n_nodes)
    W *= mask
    return BinaryMRF(W, b, struct=struct)