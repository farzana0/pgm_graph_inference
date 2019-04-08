"""

Graphical model generators
Authors: kkorovin@cs.cmu.edu

"""

import numpy as np
import networkx as nx

from graphical_models.data_structs import BinaryMRF

struct_names = ["star", "path", "cycle", "ladder", "grid",
               "circ_ladder", "barbell", "loll", "wheel",
               "bipart", "tripart", "fc"]

def generate_struct_mask(struct, n_nodes, shuffle_nodes):
    # a horrible collection of ifs due to args in nx constructors
    if struct == "star":
        g = nx.star_graph(n_nodes)
    elif struct == "binary_tree":
        raise NotImplementedError("Implement a binary tree.")
    elif struct == "path":
        g = nx.path_graph(n_nodes)
    elif struct == "cycle":
        g = nx.cycle_graph(n_nodes)
    elif struct == "ladder":
        g = nx.ladder_graph(n_nodes)
    elif struct == "grid":
        g = nx.grid_graph(n_nodes)
    elif struct == "circ_ladder":
        g = nx.circular_ladder_graph(n_nodes)
    elif struct == "barbell":
        g = nx.barbell_graph(n_nodes)
    elif struct == "loll":
        m = np.random.choice(range(n_nodes))
        g = nx.lollipop_graph(m, n_nodes-m)
    elif struct == "wheel":
        g = nx.wheel_graph(n_nodes)
    elif struct == "bipart":
        m = np.random.choice(range(n_nodes))
        blocks = (m, n_nodes-m)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "tripart":
        # allowed to be zero
        m, M = np.random.choice(range(n_nodes), size=2)
        if m > M:
            m, M = M, m
        blocks = (m, M-m, n_nodes-M)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "fc":
        g = nx.complete_graph(n_nodes)
    else:
        raise NotImplementedError("Structure {} not implemented yet.".format(struct))

    node_order = list(range(n_nodes))
    if shuffle_nodes:
        np.random.shuffle(node_order)

    # a weird subclass by default; raises a deprecation warning
    # with a new update of networkx, this should be updated to
    # nx.convert_matrix.to_numpy_array
    np_arr_g = nx.to_numpy_matrix(g, nodelist=node_order)
    return np_arr_g.astype(int)


def construct_binary_mrf(struct, n_nodes, shuffle_nodes=True):
    """Construct one binary MRF graphical model

    Arguments:
        struct {string} -- structure of the graph
        (on of "path", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
        shuffle_nodes {bool} -- whether to permute node labelings
                                uniformly at random
    Returns:
        BinaryMRF object
    """
    W = np.random.normal(0., 1., (n_nodes, n_nodes))
    W = (W + W.T) / 2
    b = np.random.normal(0., 0.25, n_nodes)
    mask = generate_struct_mask(struct, n_nodes, shuffle_nodes)
    W *= mask
    return BinaryMRF(W, b, struct=struct)


if __name__ == "__main__":
    graph = construct_binary_mrf("star", 3)
    print(graph.W, graph.b)

    print("Nodes not shuffled:")
    graph = construct_binary_mrf("wheel", 5, False)
    print(graph.W, graph.b)

    print("Nodes shuffled:")
    graph = construct_binary_mrf("wheel", 5)
    print(graph.W, graph.b)

    try:
        graph = construct_binary_mrf("fully_conn", 3)
    except NotImplementedError:
        pass

