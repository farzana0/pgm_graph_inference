"""

Data creation helpers and action.
Authors: kkorovin@cs.cmu.edu

"""

from data_structs import BinaryMRF


def construct_binary_mrf(struct, n_nodes):
    """Construct one binary MRF graphical model

    Arguments:
        struct {string} -- structure of the graph
        (on of "chain", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
    Returns:
        np.array -- BinaryMRF object
    """
    raise NotImplementedError
    # something like this:
    adj_mat = np.zeros(n_nodes, n_nodes)
    graph = BinaryMRF(adj_mat, b)
    return graph


def get_label(graph, algo="exact"):
    """
    Use specified algo to extract labels
    for a given graphical model,
    to be saved to graphical_models/datasets
    """
    pass

if __name__=="__main__":
    # parse arguments and dataset name
    # construct graphical models
    # label them using a chosen algorithm
    # save to graphical_models/datasets
    pass
