"""

Unit tests for inference objects

TODO:
* add actual tests

"""
import numpy as np
from inference import get_algorithm
from graphical_models import BinaryMRF


if __name__ == "__main__":
    W = np.ones((2,2))
    W[[0,1], [0,1]] = 0
    b = np.ones(2) * 0.5
    graph = BinaryMRF(W, b)

    print("Testing BP constructor")
    bp = get_algorithm("bp")("map")
    print(bp)

    res =bp.run([graph])
    print(res)

    print("Testing Exact constructor")
    exact = get_algorithm("exact")("marginal")
    print(exact)

    print("Testing MCMC constructor")
    mcmc = get_algorithm("mcmc")("map")
    print(mcmc)

    # print("Testing GNN constructor")
    # gnn = get_algorithm("gnn_inference")("map", OTHER ARGS)
    # print(gnn)