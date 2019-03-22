"""

Unit tests for inference objects

TODO:
* add actual tests

"""

from inference import get_algorithm


if __name__ == "__main__":
    print("Testing BP constructor")
    bp = get_algorithm("bp")("map")
    print(bp)

    print("Testing Exact constructor")
    exact = get_algorithm("exact")("marginal")
    print(exact)

    print("Testing MCMC constructor")
    mcmc = get_algorithm("mcmc")("map")
    print(mcmc)

    # print("Testing GNN constructor")
    # gnn = get_algorithm("gnn_inference")("map", OTHER ARGS)
    # print(gnn)