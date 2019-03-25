"""

Unit tests for inference objects
Authors: kkorovin@cs.cmu.edu

TODO:
* There is a large mismatch between MCMC and exact on this example
  find what is going wrong.

"""
import numpy as np
import unittest

from inference import get_algorithm
from graphical_models import construct_binary_mrf 


class TestInference(unittest.TestCase):
    def setUp(self):
        self.graph = construct_binary_mrf("star", n_nodes=5)
        self.graph2 = construct_binary_mrf("fc", n_nodes=5)

    def test_exact_probs(self):
        graph = construct_binary_mrf("fc", 3)
        # compute probs:
        probs = np.zeros((2,2,2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    state = 2*np.array([i,j,k])-1
                    probs[i, j, k] = state.dot(graph.W.dot(state)) + graph.b.dot(state)
        probs = np.exp(probs)
        probs /= probs.sum()

        exact = get_algorithm("exact")("marginal")
        exact_probs = exact.compute_probs(graph.W, graph.b, graph.n_nodes)
        assert np.allclose(probs, exact_probs)

    def test_exact(self):
        # check probs computation
        exact = get_algorithm("exact")("marginal")
        print(exact.run([self.graph]))
        #exact.reset_mode("map")
        #print(exact.run([self.graph]))

    def test_bp(self):
        # BP fails on n=2 and n=3 star (on fully-conn n=3 - ok)
        bp = get_algorithm("bp")("marginal")
        res = bp.run([self.graph])
        print(res)

    def test_mcmc(self):
        mcmc = get_algorithm("mcmc")("marginal")
        res = mcmc.run([self.graph])
        print(res)

    def test_gnn(self):
        # print("Testing GNN constructor")
        # gnn = get_algorithm("gnn_inference")("map", OTHER ARGS)
        # print(gnn)
        pass


if __name__ == "__main__":
    unittest.main()
