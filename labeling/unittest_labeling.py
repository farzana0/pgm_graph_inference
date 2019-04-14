"""
Unit tests for approximate labeling
"""

import unittest
import numpy as np

from labeling import LabelProp
from graphical_models import construct_binary_mrf
from inference import get_algorithm

class TestInference(unittest.TestCase):
    def setUp(self):
        self.graph_star = construct_binary_mrf("star", n_nodes=10,
                                        shuffle_nodes=False)
        self.graph_fc = construct_binary_mrf("star", n_nodes=10,
                                        shuffle_nodes=False)

    def run_lbp_on_graph(self, graph):
        exact = get_algorithm("exact")("marginal")

        print("With subgraph of size 1")
        lbp = LabelProp(1, exact)
        res = lbp.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

        print("With subgraph of size 5")
        lbp = LabelProp(5, exact)
        res = lbp.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

        print("With subgraph of size 10")
        lbp = LabelProp(10, exact)
        res = lbp.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

    def test_label_prop(self):
        """ Testing marginal label_prop """
        self.run_lbp_on_graph(self.graph_star)
        self.run_lbp_on_graph(self.graph_fc)

    def run_tree_on_graph(self, graph):
        exact = get_algorithm("exact")("marginal")
        lbt = LabelTree(exact)

        res = lbt.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

    def test_tree_prop(self, graph):
        self.run_tree_on_graph(self.graph_star)
        self.run_tree_on_graph(self.graph_fc)


if __name__ == "__main__":
    unittest.main()
