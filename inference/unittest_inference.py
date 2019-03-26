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
import torch
import os
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
        print("exact")
        print(exact.run([self.graph2]))
        #exact.reset_mode("map")
        #print(exact.run([self.graph]))

    def test_bp(self):
        # BP fails on n=2 and n=3 star (on fully-conn n=3 - ok)
        bp = get_algorithm("bp")("marginal")
        res = bp.run([self.graph2], use_log=True)
        print("bp")
        print(res)

    def test_mcmc(self):
        mcmc = get_algorithm("mcmc")("marginal")
        res = mcmc.run([self.graph2])
        print("mcmc")
        print(res)

    def test_gnn(self):
        # print("Testing GNN constructor")

        # GGNN parmeters
        graph = self.graph
        n_nodes = graph.W.shape[0]
        n_hidden_states = 5
        message_dim_P = 5
        hidden_unit_message_dim = 64 
        hidden_unit_readout_dim = 64
        T = 10
        learning_rate = 1e-2
        epochs = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gnn_constructor = get_algorithm("gnn_inference")
        exists = os.path.isfile('pretrained/gnn_model.pt')
        if(exists):
            gnn_inference = gnn_constructor('marginal', n_nodes, n_hidden_states, 
                message_dim_P,hidden_unit_message_dim, hidden_unit_readout_dim, T,'pretrained/gnn_model.pt')
            
            out = gnn_inference.run(graph,device)
            print('gnn')
            print(out)
        else:
            print('pretrained model needed')


if __name__ == "__main__":
    unittest.main()
