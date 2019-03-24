"""

Exact inference
Authors: kkorovin@cs.cmu.edu

"""

import numpy as np
import itertools
from inference.core import Inference


class ExactInference(Inference):
    """ Special case BinaryMRF implementation """
    def run_one(self, graph):
        W = graph.W
        b = graph.b
        n = graph.n_nodes

        print(W, b)

        # compute joint probabilities
        # array of shape [2,...,2]
        log_potentials = np.zeros([2]*n)
        for state in itertools.product([0, 1], repeat=n):
            state_ind = np.array(state)
            state_val = 2 * state_ind - 1
            log_potentials[state_ind] = state_val.dot(W.dot(state_val)) + b.dot(state_val)
        probs = np.exp(log_potentials)
        probs /= probs.sum()
        # probs now contain probabilities of all states

        # print("M1:", probs[0, :, :].sum(), probs[1, :, :].sum())
        # print("M2:", probs[:, 0, :].sum(), probs[:, 1, :].sum())

        if self.mode == "marginal":
            # select one state and compute marginal:
            marginals = np.zeros((n, 2))  # [i, 0] is P(x_i=0)
            for i in range(n):
                axes = tuple(j for j in range(n) if j != i)
                marginal = probs.sum(axis=axes)
                marginals[i] = marginal
            return marginals

        elif self.mode == "map":
            binary_ind = np.unravel_index(probs.argmax(),
                                          probs.shape)
            return 2 * np.array(binary_ind) - 1

    def run(self, graphs):
        res = []
        for graph in graphs:
            res.append(self.run_one(graph))
        return res
