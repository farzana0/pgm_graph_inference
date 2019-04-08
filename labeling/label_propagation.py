"""
Approximate labeling class with Label Propagation.

A subgraph is chosen and labeled using a passed algorithm,
then labels are propagated to the rest of the graph.

@author: kkorovin@cs.cmu.edu
"""

import numpy as np
import networkx as nx
import warnings


class LabelProp:
    """ Adapted from LabelPropagation in sklearn """
    def __init__(self, sg_size, inf_algo,
                max_iter=50, tol=1e-3):
        """Constructor.
        Arguments:
            sg_size {int} -- size of subgraph to sample
                             for running inf_algo
            inf_algo {Inference} -- Inference object
                                 to run on subgraph
            max_iter {int} -- max number of propagation iterations
            tol {float} -- early stopping criterion
        """
        self.sg_size  = sg_size
        self.inf_algo = inf_algo  # already knows about the mode
                                  # (marginal/map)
        # set label prop params:
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_ = 0

    def run_one(self, graph):
        n_nodes, n_classes = len(graph.W), 2

        # choose a random subgraph graphical model
        nodes = np.random.choice(n_nodes, min(self.sg_size, n_nodes), replace=False)
        nodes = sorted(nodes)
        sg = graph.get_subgraph_on_nodes(nodes)

        # labeled_distr = self.inf_algo.run([graph])[0]
        labeled_distr = self.inf_algo.run([sg])[0]
        graph_matrix = graph.W
        unlabeled = np.array([(i not in nodes) for i in range(n_nodes)])  #(y == -1)

        # initialize distributions
        self.label_distributions_ = np.ones((n_nodes, n_classes)) * 1/n_classes
        self.label_distributions_[nodes] = labeled_distr

        y_static = np.copy(self.label_distributions_)
        y_static[unlabeled] = 0

        l_previous = np.zeros((n_nodes, n_classes))
        unlabeled = unlabeled[:, np.newaxis]
 
        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_
            self.label_distributions_ = np.dot(graph_matrix, self.label_distributions_)

            normalizer = np.sum(
                self.label_distributions_, axis=1)[:, np.newaxis]
            assert np.all(normalizer != 0.), normalizer
            self.label_distributions_ /= normalizer
            self.label_distributions_ = np.where(unlabeled,
                                                 self.label_distributions_,
                                                 y_static)

        else:
            warnings.warn(
                f'max_iter={self.max_iter} was reached without convergence.'
            )
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        self.label_distributions_ /= normalizer
        return self.label_distributions_

    def run(self, graphs):
        res = []
        for graph in graphs:
            res.append(self.run_one(graph))
        return res

