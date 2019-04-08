"""
Approximate labeling class with Label Propagation.

A subgraph is chosen and labeled using a passed algorithm,
then labels are propagated to the rest of the graph.

@author: kkorovin@cs.cmu.edu
"""

import networkx as nx
# from sklearn.semi_supervised import LabelPropagation 
#   - does not work directly, adapt 
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/semi_supervised/label_propagation.py

class LabelProp:
    def __init__(self, sg_size, inf_algo):
        """Constructor.
        Arguments:
            sg_size {int} -- size of subgraph to sample
                             for running inf_algo
            inf_algo {[type]} -- Inference object
                                 to run on subgraph
        """
        self.sg_size  = sg_size
        self.inf_algo = inf_algo  # already knows about the mode
                                  # (marginal/map)

    def run_one(self, graph):
        n_nodes = len(graph.W)
        nx_graph = nx.from_numpy_matrix(graph.W)
        nodes = np.random.choice(min(self.sg_size, n_nodes), replace=False)
        sg = nx_graph.subgraph(nodes)  # random subgraph
        res = self.inf_algo.run([sg])[0]
        # TODO: see link
        full_res = np.zeros((n_nodes, 2))
        return full_res

    def run(self, graphs):
        for graph in graphs:
            res.append(self.run_one(graph))


if __name__ == "__main__":
    pass
