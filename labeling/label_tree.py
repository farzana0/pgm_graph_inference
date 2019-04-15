"""
Tree-based labeling
@author: kkorovin@cs.cmu.edu
"""

class LabelTree:
    def __init__(self, inf_algo):
        self.inf_algo = inf_algo  # already knows about mode
                                  # (map/marginal)

    def run_one(self, graph):
        # extract MST with the same node order
        tree = graph.get_max_abs_spanning_tree()
        # label the tree
        labels = self.inf_algo.run([tree])[0]
        return labels

    def run(self, graphs):
        res = []
        for graph in graphs:
            res.append(self.run_one(graph))
        return res
