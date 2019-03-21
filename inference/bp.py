"""

Approximate inference using Belief Propagation
Here we can rely on some existing library,
for example https://github.com/mbforbes/py-factorgraph
Authors: kkorovin@cs.cmu.edu

"""

from inference.core import Inference


class BeliefPropagation(Inference):
    """
    A special case implementation of BP
    for binary MRFs.
    """
    def run_one(self, graph):
        """
        Use a closed-form updates:
        - sum-product for marginals,
        - max-product for map
        """
        raise NotImplementedError("TODO")

    def run(self, graphs):
        res = []
        for graph in graphs:
            res.append(self.run_one(graph))
        return res
