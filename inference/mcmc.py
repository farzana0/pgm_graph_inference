"""

Implements MCMC inference procedures.
Authors: kkorovin@cs.cmu.edu

TODO:
* implement Gibbs sampling
* (or MH in the case Gibbs is not possible)
"""

from inference.core import Inference


class GibbsSampling(Inference):
    def run(self, graphs):
        raise NotImplementedError
