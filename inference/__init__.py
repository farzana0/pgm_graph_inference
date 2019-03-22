"""

Interface for inference algorithms
Authors: kkorovin@cs.cmu.edu

"""

from inference.bp import *
from inference.gnn_inference import *
from inference.exact import *
from inference.mcmc import *

def get_algorithm(algo_name):
    """ Returns a constructor """
    if algo_name == "bp":
        return BeliefPropagation
    elif algo_name == "gnn_inference":
        return GatedGNNInference
    elif algo_name == "exact":
        return ExactInference
    elif algo_name == "mcmc":
        return GibbsSampling
    else:
        raise ValueError("Inference algorithm {} not supported".format(algo_name))