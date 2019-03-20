"""

Interface for inference algorithms
Authors: kkorovin@cs.cmu.edu

"""

from inference.bp import *
from inference.gnn_inference import *

def get_algorithm(algo_name):
    if algo_name == "bp":
        return BeliefPropagation()
    elif algo_name == "gnn_inference":
        return GatedGNNInference()
    else:
        raise ValueError("Inference algorithm {} not supported".format(algo_name))