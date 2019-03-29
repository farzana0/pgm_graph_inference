"""

Runnable experiments module
Authors: kkorovin@cs.cmu.edu

TODO: need to match GNN parameters with those in train.py

"""

import os
from time import time
import matplotlib.pyplot as plt

from inference import get_algorithm
from experiments.exp_helpers import get_dataset_by_name
from constants import *


# Train-test pairs ------------------------------------------------------------

def in_sample_experiment(struct):
    train_set_name = struct + "_small"
    test_set_name  = struct + "_small"
    run_experiment(train_set_name, test_set_name)

def out_of_sample_experiment(struct):
    """ Test generalization to same- and different- structure
        larger graphs """
    train_set_name = struct + "_small"
    test_set_name = "trees_medium"  # or "conn_medium"
    run_experiment(train_set_name, test_set_name)


# Runner ----------------------------------------------------------------------

def run_experiment(train_set_name, test_set_name, base_data_dir=DFLT_DATA_DIR, model_base_dir=DFLT_MODEL_DIR):
    """
    tests for in-sample (same structure, same size, marginals)
    """
    train_path = os.path.join(base_data_dir, "train")
    test_path = os.path.join(base_data_dir, "test")
    model_load_path =os.path.join(model_base_dir, train_set_name)

    # train_data = get_dataset_by_name(train_set_name, train_path)
    test_data  = get_dataset_by_name(test_set_name, test_path)
 
    # load model
    n_hidden_states = 5
    message_dim_P = 5
    hidden_unit_message_dim = 64 
    hidden_unit_readout_dim = 64
    T = 10
    gnn_constructor = get_algorithm("gnn_inference")
    gnn_inference = gnn_constructor("marginal", n_hidden_states, 
                                    message_dim_P,hidden_unit_message_dim,
                                    hidden_unit_readout_dim, T,
                                    model_load_path, USE_SPARSE_GNN)

    # run inference on test
    times = {}

    t0 = time()
    gnn_res = gnn_inference.run(test_data, DEVICE)
    times["gnn"] = (time()-t0) / len(test_data)
    
    t0 = time()
    bp = get_algorithm("bp")("marginal")
    bp_res = bp.run(test_data, use_log=True, verbose=False)
    times["bp"] = (time()-t0) / len(test_data)

    t0 = time()
    mcmc = get_algorithm("mcmc")("marginal")
    mcmc_res = mcmc.run(test_data)
    times["mcmc"] = (time()-t0) / len(test_data)

    #--- sanity check ----#
    #exact = get_algorithm("exact")("marginal")
    #exact_res = exact.run(test_data)
    #--- sanity check ----#

    # all loaded graphs have ground truth set
    true_labels = []
    for g in test_data:
        true_labels.extend(list(m[1] for m in g.marginal))

    gnn_labels = []
    for graph_res in gnn_res:
        gnn_labels.extend(list(m[1] for m in graph_res))

    bp_labels = []
    for graph_res in bp_res:
        bp_labels.extend(list(m[1] for m in graph_res))

    mcmc_labels = []
    for graph_res in mcmc_res:
        mcmc_labels.extend(list(m[1] for m in graph_res))

    # exact_labels = []
    # for graph_res in exact_res:
    #     exact_labels.extend(list(m[1] for m in graph_res))

    plt.title("Inference results")
    #fig, axes = plt.subplots(nrows=1, ncols=3)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(30, 10))
    ax1.set_title("GNN", fontsize=20)
    ax1.scatter(true_labels, gnn_labels)
    ax2.set_title("BP", fontsize=20)
    ax2.scatter(true_labels, bp_labels)
    ax3.set_title("MCMC", fontsize=20)
    ax3.scatter(true_labels, mcmc_labels)

    #--- sanity check ----#
    #ax4.set_title("Exact (just a sanity check)")
    #ax4.scatter(true_labels, exact_labels)
    #--- sanity check ----#

    plt.savefig("./experiments/res_{}_{}.png".format(train_set_name, test_set_name))

    print("Runtimes", times)

if __name__ == "__main__":
    in_sample_experiment(struct="fc")

