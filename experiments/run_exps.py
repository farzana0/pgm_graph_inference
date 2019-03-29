"""

Runnable experiments module
Authors: kkorovin@cs.cmu.edu

TODO: need to match GNN parameters with those in train.py

"""

import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

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
    test_set_name = struct + "_medium" # "conn_medium", "trees_medium"
    run_experiment(train_set_name, test_set_name)

def upscaling_experiment(struct):
    """ trainset here combines a few structures,
        testset is increasingly large 
    """
    train_set_name = struct + "_small"
    test_set_name  = struct + "_large"
    run_experiment(train_set_name, test_set_name)

def in_sample_experiment_map(struct):
    train_set_name = struct + "_small"
    test_set_name  = struct + "_small"
    run_experiment(train_set_name, test_set_name, "map")

# Runner ----------------------------------------------------------------------

def run_experiment(train_set_name, test_set_name, inference_mode="marginal",
                   base_data_dir=DFLT_DATA_DIR, model_base_dir=DFLT_MODEL_DIR):
    """
    tests for in-sample (same structure, same size, marginals)
    """
    train_path = os.path.join(base_data_dir, "train")
    test_path = os.path.join(base_data_dir, "test")
    model_load_path =os.path.join(model_base_dir, train_set_name)

    # train_data = get_dataset_by_name(train_set_name, train_path)
    test_data  = get_dataset_by_name(test_set_name, test_path, mode=inference_mode)
 
    # load model
    n_hidden_states = 5
    message_dim_P = 5
    hidden_unit_message_dim = 64 
    hidden_unit_readout_dim = 64
    T = 10
    gnn_constructor = get_algorithm("gnn_inference")
    gnn_inference = gnn_constructor(inference_mode, n_hidden_states, 
                                    message_dim_P,hidden_unit_message_dim,
                                    hidden_unit_readout_dim, T,
                                    model_load_path, USE_SPARSE_GNN)

    # run inference on test
    times = {}

    t0 = time()
    gnn_res = gnn_inference.run(test_data, DEVICE)
    times["gnn"] = (time()-t0) / len(test_data)
    
    t0 = time()
    bp = get_algorithm("bp")(inference_mode)
    bp_res = bp.run(test_data, use_log=True, verbose=False)
    times["bp"] = (time()-t0) / len(test_data)

    t0 = time()
    mcmc = get_algorithm("mcmc")(inference_mode)
    mcmc_res = mcmc.run(test_data)
    times["mcmc"] = (time()-t0) / len(test_data)

    #--- sanity check ----#
    #exact = get_algorithm("exact")("marginal")
    #exact_res = exact.run(test_data)
    #--- sanity check ----#

    # all loaded graphs have ground truth set
    if inference_mode == "marginal":
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

        #--- sanity check ----#
        # exact_labels = []
        # for graph_res in exact_res:
        #     exact_labels.extend(list(m[1] for m in graph_res))
        #--- sanity check ----#

        plt.title("Inference results")
        #fig, axes = plt.subplots(nrows=1, ncols=3)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(30, 10))
        ax1.set_title("GNN", fontsize=40)
        ax1.scatter(true_labels, gnn_labels)
        ax2.set_title("BP", fontsize=40)
        ax2.scatter(true_labels, bp_labels)
        ax3.set_title("MCMC", fontsize=40)
        ax3.scatter(true_labels, mcmc_labels)

        #--- sanity check ----#
        #ax4.set_title("Exact (just a sanity check)")
        #ax4.scatter(true_labels, exact_labels)
        #--- sanity check ----#
        plt.savefig("./experiments/res_{}_{}.png".format(train_set_name, test_set_name))

    # MAP: only numeric
    else:
        true_labels = []
        for g in test_data:
            true_labels.extend(g.map)
        true_labels = np.array(true_labels)

        gnn_labels = []
        for graph_res in gnn_res:
            gnn_labels.extend(list(-1 if m[0]>m[1] else +1 for m in graph_res))
        gnn_labels = np.array(gnn_labels)
        gnn_accuracy = np.mean(true_labels == gnn_labels)

        bp_labels = []
        for graph_res in bp_res:
            bp_labels.extend(graph_res)
        bp_labels = np.array(bp_labels)
        bp_accuracy = np.mean(true_labels == bp_labels)

        mcmc_labels = []
        for graph_res in mcmc_res:
            mcmc_labels.extend(graph_res)
        mcmc_labels = np.array(mcmc_labels)
        mcmc_accuracy = np.mean(true_labels == mcmc_labels)

        print("Accuracies: GNN {}, BP {}, MCMC {}".format(gnn_accuracy, bp_accuracy, mcmc_accuracy))

    print("Runtimes", times)


if __name__ == "__main__":
    # in_sample_experiment(struct="path")
    # out_of_sample_experiment("bipart")
    # upscaling_experiment("fc")
    in_sample_experiment_map(struct="path")

