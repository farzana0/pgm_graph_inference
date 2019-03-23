"""

Experiment specifications:
an experiment is defined by train,test dataset pair,
each dataset is loaded from graphical_models/datasets.
Authors: kkorovin@cs.cmu.edu

"""

import os
import numpy as np

from graphical_models import BinaryMRF

data_dir = "./graphical_models/datasets"
# Give specs in form structure->size
data_specs = {
    "debug": 
            {"star": [5],
              "fc":   []}
    "debug_test":
            {"star": [5],
              "fc":   []}
}


# Data loading ----------------------------------------------------------------
def get_dataset_by_name(specs_name, 
    data_dir="./graphical_models/datasets/"):
    """
    Assumes graphs live as
    graphical_models/datasets/
        |-- star/
        |    |-  9/<file1.npy>, <file2.npy> ...
        |    |- 10/
             |- 11/
       ...  ...
    Loads all graphs of given size and structure,
    this needs to be updated in the future
    (so that we can train and test on the same structures).

    """
    if specs_name not in data_specs:
        raise ValueError("Specification {} not supported".format(exp_name))
    specs = data_specs[specs_name]
    graphs = []
    for struct in specs:
        size_list = specs[struct]
        for size in size_list:
            # go to specified dir, load and append
            directory = os.path.join(data_dir, struct, str(size))

            for filename in os.listdir(directory):
                if filename.endswith(".npy"):
                    path_to_graph = os.path.join(directory, filename)
                    data_dict = np.load(path_to_graph)[()]  # funny indexing
                    graph = BinaryMRF(data_dict["W"], data_dict["b"])
                    graph.set_ground_truth(marginal_est=data_dict["marginal"],
                                           map_est=data_dict["map"])
                    graphs.append(graph)

    print("Loaded {} graphs".format(len(graphs)))
    return graphs

# Train-test pairs ------------------------------------------------------------
def dummy_experiment():
    """ Just for an example """

    # TODO: how to organize t/t splitting?
    train_specs_name = "debug"
    test_specs_name = "debug"

    train_data = get_dataset_by_name(train_specs_name)
    test_data = get_dataset_by_name(test_specs_name)

    # load the model with train_specs_name postfix from inference/pretrained
    # TODO: gnn = 

    # TODO: check accuracy of different types of inference


# Some simple checks ----------------------------------------------------------
if __name__ == "__main__":
    train_data = get_dataset_by_name("debug")
    print(train_data[0])
    print("W, b:", train_data[0].W, train_data[0].b)
    print("Marginals:", train_data[0].marginal)
    print("MAP:", train_data[0].map)

