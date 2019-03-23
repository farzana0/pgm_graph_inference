# Inference in graphical models with GNNs

*Authors: Ksenia Korovina, Lingxiao Zhao, Mark Cheung, Wenwen Si*

## Structure of the repo

* `graphical_models` directory contains definitions of `GraphicalModel` class (objects abbreviated as "graphs"); `graphical_models/datasets` contains labeled data. Labeled graphs are stored as `.npy` files in the following directory structure:
```
    graphical_models/datasets/
        |-- star/
        |    |-  9/<file1.npy>, <file2.npy> ...
        |    |- 10/
             |- 11/
       ...  ...
```
* `inference` directory contains all methods for performing inference on a `GraphicalModel`, including BP and GNN-based methods; `inference/pretrained` contains pretrained GNN inference procedures.
* `experiments` directory contains specification for loading data (combinations of graph structures and sizes to compose training and testing) and inference experiments. If an experiment uses GNN inference, it first checks if an appropriate pretrained model exists (using `train.py`) by matching model postfix and `train_specs_name` in experiment.
* `create_data.py` generates graphs from user-specified arguments and saves to `graphical_models/datasets` by default.
* `train.py` uses one of generated datasets to train a GNN inference procedure (such as GatedGNN).

## Getting started

For imports to work correctly, add root of the repository to `PYTHONPATH` by running

```bash
source setup.sh
```

## References

[ICLR18](https://arxiv.org/abs/1803.07710)