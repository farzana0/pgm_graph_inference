# Inference in graphical models with GNNs

*Authors: Ksenia Korovina, Lingxiao Zhao, Mark Cheung, Wenwen Si*

## Structure of the repo

* `graphical_models` directory contains definitions of `GraphicalModel` class (objects abbreviated as "graphs"), data generation and manipulation procedures; `graphical_models/datasets` contains labeled data
* `inference` directory contains all methods for performing inference on a `GraphicalModel`, including BP and GNN-based methods; `inference/pretrained` contains pretrained GNN inference procedures
* `train.py` uses one of generated datasets to train a GNN inference procedure (such as GatedGNN)
* `experiments` directory contains inference experiments. If an experiment uses GNN inference, it first checks if an appropriate pretrained model has been trained (using `train.py`).

## Getting started

For imports to work correctly, add root of the repository to `PYTHONPATH` by running

```bash
source setup.sh
```

## References

[ICLR18](https://arxiv.org/abs/1803.07710)