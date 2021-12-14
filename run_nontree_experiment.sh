#!/bin/bash
# Runner of nontree_approx experiment

# source setup.sh  # just in case
if [ $1 == 'make_data' ]
# if true;
then
    # make test data: distributed through google drive
    # echo -e "\tCreating test dataset with MCMC labels"
    # python create_data.py --graph_struct path --size_range 5_5 \
    #                       --num 500 --data_mode test --mode marginal --algo mcmc \
    #                       --verbose True
    python create_data.py --graph_struct grid --size_range 49_49 \
                          --num 500 --data_mode test --mode marginal --algo mcmc \
                          --verbose True
    # python create_data.py --graph_struct grid --size_range 50_50 \
    #                       --num 500 --data_mode test --mode marginal --algo mcmc \
    #                       --verbose True
    # python create_data.py --graph_struct grid --size_range 100_100 \
    #                       --num 500 --data_mode test --mode marginal --algo mcmc \
    #                       --verbose True

    # make unlabeled training graphs: distributed through google drive
    echo -e "\tStarted generating graphs from given parameters"
    python create_data.py --graph_struct grid --size_range 49_49 \
                          --num 750 --data_mode train --mode marginal --algo none \
                          --verbose True --unlab_graphs_path grid_train
#     python create_data.py --graph_struct grid --size_range 50_50 \
#                           --num 750 --data_mode train --mode marginal --algo none \
#                           --verbose True --unlab_graphs_path path_train
#     python create_data.py --graph_struct grid --size_range 100_100 \
#                           --num 750 --data_mode train --mode marginal --algo none \
#                           --verbose True --unlab_graphs_path path_train
# #     echo -e "\tNow you need to merge the two .npys with lists of graphs into nontree_train.npy"
#     /opt/miniconda3/bin/python3 -c '
# import numpy as np;
# fc = np.load("./graphical_models/datasets/fc_train.npy");
# barb = np.load("./graphical_models/datasets/barbell_train.npy");
# mylist = list(barb) + list(fc);
# np.save("./graphical_models/datasets/nontrees_train.npy", mylist)
# '
elif [ $1 == 'make_labels' ]
# if true;
then
    read -p 'Choose labeling algo: 1) label_prop_exact_n1_..., 2) label_sg_louvain, 3) label_tree  ' label_algo
    if [ $label_algo == 'label_prop'* ]
    then
        #make label-propagation labels for training, use format label_prop_exact_10
        echo -e "\tStarting labeling with label propagation"
    elif [ $label_algo == 'label_sg'* ]
    then
        #make label-propagation labels for training, use format label_sg_Louvain
        echo -e "\tStarting labeling with subgraph labeling"
    elif [ $label_algo == 'label_tree' ]
    then
        echo -e "\tStarting labeling with spanning tree"
    fi
    # rm -rf ./graphical_models/datasets/train/barbell  # don't want duplicating graphs
    rm -rf ./graphical_models/datasets/train/wheel
    python create_data.py --data_mode train --mode marginal --algo $label_algo \
                          --verbose True --unlab_graphs_path grid_train # used to be: nontrees_train (all at once)

elif [ $1 == 'train' ]
# if true;
then
    echo -e "\tTraining your GNN"
    # pretraining stage
    python train.py --train_set_name grid_large --mode marginal --epochs 5 --verbose True

    # finetuning stage
    # python train.py --train_set_name barbell_approx --mode marginal --epochs 5 --verbose True --use_pretrained trees_approx
    # python train.py --train_set_name wheel_approx --mode marginal --epochs 5 --verbose True --use_pretrained trees_approx

elif [ $1 == 'test' ]
# if True:
then
    echo -e "\tRunning tests"
    # python ./experiments/run_exps.py  --exp_name res_barbell_approx
    python ./experiments/run_exps.py --exp_name in_sample_grid
    # python ./experiments/run_exps.py --exp_name res_wheel_approx
fi