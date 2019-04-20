# make test data: distributed through google drive
# echo -e "\tCreating test dataset with BP labels"
# python create_data.py --graph_struct path --size_range 100_100 \
#                       --num 500 --data_mode test --mode marginal --algo bp \
#                       --verbose True

# make unlabeled training graphs: distributed through google drive
# echo -e "\tStarted generating graphs from given parameters"
# python create_data.py --graph_struct random_tree --size_range 100_100 \
#                       --num 1500 --data_mode train --mode marginal --algo none \
#                       --verbose True --unlab_graphs_path trees_train

# make label-propagation labels for training
echo -e "\tStarting labeling with label propagation"
python create_data.py --graph_struct random_tree --size_range 100_100 \
                      --num 1500 --data_mode train --mode marginal --algo label_prop_exact_10 \
                      --verbose True --unlab_graphs_path trees_train
