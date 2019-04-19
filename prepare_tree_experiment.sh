# make test data
# python create_data.py --graph_struct random_tree --size_range 100_100 \
#                       --num 500 --data_mode test --mode marginal --algo bp \
#                       --verbose True

# make label-propagation labels for training
python create_data.py --graph_struct random_tree --size_range 100_100 \
                      --num 1500 --data_mode train --mode marginal --algo label_prop_exact_20 \
                      --verbose True
