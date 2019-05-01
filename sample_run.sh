
#!/bin/bash
# Sample run with full pipeline
# run as bash run_small.sh 

echo -e "\tCreating train and test dataset with path structure and 9 nodes (small)"
echo -e "\tNote you only need to run create_data and train.py once"
python create_data.py --graph_struct path --size_range 9_9 \
                      --num 1300 --data_mode train --mode marginal --algo exact \
                      --verbose True
python create_data.py --graph_struct path --size_range 9_9 \
                      --num 300 --data_mode test --mode marginal --algo exact \
                      --verbose True

echo -e "\tTraining dataset (see run_exps.py for possible train->test, and exp_helpers for their defintiions)"
echo -e "\t For example, trees_medium contains "star": [15, 16, 17], and "path": [15, 16, 17] where 15,16,17 are number of nodes"   

python train.py --train_set_name path_small --mode marginal --epochs 5 --verbose True

echo -e "Generate plots for bp, MCMC, and GNN for the given experiment in experiments folder and saved the output in experiments/saved_exp_res"
python ./experiments/run_exps.py --exp_name in_sample_path

echo -e "compute MAP for bp, MCMC, and gnn"
python ./experiments/saved_exp_res/compute_MAP_accuracy --data_file res_path_small_path_small.npy
