
#!/bin/bash
# Sample run with full pipeline
# run as run_small.sh 

python create_data.py --graph_struct path --size_range 9_9 \
                      --num 1300 --data_mode train --mode marginal --algo exact \
                      --verbose True
python create_data.py --graph_struct path --size_range 9_9 \
                      --num 300 --data_mode test --mode marginal --algo exact \
                      --verbose True

python train.py --train_set_name path_small --mode marginal --epochs 5 --verbose True


python ./experiments/run_exps.py --exp_name in_sample_path
