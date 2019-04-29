import numpy as np
import argparse

def parse_train_args():
    parser = argparse.ArgumentParser()

    # critical arguments, change them
    parser.add_argument('--data_file', type=str,
                        help='name of data file')
    parser.add_argument('--map_threshold', default=0.5, type=float,
                        help='threshold for MAP')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_train_args()
    data = np.load(args.data_file).item(0)
    # print(data.keys())

    # print(data['true_labels'])

    MAP_true = np.array(data['true_labels']) > args.map_threshold

    GNN_true = np.array(data['gnn_labels']) > args.map_threshold

    BP_true = np.array(data['bp_labels']) > args.map_threshold
    MCMC_true = np.array(data['mcmc_labels']) > args.map_threshold

    number_of_samples =MAP_true.shape[0]

    print('GNN MAP Accuracy: ', np.sum(GNN_true==MAP_true)/number_of_samples)

    print('BP MAP Accuracy: ', np.sum(BP_true==MAP_true)/number_of_samples)

    print('MCMC MAP Accuracy: ', np.sum(MCMC_true==MAP_true)/number_of_samples)

