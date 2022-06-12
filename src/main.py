from src.my_data_generation import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
    train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
    test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    generate_rgbd_images_and_masks(train_dataset, test_dataset, '../data/N_MNIST_images_20ms_skip_50', cleanup=True, skip=50)
