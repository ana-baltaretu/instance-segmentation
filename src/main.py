import time
from my_data_generation import *
# from event-datasets import custom_transforms as ct
from event_datasets import custom_transforms as ct

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    ALEX_ENV = True

    # if ALEX_ENV:
    # train_dataset, test_dataset = ct.get_sorted_datasets(train_dataset_path, test_dataset_path)
    # train_dataset = ct.load_dataset_merged(ct.TRAIN_MERGED_PATH)
    # test_dataset = ct.load_dataset_merged(ct.TEST_MERGED_PATH)
    print("Successfully loaded datasets.")
    # else:
    original_nmnist_train = tonic.datasets.NMNIST(save_to='../data', train=True)
    original_nmnist_test = tonic.datasets.NMNIST(save_to='../data', train=False)
    # original_nmnist_train, original_nmnist_test = ct.get_sorted_datasets(ct.TRAIN_PATH, ct.TEST_PATH)
    

    # split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    generate_rgbd_images_and_masks(original_nmnist_train, original_nmnist_test, '../data/N_MNIST_skip_50', original_nmnist_test, original_nmnist_train, cleanup=True, skip=50)

