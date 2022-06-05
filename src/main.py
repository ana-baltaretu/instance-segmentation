import time
from my_data_generation import *
# from event-datasets import custom_transforms as ct
from event_datasets import custom_transforms as ct

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    ALEX_ENV = True

    if ALEX_ENV:
        # train_dataset, test_dataset = ct.get_sorted_datasets(train_dataset_path, test_dataset_path)
        train_dataset = ct.load_dataset_merged(ct.TRAIN_MERGED_PATH)
        test_dataset = ct.load_dataset_merged(ct.TEST_MERGED_PATH)
        print(train_dataset)
        print(test_dataset)
        print("Successfully loaded datasets.")
    else:
        train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
        test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    # split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    generate_rgbd_images_and_masks(train_dataset, test_dataset, '../data/N_MNIST_images_Alex', cleanup=True, skip=50)

