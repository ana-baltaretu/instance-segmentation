from my_data_generation import *
# from event-datasets import custom_transforms as ct
from event_datasets import custom_transforms as ct

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    ALEX_ENV = True
    train_dataset_path = "../data/sorted_NMNIST/train_dataset.npy"
    test_dataset_path = "../data/sorted_NMNIST/test_dataset.npy"

    if ALEX_ENV:
        train_dataset, test_dataset = ct.get_sorted_datasets(train_dataset_path, test_dataset_path)
        print("Successfully loaded sorted datasets.")
    else:
        train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
        test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    ncaltech101_dataset = tonic.datasets.NCALTECH101(save_to='../data')
    

    ct.combine_datasets_parallel(nmnist=train_dataset, ncaltech101=ncaltech101_dataset)


    # noisy_train = ct.combine_datasets(nmnist=train_dataset[:50], ncaltech101=ncaltech101_dataset, save_path="../data/alex_data/noisy_train")

    # print("Train:")
    # noisy_train = ct.combine_datasets(nmnist=train_dataset[:50], ncaltech101=ncaltech101_dataset, save_path="../data/alex_data/noisy_train")
    # print(type(np.array(noisy_train, dtype=object)))
    # np.save("../data/alex_data/noisy_train", np.array(noisy_train, dtype=object), allow_pickle=True)
    

    # print("Test:")
    # new_test = ct.combine_datasets(nmnist=test_dataset, ncaltech101=ncaltech101_dataset)
    # print(new_test)
    # np.save("../data/alex_data/new_test", new_test, allow_pickle=True)



    # split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    # generate_rgbd_images_and_masks(train_dataset, test_dataset, '../data/N_MNIST_images_Alex', cleanup=True, skip=50)

