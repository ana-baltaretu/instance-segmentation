from my_data_generation import *
from event_datasets import custom_transforms as ct

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
    test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)
    ncaltech101_dataset = tonic.datasets.NCALTECH101(save_to='../data')

    print("Train:")
    new_train = ct.combine_datasets(nmnist=train_dataset, ncaltech101=ncaltech101_dataset)
    # print(type(np.array(new_train, dtype=object)))
    np.save("../data/alex_data/new_train", np.array(new_train, dtype=object), allow_pickle=True)
    
    
    # l = np.load("../data/alex_data/new_train.npy", allow_pickle=True)
    # print(l)
    
    # print("Test:")
    # new_test = ct.combine_datasets(nmnist=test_dataset, ncaltech101=ncaltech101_dataset)
    # print(new_test)
    # np.save("../data/alex_data/new_test", new_test, allow_pickle=True)




    # Masks
    # split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    # generate_rgbd_images_and_masks(train_dataset, test_dataset, '../data/N_MNIST_images_Alex', cleanup=True, skip=1000)
