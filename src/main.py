from src.my_data_generation import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_dataset_merged(dir_path):
    list_of_files = sorted(filter(lambda x:
                                  os.path.isfile(os.path.join(dir_path, x)) and
                                  x.endswith(".npy"), os.listdir(dir_path)))

    arrays = None
    for file in list_of_files:
        file_path = os.path.join(dir_path, file)

        a = np.load(file_path, allow_pickle=True)
        if arrays is None:
            arrays = a
        else:
            arrays = np.concatenate((arrays, a))

    return arrays


if __name__ == '__main__':
    train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
    test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    print("Loading noisy dataset!")

    noisy_train_dataset = load_dataset_merged('../data/NMNIST_noise/Train')
    noisy_test_dataset = load_dataset_merged('../data/NMNIST_noise/Test')

    print("Loaded everything!")

    generate_rgbd_images_and_masks(train_dataset, test_dataset,
                                   noisy_train_dataset, noisy_test_dataset,
                                   '../data/N_MNIST_alex_new', cleanup=True, skip=50)
