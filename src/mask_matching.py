import numpy as np

from src.my_visualize import *
from src.my_util import *
from src.my_event_frame_generation import *
from src.my_label_generation import *

from keras.datasets import mnist
from matplotlib import pyplot


def remove_zero_pad(image):
    dummy = np.argwhere(image != 0)
    max_y = dummy[:, 0].max() + 1
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max() + 1
    crop_image = image[min_y:max_y, min_x:max_x]
    # print(crop_image.shape)
    return crop_image


def get_mask_for_index_and_label(index, chosen_label, mnist_dataset, mask_indices_per_label):
    where_from = mnist_dataset[mask_indices_per_label[chosen_label]]
    # frames, colorized_masks, target, time_frames = generate_masks(train_dataset[index])
    (thresh, mask) = cv2.threshold(where_from[index], 125, 255, cv2.THRESH_BINARY)  # 36
    cropped = remove_zero_pad(mask)
    return cropped


# TODO: Make masks from MNIST to be labels for N-MNIST

train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# train_y = np.array(train_y)

labels = range(0, 10)
mask_indices_per_label_train, mask_indices_per_label_test = [], []
for label in labels:

    indices_with_this_label_train = np.where(train_y == label)
    mask_indices_per_label_train.append(indices_with_this_label_train)
    indices_with_this_label_test = np.where(test_y == label)
    mask_indices_per_label_test.append(indices_with_this_label_test)

    print('Label:', label, 'Len:', len(train_y[indices_with_this_label_train]))
    print(train_y[indices_with_this_label_train])
    print('Label:', label, 'Len:', len(test_y[indices_with_this_label_test]))
    # print(test_y[indices_with_this_label_test])

# print(len(train_dataset))
# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  ' + str(test_X.shape))
# print('Y_test:  ' + str(test_y.shape))

# for i in range(5):
#     ind = random.randint(0, 100)
#     cropped_image = get_mask_for_index_and_label(ind, 0, train_X, mask_indices_per_label_train)
#     cv2.imshow('cropped', cropped_image)
#     cv2.waitKey(200)

# Label: 0 Len: 5923
# Label: 1 Len: 6742
# Label: 2 Len: 5958
# Label: 3 Len: 6131
# Label: 4 Len: 5842
# Label: 5 Len: 5421
# Label: 6 Len: 5918
# Label: 7 Len: 6265
# Label: 8 Len: 5851
# Label: 9 Len: 5949





