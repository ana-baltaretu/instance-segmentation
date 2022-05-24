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
    print(crop_image.shape)
    return crop_image


# TODO: Make masks from MNIST to be labels for N-MNIST

train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_y = np.array(train_y)

positions_of_index = np.where(train_y == 0)

print('Len:', len(train_y[positions_of_index]))
print(train_y[positions_of_index])


print(len(train_dataset))
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

train_X_clone = train_X[positions_of_index]

for i in range(5):
    ind = random.randint(0, 100)
    frames, colorized_masks, target, time_frames = generate_masks(train_dataset[ind])

    (thresh, mask) = cv2.threshold(train_X_clone[ind], 125, 255, cv2.THRESH_BINARY)  # 36
    cropped = remove_zero_pad(mask)
    # cv2.imshow('beforemask', train_X_clone[ind])
    cv2.imshow('mask', mask)
    cv2.imshow('cropped', cropped)
    show_events(frames, 'frames/mask_matching')







