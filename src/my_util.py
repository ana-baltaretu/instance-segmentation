import numpy as np
import math
import cv2
import shutil
import os
import random


def make_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_pair = (kernel_size, kernel_size)
    return kernel, kernel_pair


def contrast_stretch(input_image, total_entries):
    stretched_mat = input_image.copy()
    frame_width, frame_height = stretched_mat.shape
    for x in range(frame_height):
        for y in range(frame_width):
            stretched_mat[y][x] = math.ceil(stretched_mat[y][x] / total_entries * 255)
    return stretched_mat


def resize_image(img, percentage):
    """
    img = the image you want to resize
    percentage = how big you want it to be
    (ex: 20 for making the image 5 times smaller)
    (ex: 200 for making the image 2 times larger)
    """
    w = int(img.shape[1] * percentage / 100)
    h = int(img.shape[0] * percentage / 100)
    dim = (w, h)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def split_train_test_validation(input_path, output_path, cleanup=False, train_data_percentage=0.8):
    """
    NMNIST is downloaded as a folder with 2 sub-folders ('Train' and 'Test')
    This method splits the 'Train' folder into 'training' and 'testing' folders (which
    will be used for training) and the 'Test' folder becomes the 'validation' set.
    Result is in the N_MNIST folder.
    """

    input_train_path = os.path.join(input_path, 'Train')
    input_test_path = os.path.join(input_path, 'Test')

    assert os.path.exists(input_path)
    assert os.path.exists(input_train_path)
    assert os.path.exists(input_test_path)

    # Reset data folder
    if cleanup is True and os.path.exists(output_path):
        shutil.rmtree(output_path)

    # If it already exists don't generate it again
    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        return

    # Make directory if it doesn't exist
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # Copy 'Test' to 'validation'
    print(input_test_path)
    print(os.path.join(output_path, 'validation'))
    shutil.copytree(input_test_path, os.path.join(output_path, 'validation'))
    print("Copied!")

    # Generate directories if they don't exist
    subdirs = ['training', 'testing']
    for subdir in subdirs:
        subdir_path = os.path.join(output_path, subdir)
        print(subdir_path)
        if os.path.exists(subdir_path) is False:
            os.mkdir(subdir_path)

    for number_folder in os.listdir(input_train_path):
        input_number_path = os.path.join(input_train_path, number_folder)

        training_path = os.path.join(os.path.join(output_path, 'training'), number_folder)
        if os.path.exists(training_path) is False:
            os.mkdir(training_path)

        testing_path = os.path.join(os.path.join(output_path, 'testing'), number_folder)
        if os.path.exists(testing_path) is False:
            os.mkdir(testing_path)

        print(input_number_path)
        print(training_path)
        print(testing_path)
        print()

        for file in os.listdir(input_number_path):
            rand = random.random()
            current_file_path = os.path.join(input_number_path, file)
            if rand < train_data_percentage:
                shutil.copyfile(current_file_path, os.path.join(training_path, file))
                # print('training')
            else:
                shutil.copyfile(current_file_path, os.path.join(testing_path, file))
                # print('testing')







