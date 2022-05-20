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


def generate_directories(output_path, subdirs):
    """
    Generate directories if they don't exist
    :param output_path: path to parent folder
    :param subdirs: what subdirectories need to be generated
    """
    for subdir in subdirs:
        subdir_path = os.path.join(output_path, subdir)
        print(subdir_path)
        if os.path.exists(subdir_path) is False:
            os.mkdir(subdir_path)


def cleanup_files(output_path, cleanup):
    # Reset data folder
    if cleanup is True and os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Make directory if it doesn't exist
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)





