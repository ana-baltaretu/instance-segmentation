import numpy as np
import math
import cv2


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










