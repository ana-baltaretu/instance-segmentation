import math

import numpy as np
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import cv2

from src.visualize import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def make_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_pair = (kernel_size, kernel_size)
    return kernel, kernel_pair


def create_gray_summed_mat(cropped_frames, len_y, len_x):
    gray_summed_mat = cv2.cvtColor(np.zeros((len_y, len_x, 3), np.uint8), cv2.COLOR_BGR2GRAY)

    for cropped in cropped_frames:
        gray_scaled = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        x_arr, y_arr = np.where(gray_scaled > 0)
        for ind, x in enumerate(x_arr):
            y = y_arr[ind]
            gray_summed_mat[x][y] += 1

    return gray_summed_mat


def contrast_stretch(stretched_mat, total_frames):
    frame_width, frame_height = stretched_mat.shape
    for x in range(frame_height):
        for y in range(frame_width):
            stretched_mat[y][x] = math.ceil(stretched_mat[y][x] / total_frames * 255)
    return stretched_mat


def show_partial_matrices(closing_gray_summed_mat, stretched_mat,
                          th_gray_summed_mat, revert_resize, colorized_mask):
    cv2.imshow("closing_gray_summed_mat", closing_gray_summed_mat)
    cv2.imshow("stretched_mat", stretched_mat)
    cv2.imshow("th_gray_summed_mat", th_gray_summed_mat)
    cv2.imshow("revert_resize", revert_resize)
    cv2.imshow("colorized_mask", colorized_mask)
    cv2.waitKey(0)


def generate_average_mask(stretched_mat):
    resized = resize_image(stretched_mat, 2000)
    kernel_size = 11
    closing_gray_summed_mat = cv2.morphologyEx(resized, cv2.MORPH_CLOSE,
                                               make_kernel(kernel_size)[0], iterations=3, borderType=cv2.BORDER_CONSTANT)
    (thresh, th_gray_summed_mat) = cv2.threshold(closing_gray_summed_mat, 100, 255, cv2.THRESH_BINARY)  # 36
    revert_resize = resize_image(th_gray_summed_mat, 5)

    x, y = np.where(revert_resize > 0)
    colorized_mask = cv2.cvtColor(revert_resize, cv2.COLOR_GRAY2BGRA)
    colorized_mask[x, y] = (0, 255, 255, 255)   # YELLOW
    # show_partial_matrices(closing_gray_summed_mat, stretched_mat, th_gray_summed_mat, revert_resize, colorized_mask)

    return colorized_mask


def generate_masks(dataset_entry):
    my_events, target = dataset_entry
    print(target)

    denoise_transform = tonic.transforms.Denoise(filter_time=5000)
    events_denoised = denoise_transform(my_events)

    positive_event_array = generate_event_arrays(events_denoised, 1)
    negative_event_array = generate_event_arrays(events_denoised, 0)

    frames, cropped_frames, len_x, len_y, cropping_positions \
        = generate_fixed_num_events_frames(positive_event_array, negative_event_array)

    gray_summed_mat = create_gray_summed_mat(cropped_frames, len_y, len_x)
    stretched_mat = contrast_stretch(gray_summed_mat.copy(), len(cropped_frames))

    colorized_mask = generate_average_mask(stretched_mat)

    label_masked_frames = []
    for ind, frame in enumerate(frames):
        (y0, y1, x0, x1) = cropping_positions[ind]
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        positioned_colorized_mask = np.zeros(result.shape, np.uint8)
        positioned_colorized_mask[y0:y1, x0:x1] = colorized_mask
        result = cv2.addWeighted(result, 1, positioned_colorized_mask, 0.3, 0)
        label_masked_frames.append(result)

    show_events(label_masked_frames, 'frames/label_masked_frame' + str(target) + '_')


if __name__ == '__main__':
    dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    #                0,  1,    2,    3,    4,    5,    6,    7,    8,    9
    target_arrays = [0, 1000, 3000, 4000, 4900, 5500, 6500, 7000, 8500, 9200]
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)

    # for i in target_arrays:
    #     generate_masks(dataset[i])
    generate_masks(dataset[9200])

    frames_path = 'frames/'
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)

    file_paths = os.listdir(frames_path)
    images = []
    for image_path in file_paths:
        image = cv2.imread(frames_path + image_path, cv2.IMREAD_COLOR)
        images.append(image)
    generate_gif('./mask_applied.gif', images)

















