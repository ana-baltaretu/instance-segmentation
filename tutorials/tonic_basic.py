# https://tonic.readthedocs.io/en/latest/tutorials/nmnist.html

# https://github.com/neuromorphs/tonic
# https://github.com/tihbe/python-ebdataset
# https://github.com/TimoStoff/event_utils
import math

import numpy as np
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import cv2

from src.visualize import plot_1_channel_3D, generate_event_arrays, plot_frames_denoised, plot_2_channel_3D, \
    generate_event_frames, show_events, generate_fixed_num_events_frames, resize_image
from src.canny import apply_canny_part

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

#                0,  1,    2,    3,    4,    5,    6,    7,    8,    9
target_arrays = [0, 1000, 3000, 4000, 4900, 5500, 6500, 7000, 8500, 9000]
my_events, target = dataset[0]

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)

frames = frame_transform(my_events)


def plot_voxel_grid(events):
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    events_denoised = denoise_transform(events)

    volume = transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=3)(events_denoised)

    fig, axes = plt.subplots(1, len(volume))
    for axis, slice in zip(axes, volume):
        axis.imshow(slice)
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_time_surfaces(events):
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    events_denoised = denoise_transform(events)

    surfaces = transforms.ToTimesurface(sensor_size=sensor_size, surface_dimensions=None, tau=10000, decay='exp')(events_denoised)
    n_events = events_denoised.shape[0]
    n_events_per_slice = n_events // 3
    fig, axes = plt.subplots(1, 3)
    for i, axis in enumerate(axes):
        surf = surfaces[(i + 1) * n_events_per_slice - 1]
        axis.imshow(surf[0] - surf[1])
        axis.axis("off")
    plt.tight_layout()

    plt.show()

def make_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_pair = (kernel_size, kernel_size)
    return kernel, kernel_pair

if __name__ == '__main__':
    print(target)
    # plot_frames(frames)

    # plot_frames_denoised(frame_transform, my_events)

    denoise_transform = tonic.transforms.Denoise(filter_time=5000)
    events_denoised = denoise_transform(my_events)

    positive_event_array = generate_event_arrays(events_denoised, 1, 300, 100)
    negative_event_array = generate_event_arrays(events_denoised, 0, 300, 100)

    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    # frames = generate_event_frames(positive_event_array, negative_event_array)
    # show_events(frames, 'frames/fixed_time')

    frames, cropped_frames, len_x, len_y = generate_fixed_num_events_frames(positive_event_array, negative_event_array)
    # show_events(frames, 'frames/fixed_events')
    # show_events(cropped_frames, 'frames/cropped_frames')

    kernel_size = 15

    summed_mat = np.zeros((len_y + 5, len_x + 5, 3), np.uint8)
    gray_summed_mat = cv2.cvtColor(summed_mat, cv2.COLOR_BGR2GRAY)

    for cropped in cropped_frames:
        gray_scaled = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        x_arr, y_arr = np.where(gray_scaled > 0)
        print(x_arr, y_arr)
        for ind, x in enumerate(x_arr):
            y = y_arr[ind]
            gray_summed_mat[x][y] += 1

    W, H = gray_summed_mat.shape
    for x in range(H):
        for y in range(W):
            gray_summed_mat[y][x] = math.ceil(gray_summed_mat[y][x] / len(cropped_frames) * 255)

        # print(max(gray_summed_mat))

    resized = resize_image(gray_summed_mat, 2000)
    closing_gray_summed_mat = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, make_kernel(11)[0], iterations=3, borderType=cv2.BORDER_CONSTANT)
    (thresh, th_gray_summed_mat) = cv2.threshold(closing_gray_summed_mat, 100, 255, cv2.THRESH_BINARY) # 36

    cv2.imshow("gray_summed_mat", gray_summed_mat)
    cv2.imshow("th_gray_summed_mat", th_gray_summed_mat)
    cv2.imshow("closing_gray_summed_mat", closing_gray_summed_mat)

    edged = cv2.Canny(th_gray_summed_mat, 30, 200)
    cv2.imshow("edged", edged)
    cv2.waitKey(0)

    dilated_frames = []
    for cropped in cropped_frames:
        resized = resize_image(cropped, 2000)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        (thresh, black_and_white) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        cv2.imshow("black_and_white", black_and_white)
        closing = cv2.morphologyEx(black_and_white, cv2.MORPH_CLOSE, make_kernel(17)[0], iterations=2)

        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, make_kernel(11)[0], iterations=5)
        # eroded_image = cv2.dilate(gray, kernel, iterations=1)
        # dilated_image = cv2.dilate(closing, kernel, iterations=3)

        # cv2.imshow("closing", closing)
        # cv2.waitKey(200)


        # (thresh, black_and_white) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


        edged = cv2.Canny(closing, 30, 200)

        cv2.imshow("edged", edged)
        cv2.waitKey(200)

        contours, hierarchy = cv2.findContours(edged,
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(resized, contours, -1, (0, 255, 0), 3)
        resized = resize_image(resized, 25)

        # dil = cv2.blur(resized, kernel_pair)
        # # dil = cv2.dilate(dil, kernel)
        # # dil = cv2.dilate(dil, kernel)
        # # dil = cv2.dilate(dil, kernel)
        # (thresh, ffff) = cv2.threshold(dil, 100, 255, cv2.THRESH_BINARY)
        # ffff = resize_image(ffff, 10)
        dilated_frames.append(resized)
        # cv2.imshow("dilated", dil)
        # cv2.waitKey(200)


    show_events(dilated_frames, 'frames/dilated_frames')


    # mx_time_stamp = max(max(time_data_pos), max(time_data_neg))
    # mn_time_stamp = min(min(time_data_pos), min(time_data_neg))
    # time_diff = mx_time_stamp - mn_time_stamp
    #
    # print(mn_time_stamp, mx_time_stamp)
    # print(time_diff)
    #
    # print(time_diff / 4)

    #
    # plot_1_channel_3D(x_data_pos, y_data_pos, z_data_pos, "Blues", "plots/pos")
    # plot_1_channel_3D(x_data_neg, y_data_neg, z_data_neg, "Reds", "plots/neg")

    # x_data_pos, y_data_pos, z_data_pos, time_data_pos = generate_event_arrays(my_events[3*int(len(my_events)/4):], 1)
    # x_data_neg, y_data_neg, z_data_neg, time_data_neg = generate_event_arrays(my_events[3*int(len(my_events)/4):], 0)

    # plot_2_channel_3D(x_data_pos, y_data_pos, z_data_pos, "Blues", x_data_neg, y_data_neg, z_data_neg, "Reds", "plots/comb")



    # img = apply_canny_part(img)
    # plot_voxel_grid(my_events)
    # plot_time_surfaces(my_events)



