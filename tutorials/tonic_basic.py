# https://tonic.readthedocs.io/en/latest/tutorials/nmnist.html

# https://github.com/neuromorphs/tonic
# https://github.com/tihbe/python-ebdataset
# https://github.com/TimoStoff/event_utils
import numpy as np
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import cv2

from src.visualize import plot_1_channel_3D, generate_event_arrays, plot_frames_denoised, plot_2_channel_3D, generate_event_frames, show_events
from src.canny import apply_canny_part

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset = tonic.datasets.NMNIST(save_to='../data', train=False)
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





if __name__ == '__main__':
    # plot_frames(frames)

    # plot_frames_denoised(frame_transform, my_events)
    positive_event_array = generate_event_arrays(my_events, 1, 300, 100)
    negative_event_array = generate_event_arrays(my_events, 0, 300, 100)

    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    frames = generate_event_frames(positive_event_array, negative_event_array)
    show_events(frames)

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



