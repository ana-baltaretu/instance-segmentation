# https://tonic.readthedocs.io/en/latest/tutorials/nmnist.html

# https://github.com/neuromorphs/tonic
# https://github.com/tihbe/python-ebdataset
# https://github.com/TimoStoff/event_utils
import numpy as np
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset = tonic.datasets.NMNIST(save_to='../data', train=False)
events, target = dataset[1000]

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)

frames = frame_transform(events)

denoise_transform = tonic.transforms.Denoise(filter_time=10000)
events_denoised = denoise_transform(events)

surfaces = transforms.ToTimesurface(sensor_size=sensor_size, surface_dimensions=None, tau=10000, decay='exp')(events_denoised)


def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_frames_denoised():
    frames_denoised = frame_transform(events_denoised)

    plot_frames(frames_denoised)


def plot_voxel_grid():
    volume = transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=3)(events_denoised)

    fig, axes = plt.subplots(1, len(volume))
    for axis, slice in zip(axes, volume):
        axis.imshow(slice)
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_time_surfaces():
    n_events = events_denoised.shape[0]
    n_events_per_slice = n_events // 3
    fig, axes = plt.subplots(1, 3)
    for i, axis in enumerate(axes):
        surf = surfaces[(i + 1) * n_events_per_slice - 1]
        axis.imshow(surf[0] - surf[1])
        axis.axis("off")
    plt.tight_layout()

    plt.show()


def plot_1_channel_3D(x_data, y_data, z_data, cmap, save_path):
    ax = plt.axes(projection='3d')

    ax.scatter3D(z_data, x_data, y_data, c=z_data, cmap=cmap)
    for ii in range(0, 360, 10):
        ax.view_init(elev=0, azim=ii)
        plt.savefig(save_path + "%d.png" % ii)
    plt.show()


if __name__ == '__main__':
    # plot_frames(frames)

    plot_frames_denoised()

    # plot_voxel_grid()
    # print(events)

    x_data_pos = np.array([])   # X-axis
    y_data_pos = np.array([])   # Y-axis
    z_data_pos = np.array([])   # Time

    x_data_neg = np.array([])   # X-axis
    y_data_neg = np.array([])   # Y-axis
    z_data_neg = np.array([])   # Time

    time_pos = 0
    time_neg = 0

    div_rate = 100
    mult_rate = 100

    for (x, y, time, p) in events:
        if p == 0:
            x_data_neg = np.insert(x_data_neg, len(x_data_neg), x)
            y_data_neg = np.insert(y_data_neg, len(y_data_neg), y)
            z_data_neg = np.insert(z_data_neg, len(z_data_neg), int(time_neg/div_rate) * mult_rate)
            time_neg += 1

        else:
            x_data_pos = np.insert(x_data_pos, len(x_data_pos), x)
            y_data_pos = np.insert(y_data_pos, len(y_data_pos), y)
            z_data_pos = np.insert(z_data_pos, len(z_data_pos), int(time_pos/div_rate) * mult_rate)
            time_pos += 1

    plot_1_channel_3D(x_data_pos, y_data_pos, z_data_pos, "Blues", "plots/pos")
    plot_1_channel_3D(x_data_neg, y_data_neg, z_data_neg, "Reds", "plots/neg")
    # plot_time_surfaces()
