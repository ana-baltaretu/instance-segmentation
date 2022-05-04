# https://tonic.readthedocs.io/en/latest/tutorials/nmnist.html

# https://github.com/neuromorphs/tonic
# https://github.com/tihbe/python-ebdataset
# https://github.com/TimoStoff/event_utils

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

if __name__ == '__main__':
    # plot_frames(frames)

    # plot_frames_denoised()

    # plot_voxel_grid()

    plot_time_surfaces()
