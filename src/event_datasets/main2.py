import tonic
import matplotlib.pyplot as plt
import tonic.transforms as transforms
import torch
torch.manual_seed(1234)
import cv2


def to_voxel():
    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    events, target = dataset[1000]
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    noised_transform = tonic.transforms.UniformNoise(sensor_size=sensor_size, n_noise_events=events.shape[0] // 3)

    events_denoised = denoise_transform(events)
    events_noised = noised_transform(events_denoised)

    volume = transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=3)(events_noised)
    fig, axes = plt.subplots(1, len(volume))
    for axis, slice in zip(axes, volume):
        axis.imshow(slice)
        axis.axis("off")
    plt.tight_layout()
    plt.show()

def to_time_surfaces():
    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    events, target = dataset[1000]
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)

    events_denoised = denoise_transform(events)

    surfaces = transforms.ToTimesurface(sensor_size=sensor_size, surface_dimensions=None, tau=10000, decay='exp')(events_denoised)
    n_events = events_denoised.shape[0]
    n_events_per_slice = n_events // 3
    fig, axes = plt.subplots(1, 3)
    for i, axis in enumerate(axes):
        surf = surfaces[(i+1)*n_events_per_slice - 1]
        axis.imshow(surf[0] - surf[1])
        axis.axis("off")
    plt.tight_layout()
    plt.show()

def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1]-frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def save_frames(frames, path):
    for i, frame in enumerate(frames):
        frame = resize_image(frame, 1000)
        cv2.imwrite(path + str(i) + '.png', frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(200)
    cv2.destroyAllWindows()


def resize_image(img, percentage):
    """
    img = the image you want to resize
    percentage = how big you want it to be
    (ex: 20 for making the image 5 times smaller)
    (ex: 200 for making the image 2 times larger)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    w = int(img.shape[1] * percentage / 100)
    h = int(img.shape[0] * percentage / 100)
    dim = (w, h)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


nmnist_dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
ncaltech101_dataset = tonic.datasets.NCALTECH101(save_to='./data')


sensor_size = tonic.datasets.NCALTECH101.sensor_size
events_foreground, _ = nmnist_dataset[1000]
events_background = ncaltech101_dataset[10][0]

print(ncaltech101_dataset[500])

denoised_foreground = tonic.transforms.Denoise(filter_time=10000)(events_foreground)
denoised_background = tonic.transforms.Denoise(filter_time=10000)(events_background)
added_events = tonic.transforms.UnionDataset(new_events=denoised_foreground)(denoised_background)


frames = transforms.ToFrame(sensor_size=sensor_size, n_event_bins=3)(added_events)
image = transforms.ToImage(sensor_size)(added_events)

# td = ev.Events(added_events.size, 232, 170)
# td.data.x = added_events["x"]
# td.width = td.data.x.max() + 1
# td.data.y = added_events["y"]
# td.height = td.data.y.max() + 1
# td.data.ts = added_events["t"]
# td.data.p = added_events["p"]

# td.save_td()

# save_frames(frames, "./frames/output_")

# plot_frames(frames.squeeze())


# dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
# events_to_merge, target = dataset[1000]
# print(target)
# ncaltech101 = tonic.datasets.NCALTECH101(save_to='./data')
# ncal_to_merge = ncaltech101[0][0]


# sensor_size = tonic.datasets.NMNIST.sensor_size
# frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_event_bins=8)

# denoise_transform = tonic.transforms.Denoise(filter_time=10000)
# noised_transform = tonic.transforms.UniformNoise(sensor_size=sensor_size, n_noise_events=10000)
# add_transform = tonic.transforms.UnionDataset(new_events=denoise_transform(events_to_merge))

# transform = transforms.Compose([denoise_transform, add_transform, frame_transform])
# dataset = tonic.datasets.NCALTECH101(save_to='./data',
#                                 transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

# frames, target = next(iter(dataloader))

# plot_frames(frames.squeeze())
# tonic.utils.plot_event_grid(events_to_merge, axis_array=(2,2))

# ncaltech101 = tonic.datasets.NCALTECH101(save_to='./data')

# print (ncaltech101[0])
# print (dataset[0])

# to_time_surfaces()
# to_voxel()

import numpy as np


def groupby(X):
    X = np.asarray(X)
    x_uniques = np.unique(X)
    return {xi: X[X == xi] for xi in x_uniques}


a = np.array([[0, 0, 1, 1], [0, 0, 1, -1], [1, 1, 2, 0]])

print(groupby(a))
