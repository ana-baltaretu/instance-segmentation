# https://tonic.readthedocs.io/en/latest/tutorials/davis_data.html

import tonic
import numpy as np
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset = tonic.datasets.DAVISDATA(save_to='../data', recording='shapes_6dof')

data, targets = dataset[0]
events, imu, images = data

print(events["t"])

print(images["ts"])

mean_diff = np.diff(list(zip(images["ts"], images["ts"][1:]))).mean()
print(f"Average difference in image timestamps in microseconds: {mean_diff}")

sensor_size = tonic.datasets.DAVISDATA.sensor_size
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=mean_diff)

image_center_crop = torchvision.transforms.Compose([torch.tensor,
                                                    torchvision.transforms.CenterCrop((100,100))])

def data_transform(data):
    # first we have to unpack our data
    events, imu, images = data
    # we bin events to event frames
    frames = frame_transform(events)
    # then we can apply frame transforms to both event frames and images at the same time
    frames_cropped = image_center_crop(frames)
    images_cropped = image_center_crop(images["frames"])
    return frames_cropped, imu, images_cropped


dataset = tonic.datasets.DAVISDATA(save_to='../data',
                                   recording='slider_depth',
                                   transform=data_transform)

data, targets = dataset[0]
frames_cropped, imu, images_cropped = data


fig, (ax1, ax2) = plt.subplots(1,2)
event_frame = frames_cropped[10]
ax1.imshow(event_frame[0]-event_frame[1])
ax1.set_title("event frame")
ax2.imshow(images_cropped[10], cmap=mpl.cm.gray)
ax2.set_title("grey level image")

plt.show()
