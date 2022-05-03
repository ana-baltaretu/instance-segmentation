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
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)

frames = frame_transform(events)

def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_frames(frames)