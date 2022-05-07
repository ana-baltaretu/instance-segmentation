# NEUROMORPHIC DATASETS WITH TONIC + SNNTORCH
# https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_7.html
# Events:
#   - 34x34 grid
#   - timestamp in microseconds
#   - polarity: +1 on spike, -1 off spike

# https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb#scrollTo=ace6cd0b-7b56-4422-b3bd-23bac65db9bd

import tonic
import tonic.transforms as transforms
import os
import matplotlib as plt

from torch.utils.data import DataLoader
from tonic import CachedDataset


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def basic_load():
    dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
    events, target = dataset[0]

    print(events)
    tonic.utils.plot_event_grid(events, axis_array=(3, 3))


def load_sample_simple(trainset):
    for i in range(100):
        events, target = trainset[i]
        # print(events)


def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def load_sample_cached(cached_dataloader):
    for i, (event_tensor, target) in enumerate(iter(cached_dataloader)):
        if i > 99: break
        if i == 0:
            print(event_tensor)
            print(event_tensor.shape)
            event = event_tensor
            print(event)


            # sensor_size = tonic.datasets.NMNIST.sensor_size
            # frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)
            # frames = frame_transform(event_tensor)
            # plot_frames(frames)
            # tonic.utils.plot_event_grid(events)



def cached_load(trainset):
    cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train')
    cached_dataloader = DataLoader(cached_trainset)
    load_sample_cached(cached_dataloader)


def denoise_load():
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])

    trainset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=False)

    # load_sample_simple(trainset)
    cached_load(trainset)

if __name__ == '__main__':
    basic_load()

    # denoise_load()

