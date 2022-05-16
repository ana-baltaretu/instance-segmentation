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
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from tonic import CachedDataset

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn

from src.visualize import plot_frames

from IPython.display import HTML


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


def load_sample_batched(trainset):
    cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train')
    batch_size = 100
    batched_loader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    events, target = next(iter(batched_loader))


# this time, we won't return membrane as we don't need it

def forward_pass(net, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)


if __name__ == '__main__':
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])

    trainset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=False)

    # basic_load()
    # cached_load(trainset)

    transform = tonic.transforms.Compose([torch.from_numpy,
                                          torchvision.transforms.RandomRotation([-10, 10])])

    cached_trainset = CachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

    # no augmentations for the testset
    cached_testset = CachedDataset(testset, cache_path='./cache/nmnist/test')

    batch_size = 128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(),
                             shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

    event_tensor, target = next(iter(trainloader))
    print(event_tensor.shape)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # neuron and simulation parameters
    spike_grad = surrogate.fast_sigmoid(slope=75)
    beta = 0.5

    #  Initialize Network
    net = nn.Sequential(nn.Conv2d(2, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 32, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(32 * 5 * 5, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    num_epochs = 1
    num_iters = 50

    loss_hist = []
    acc_hist = []

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

            # training loop breaks after 50 iterations
            if i == num_iters:
                break



    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.title("Train Set Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()

    spk_rec = forward_pass(net, data)



    idx = 0

    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(f"The target label is: {targets[idx]}")

    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

    #  Plot spike count histogram
    anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
                            animate=True, interpolate=1)

    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")





