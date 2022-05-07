# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html
# https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#
# https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.delta
# https://raw.githubusercontent.com/jeshraghian/snntorch/master/docs/_static/img/examples/tutorial1/1_2_3_spikeconv.png

import snntorch as snn
from snntorch import utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
from cv2 import cv2
######### SETUP #########

# Training Parameters
batch_size = 128
data_path = '../data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

######### SETUP DATASET #########

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)

print(f"The size of mnist_train is {len(mnist_train)}")

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

######### SPIKE ENCODING #########

num_steps = 100
# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)
print(spike_data.size())





