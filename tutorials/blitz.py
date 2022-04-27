import torch
import torchvision
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

def tutorial1():
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        print(f"Device tensor is stored on: {tensor.device}")

    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(tensor)

    t1 = torch.cat([tensor, tensor, tensor], dim=0)
    t2 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1, t2)

    # This computes the element-wise product
    print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
    # Alternative syntax:
    print(f"tensor * tensor \n {tensor * tensor}")

    print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
    # Alternative syntax:
    print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

    # In-place operations Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)

    # Tensor to NumPy array
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    # A change in the tensor reflects in the NumPy array.
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")


def tutorial2_pytorch():
    model = torchvision.models.resnet18(pretrained=True)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)
    prediction = model(data)
    loss = (prediction - labels).sum()
    loss.backward()
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()  # gradient descent


def tutorial2_autograd():
    import torch

    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)
    Q = 3 * a ** 3 - b ** 2

    external_grad = torch.tensor([1., 1.])
    Q.backward(gradient=external_grad)

    # check if collected gradients are correct
    print(9 * a ** 2 == a.grad)
    print(-2 * b == b.grad)


def tutorial2_dag():
    x = torch.rand(5, 5)
    y = torch.rand(5, 5)
    z = torch.rand((5, 5), requires_grad=True)

    a = x + y
    print(f"Does `a` require gradients? : {a.requires_grad}")
    b = x + z
    print(f"Does `b` require gradients?: {b.requires_grad}")

    model = torchvision.models.resnet18(pretrained=True)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, 10)

    # Optimize only the classifier
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


####### TUTORIAL 3


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def tutorial3_nn():
    # NN TRAINING
    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    # torch.nn only supports mini-batches.
    # The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
    # If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

    # LOSS

    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    # BACKPROP
    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    # WEIGHTS UPDATE

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update


# def tutorial3_loss(net):

