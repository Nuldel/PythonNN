import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn

mnist_data = datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

random_seed = 1
torch.manual_seed(random_seed)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 784)

class Recons(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 1, 28, 28)


class net(nn.Module):
    def __init__(self, hiddenLayer):
        super(net, self).__init__()
        self.linear1 = nn.Linear(784, hiddenLayer)
        self.linear2 = nn.Linear(hiddenLayer, 784)
        self.dropOut1 = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = Flatten()
        self.recons = Recons()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.dropOut1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.recons(x)
        return x


def train(network, optimizer, train_loader, epoch, log_interval):
    network.train()
    MSELoss = nn.MSELoss()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = MSELoss(output, data)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
    return train_losses

def test(network, test_loader):
    MSELoss = nn.MSELoss()
    network.eval()
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            loss = MSELoss(output, data)
            test_losses.append(loss.item())
    print('\nTest set: Avg. loss: {:.6f}\n'.format(np.mean(test_losses)))
    return test_losses


def buildAndTest(hidden_layer=64, n_epochs=3, batch_size_train=1000, batch_size_test=1000,
                 learning_rate=0.1, momentum=0.5, decay=0, log_interval=5):
    train_loader = torch.utils.data.DataLoader(mnist_data,
                                               batch_size=batch_size_train,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test,
                                              batch_size=batch_size_test)

    network = net(hidden_layer)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=decay,
                                nesterov=True)

    train_losses = []
    test_losses = []
    test_losses.append(np.mean(test(network, test_loader)))
    for epoch in range(1, n_epochs + 1):
        train_losses.append(np.mean(train(network, optimizer, train_loader, epoch, log_interval)))
        test_losses.append(np.mean(test(network, test_loader)))

    return (train_losses, test_losses)

def learnComparison(n_epochs):
  plt.title('Aprendizaje en función de las épocas')
  plt.xlabel('Épocas')
  plt.ylabel('Promedio de pérdida')
  for i in np.linspace(0.05, 0.95, 5):
    banner = '****************************************'
    print((banner + '\nLEARNING RATE: {:.2f} - MOMENTUM: {:.2f}\n' + banner).format(i, i))
    losses = buildAndTest(hidden_layer=64, n_epochs=n_epochs, learning_rate=i, momentum=i, log_interval=10)[1]
    plt.plot(range(n_epochs+1), losses, label='LR = MOM = {:.2f}'.format(i))
  plt.legend()
  plt.show()

def layerSizeComparison(n_epochs):
  losses = []
  for h in [64, 128, 256, 512]:
    banner = '****************************************'
    print((banner + '\nHIDDEN LAYER SIZE: {:.0f}\n' + banner).format(h))
    losses.append(buildAndTest(hidden_layer=h, n_epochs=n_epochs, learning_rate=0.95, momentum=0.95, log_interval=10))
  return losses