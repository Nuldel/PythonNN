import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
import autoEncoder as ae
import matplotlib.pyplot as plt

mnist_data = datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# plt.imshow(mnist_data[0][0].numpy().squeeze())
# plt.imshow(mnist_test[0][0].numpy().squeeze())


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class Recons(nn.Module):
    def forward(self, inp):
        size = int(np.sqrt(inp.size(1)))
        return inp.view(inp.size(0), 1, size, size)


# Precondiciones:
# len(sizes) >= 2; len(actFunctions) == len(sizes) - 1
class GeneralNN(nn.Module):
    def __init__(self, sizes, actFunctions, dropOut=0.0):
        super(GeneralNN, self).__init__()
        self.layers = []
        for s in range(len(sizes) - 1):
            newLayer = nn.Linear(sizes[s], sizes[s+1])
            setattr(self, 'layer' + str(s), newLayer)
            self.layers.append(newLayer)
        self.dropOut = dropOut
        self.dropOutLayer = nn.Dropout(p=dropOut)
        self.actFunctions = actFunctions
        self.toTensor = torchvision.transforms.ToTensor()

    def autoEncode(self, learningRates, actFunction, n_epochs):
        for lay in range(len(self.layers)):
            autoEncoder = ae.AutoEncoder(self.layers[lay], actFunction, dropOut=self.dropOut)
            optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=learningRates[lay])

            data = [np.random.randint(0, 255, (1, self.layers[lay].in_features, 1), np.uint8) for i in range(50000)]
            train_data = [self.toTensor(d) for d in data]
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)

            print('TRAINING AUTOENCODER {:.0f}\n\n'.format(lay+1))
            for epoch in range(1, n_epochs+1):
                ae.train(autoEncoder, optimizer, train_loader, epoch, 10)
                print('\n')

    def forward(self, x):
        for lay in range(len(self.layers)):
            x = self.layers[lay](x)
            x = self.actFunctions[lay](x)
            if lay < len(self.layers)-1:
                x = self.dropOutLayer(x)
        return x


random_seed = 1
torch.manual_seed(random_seed)


def trainImages(network, optimizer, train_loader, epoch, log_interval):
    network.train()
    crossEntropy = nn.CrossEntropyLoss()
    flatten = Flatten()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        vectorData = flatten(data)
        output = network(vectorData)
        loss = crossEntropy(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
    return train_losses


def testImages(network, test_loader):
    crossEntropy = nn.CrossEntropyLoss()
    network.eval()
    flatten = Flatten()
    test_losses = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            vectorData = flatten(data)
            output = network(vectorData)
            loss = crossEntropy(output, target)
            test_losses.append(loss.item())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print('\nTest set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(np.mean(test_losses), correct,
                                                                              len(test_loader.dataset),
                                                                              100. * correct / len(test_loader.dataset)))
    return test_losses


def buildAndTrainImages(n_epochs=5, batch_size_train=64, batch_size_test=1000, learning_rate=0.01, log_interval=10,
                        encoder_epochs=0):
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size_test)

    network = GeneralNN([784, 100, 20, 10], actFunctions=[nn.ReLU(), nn.ReLU(), nn.ReLU()], dropOut=0.0)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = [np.mean(testImages(network, test_loader))]
    if encoder_epochs > 0:
        network.autoEncode(learningRates=[0.001, 0.001, 0.0001], actFunction=nn.ReLU(), n_epochs=encoder_epochs)
        testImages(network, test_loader)

    for epoch in range(1, n_epochs+1):
        train_losses.append(np.mean(trainImages(network, optimizer, train_loader, epoch, log_interval)))
        test_losses.append(np.mean(testImages(network, test_loader)))

    return train_losses, test_losses


def compareMethods(n_epochs=10, encoder_epochs=10):
    losses2 = buildAndTrainImages(n_epochs=n_epochs, encoder_epochs=encoder_epochs)[1]
    losses1 = buildAndTrainImages(n_epochs=n_epochs)[1]

    plt.title("Comparativa del uso de autoencoder")
    plt.xlabel("Épocas")
    plt.ylabel("Error/costo de testeo")
    plt.plot(range(n_epochs+1), losses1, label="Entrenamiento sin autoencoder")
    plt.plot(range(n_epochs + 1), losses2, label="Entrenamiento con autoencoder")
    plt.legend()
    plt.show()
