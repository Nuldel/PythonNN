from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, inputLayer, actFunction, dropOut):
        super(AutoEncoder, self).__init__()
        self.inputLayer = inputLayer
        self.dummyLayer = nn.Linear(inputLayer.out_features, inputLayer.in_features)
        self.dropOut = nn.Dropout(p=dropOut)
        self.actFunction = actFunction

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.actFunction(x)
        x = self.dropOut(x)
        x = self.dummyLayer(x)
        x = self.actFunction(x)
        return x


def train(network, optimizer, train_loader, epoch, log_interval):
    network.train()
    MSELoss = nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = MSELoss(output, data)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
