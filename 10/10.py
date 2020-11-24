import torch
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])

data_path = './MNIST'
data_train = datasets.MNIST(root = data_path, train = False, download = True, transform = transform)
data_test = datasets.MNIST(root = data_path, train = True, download = True, transform = transform)

class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.Softsign(),
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.Softsign(),
        )
        
        self.classifier3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=10),
            nn.LogSoftmax(dim=1),
        )
        
    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        out = self.classifier3(x)
        return out

def train(model, data_train, data_train_batch, optimizer, criterion, device='cuda'):
    model.train()
    n_batch = 0
    avg_loss = 0
    avg_acc = 0
    for batch_idx, (x, y) in enumerate(data_train_batch):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model.forward(x)
        loss = criterion(pred, y)
        avg_loss += loss.item()
        avg_acc += accuracy(pred, y)
        n_batch += 1

        loss.backward()
        optimizer.step()

    avg_loss /= n_batch
    avg_acc /= n_batch
    return avg_loss, avg_acc
