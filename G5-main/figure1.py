import pickle
import copy
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src import datasets, perturbed_dataloader, training

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

ds_train = datasets.load_MNIST(True)
ds_test = datasets.load_MNIST(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 200
lr = 0.02
alphas = [0.8,0.9,1]
num_data = 50000
batch_size = 100
indices_test = [i for i in range(len(ds_test))]
random.shuffle(indices_test)

stats = {}
for alpha in alphas:
    print("alpha", alpha)
    dataset_train = perturbed_dataloader.PerturbedDataset(ds_train, alpha, size = num_data,enforce_false = False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = torch.utils.data.Subset(ds_test, indices_test[:10000])
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=300, shuffle=False)

    model = LeNet()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    loss = torch.nn.CrossEntropyLoss()

    s = training.train(model, optimizer, loss, dataloader_train, dataloader_test, num_epochs, device=device)
    stats[alpha] = s
    
    with open('./data/figure1_stats.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)




