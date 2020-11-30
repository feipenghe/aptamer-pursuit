import os, sys
import numpy as np
import json
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA

class ConvTwoHead(nn.Module):
    def __init__(self):
        super(ConvTwoHead, self).__init__()
        self.name = "ConvTwoHead"
        self.single_alphabet=False
        
        self.cnn_apt_1 = nn.Conv1d(4, 100, 3, padding=2) 
        self.cnn_apt_2 = nn.Conv1d(50, 150, 3, padding=2) 

        
        # There are 20 channels
        self.cnn_pep_1 = nn.Conv1d(20, 100, 3, padding=2)
        self.cnn_pep_2 = nn.Conv1d(75, 150, 3, padding=2)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2) 
        
        self.cnn_apt = nn.Sequential(self.cnn_apt_1, self.maxpool, self.relu, 
                                     )
        self.cnn_pep = nn.Sequential(self.cnn_pep_1, self.maxpool, self.relu,
                                     )
        
        self.fc1 = nn.Linear(2600, 1300)
        self.fc2 = nn.Linear(1300, 1)
    
    def forward(self, apt, pep):
        apt = apt.view(512, 4, 10).long()
        self.cnn_apt_1(apt.double())
        apt = self.cnn_apt(apt.long())
        pep = self.cnn_pep(pep)
        
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Expects peptides to be encoding according to BLOSUM62 matrix
# Expects aptamers to be one hot encoded
class BlosumNet(nn.Module):
    def __init__(self):
        super(BlosumNet, self).__init__()
        self.name = "BlosumNet"
        self.single_alphabet = False
        
        self.cnn_apt_1 = nn.Conv1d(4, 25, 3, padding=2) 
        self.cnn_apt_2 = nn.Conv1d(25, 50, 3, padding=2) 
        self.cnn_apt_3 = nn.Conv1d(50, 25, 3, padding=2) 
        self.cnn_apt_4 = nn.Conv1d(25, 10, 3) 
        
        # There are 20 channels
        self.cnn_pep_1 = nn.Conv1d(20, 40, 3, padding=2)
        self.cnn_pep_2 = nn.Conv1d(40, 80, 3, padding=2)
        self.cnn_pep_3 = nn.Conv1d(80, 150, 3, padding=2)
        self.cnn_pep_4 = nn.Conv1d(150, 50, 3, padding=2)
        self.cnn_pep_5 = nn.Conv1d(50, 10, 3, padding=2)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2) 
        
        self.cnn_apt = nn.Sequential(self.cnn_apt_1, self.maxpool, self.relu, 
                                     self.cnn_apt_2, self.maxpool, self.relu)
        self.cnn_pep = nn.Sequential(self.cnn_pep_1, self.maxpool, self.relu,
                                     self.cnn_pep_2, self.maxpool, self.relu)
        
        self.fc1 = nn.Linear(790, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 1)
    
    def forward(self, apt, pep):
        apt = self.cnn_apt(apt)
        pep = self.cnn_pep(pep)
        
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# Expects peptides to be encoding according to BLOSUM62 matrix
# Expects aptamers to be one hot encoded
class BlosumConvNet(nn.Module):
    def __init__(self):
        super(BlosumConvNet, self).__init__()
        self.name = "BlosumConvNet"
        self.single_alphabet = False
        
        self.cnn_apt_1 = nn.Conv1d(4, 25, 3, padding=2) 
        self.cnn_apt_2 = nn.Conv1d(25, 100, 3, padding=2) 
        self.cnn_apt_3 = nn.Conv1d(100, 200, 3, padding=2) 
        self.cnn_apt_4 = nn.Conv1d(200, 300, 3) 
        
        # There are 20 channels
        self.cnn_pep_1 = nn.Conv1d(20, 40, 3, padding=2)
        self.cnn_pep_2 = nn.Conv1d(40, 100, 3, padding=2)
        self.cnn_pep_3 = nn.Conv1d(100, 200, 3, padding=2)
        self.cnn_pep_4 = nn.Conv1d(200, 300, 3, padding=2)
        self.cnn_pep_5 = nn.Conv1d(300, 350, 3, padding=2)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2) 
        
        self.cnn_apt = nn.Sequential(self.cnn_apt_1, self.maxpool, self.relu, 
                                     self.cnn_apt_2, self.maxpool, self.relu)
        self.cnn_pep = nn.Sequential(self.cnn_pep_1, self.maxpool, self.relu,
                                     self.cnn_pep_2, self.maxpool, self.relu)
        
        self.fc1 = nn.Linear(1400, 700 )
        self.fc2 = nn.Linear(700, 250)
        self.fc3 = nn.Linear(250, 1)
    
    def forward(self, apt, pep):
        apt = self.cnn_apt(apt)
        pep = self.cnn_pep(pep)
        
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# Expects peptides to be encoding according to BLOSUM62 matrix
# Expects aptamers to be one hot encoded
class BlosumLinearNet(nn.Module):
    def __init__(self):
        super(BlosumLinearNet, self).__init__()
        self.name = "BlosumLinearNet"
        self.single_alphabet = False
        
        self.fc_apt_1 = nn.Linear(160, 200) 
        self.fc_apt_2 = nn.Linear(200, 250)
        self.fc_apt_3 = nn.Linear(250, 300)
        
        self.fc_pep_1 = nn.Linear(160, 200)
        self.fc_pep_2 = nn.Linear(200, 250)
        
        self.relu = nn.ReLU()
        
        self.fc_apt = nn.Sequential(self.fc_apt_1, self.fc_apt_2, self.fc_apt_3)
        self.fc_pep = nn.Sequential(self.fc_pep_1, self.fc_pep_2)
        
        self.fc1 = nn.Linear(550, 600)
        self.fc2 = nn.Linear(600, 1)
        
    def forward(self, apt, pep):
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        
        apt = self.fc_apt(apt)
        pep = self.fc_pep(pep)
        x = torch.cat((apt, pep), 1)
        x = self.fc2(self.fc1(x))
        x = torch.sigmoid(x)
        return x

class LinearBaseline(nn.Module):
    def __init__(self):
        super(LinearBaseline, self).__init__()
        self.name = "LinearBaseline"
        self.single_alphabet = False
        
        self.fc_1 = nn.Linear(320, 1)
    
    def forward(self, apt, pep):
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        
        x = torch.cat((apt, pep), 1)
        x = self.fc_1(x)
        x = torch.sigmoid(x)
        return x

class ConvBaseline(nn.Module):
    def __init__(self):
        super(ConvBaseline, self).__init__()
        self.name = "ConvBaseline"
        self.single_alphabet = True
        
        self.cnn_1 = nn.Conv1d(24, 100, 3)
        self.fc_1 = nn.Linear(4600, 1)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2) 
        
    
    def forward(self, pair):
        x = self.cnn_1(pair)
        x = x.view(-1, 1).T
        x = self.fc_1(x)
        x = self.relu(self.maxpool(x))
        x = torch.sigmoid(x)
        return x

class LinearTwoHead(nn.Module):
    def __init__(self):
        super(LinearTwoHead, self).__init__()
        self.name = "LinearTwoHead"
        self.single_alphabet=False
        
        self.fc_apt_1 = nn.Linear(160, 200) 
        self.fc_apt_2 = nn.Linear(200, 150)
        self.fc_apt_3 = nn.Linear(150, 100)
        
        self.fc_pep_1 = nn.Linear(160, 250)
        self.fc_pep_2 = nn.Linear(250, 100)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2) 

        
        
        self.fc_apt = nn.Sequential(self.fc_apt_1, self.relu, self.fc_apt_2, self.relu,  self.fc_apt_3)
        self.fc_pep = nn.Sequential(self.fc_pep_1, self.relu,  self.fc_pep_2)
        
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, apt, pep):
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        
        apt = self.fc_apt(apt)
        pep = self.fc_pep(pep)
        x = torch.cat((apt, pep), 1)
        x = self.fc2(self.fc1(x))
        x = torch.sigmoid(x)
        return x       

def loss(prediction, label, reduction = 'mean'):
    loss_val = F.binary_cross_entropy(prediction.squeeze(), label.squeeze(), reduction = reduction)
    return loss_val

# return number of correct (not percentage)
def num_correct(label, pred):
    #assert label.ndim == 1 and label.size() == pred.size()
    pred = pred.squeeze() > 0.5
    return (label.squeeze() == pred).sum().item()

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (apt, pep, label) in enumerate(train_loader):
        apt, pep, label = apt.to(device), pep.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(apt, pep)
        loss = loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)

def test(model, device, test_loader, log_interval=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (apt, pep, label) in enumerate(test_loader):
            apt, pep, label = apt.to(device), pep.to(device), label.to(device)
            output = model(apt, pep)
            test_loss_on = loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            correct += num_correct(output, label)
            if log_interval is not None and batch_idx % log_interval == 0:
                print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy