#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


# ## Preliminary

# In[3]:


na_list = ['A', 'C', 'G', 'T'] #nucleic acids
aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W', 'Y'] #amino acids
NNK_freq = [0.09375]*3 + [0.0625]*5 + [0.03125]*13 #freq of 21 NNK codons including the stop codon
sum_20 = 0.0625*5 + 0.09375*3 + 0.03125*12 #sum of freq without the stop codon
pvals = [0.09375/sum_20]*3 + [0.0625/sum_20]*5 + [0.03125/sum_20]*12 #normalize freq for 20 codons
pvals = [0.09375/sum_20]*3 + [0.0625/sum_20]*5 + [0.03125/sum_20]*11 +         [1- sum([0.09375/sum_20]*3 + [0.0625/sum_20]*5 + [0.03125/sum_20]*11)] 
        #adjust sum to 1 due to numerical issue
aa_dict = dict(zip(aa_list, pvals))


# ## Dataset

# In[4]:


aptamer_dataset_file = "../data/aptamer_dataset.json"

def construct_dataset():
    with open(aptamer_dataset_file, 'r') as f:
        aptamer_data = json.load(f)
    full_dataset = []
    aptamers = []
    peptides = []
    for aptamer in aptamer_data:
        peptides = aptamer_data[aptamer]
        if aptamer == "CTTTGTAATTGGTTCTGAGTTCCGTTGTGGGAGGAACATG": #took out aptamer control
            continue
        for peptide, _ in peptides:
            peptide = peptide.replace("_", "") #removed stop codons
            if "RRRRRR" in peptide: #took out peptide control
                continue
            if len(aptamer) == 40 and len(peptide) == 8: #making sure right length
                full_dataset.append((aptamer, peptide))
    full_dataset = list(set(full_dataset)) #removed duplicates
    for pair in full_dataset:
        aptamers.append(pair[0])
        peptides.append(pair[1])
    return full_dataset, aptamers, peptides 


# In[5]:


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, training_set):
        super(TrainDataset, self).__init__() 
        self.training_set = training_set
        
    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, idx):
        aptamer, peptide = self.training_set[idx]
        return aptamer, peptide
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_set):
        super(TestDataset, self).__init__() 
        self.test_set = test_set
        
    def __len__(self):
        return len(self.test_set)

    def __getitem__(self, idx):
        aptamer, peptide = self.test_set[idx]
        return aptamer, peptide


# In[6]:


full_dataset, aptamers, peptides = construct_dataset()
n = len(full_dataset)
training_set = full_dataset[:int(0.8*n)]
test_set = full_dataset[int(0.8*n):]
train_dataset = TrainDataset(training_set)
test_dataset = TestDataset(test_set)
train_loader = torch.utils.data.DataLoader(train_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset)


# ## One-hot encoding

# In[7]:


## Takes a peptide and aptamer sequence and converts to one-hot matrix
def one_hot(sequence_list, seq_type='peptide'):
    if seq_type == 'peptide':
        letters = aa_list
    else:
        letters = na_list
    
    one_hot = np.zeros((len(sequence_list), len(sequence_list[0]), len(letters)))
    
    for j in range(len(sequence_list)):
        sequence = sequence_list[j]
        for i in range(len(sequence)):
            element = sequence[i]
            idx = letters.index(element)
            one_hot[j][i][idx] = 1
    return one_hot


# ## NN Models

# In[8]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cnn_apt_1 = nn.Conv2d(40, 20, 1)
        self.cnn_apt_2 = nn.Conv2d(20, 10, 1)
        self.cnn_apt_3 = nn.Conv2d(10, 1, 1)
        self.fc_apt_1 = nn.Linear(160, 1)
        
        self.cnn_pep_1 = nn.Conv2d(8, 4, 1)
        self.cnn_pep_2 = nn.Conv2d(4, 3, 1)
        self.fc_pep_1 = nn.Linear(64, 1)
        
        self.pool = nn.MaxPool2d(1, 1)
        self.relu = nn.ReLU()
                
        self.sequential_pep = nn.Sequential(self.cnn_pep_1,
                                            self.relu, 
                                            self.pool, 
                                            self.cnn_pep_2)
        
        self.sequential_apt = nn.Sequential(self.cnn_apt_1, 
                                            self.relu, 
                                            self.pool, 
                                            self.cnn_apt_2, 
                                            self.relu, 
                                            self.pool, 
                                            self.cnn_apt_3)
        
        self.fc1 = nn.Linear(64, 1)
        
    def forward(self, apt, pep):
        apt = self.sequential_apt(apt).cuda()
        pep = self.sequential_pep(pep).cuda()
        
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


# In[16]:


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.cnn_apt_1 = nn.Conv2d(40, 10, (3,1))
        self.cnn_pep_1 = nn.Conv2d(8, 4, (2,1))
        self.pool = nn.MaxPool2d(1, 1)
        self.relu = nn.ReLU()
                
        self.sequential_pep = nn.Sequential(self.cnn_pep_1,
                                            self.relu, 
                                            self.pool)
        
        self.sequential_apt = nn.Sequential(self.cnn_apt_1, 
                                            self.relu, 
                                            self.pool)
        
        self.fc1 = nn.Linear(96, 1)
        
    def forward(self, apt, pep):
        print(apt.shape)
        print(pep.shape)
        apt = self.sequential_apt(apt).cuda()
        pep = self.sequential_pep(pep).cuda()
        
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


# In[9]:


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.zeros_(m.bias.data)


# ## Sampling

# In[10]:


# Sample x from P_X (assume peptides follow NNK)
def get_x():
    x_idx = np.random.choice(20, 7, p=pvals)
    x = "M"
    for i in x_idx:
        x += aa_list[i]
    return x

# Sample y from P_Y (assume apatamers follow uniform)
def get_y():
    y_idx = np.random.randint(0, 4, 40)
    y = ""
    for i in y_idx:
        y += na_list[i]
    return y

# Generate uniformly from S without replacement
def get_xy(k):
    samples = [full_dataset[i] for i in np.random.choice(len(full_dataset), k, replace=False)]
    return samples

# S' contains S with double the size of S (domain for Importance Sampling)
def get_S_prime(k):
    S_prime = full_dataset[:]
    for _ in range(k):
        S_prime.append((get_y(), get_x()))
    return list(set(S_prime))

# Sample from S' without replacement
def get_xy_prime(k):
    samples = [S_prime[i] for i in np.random.choice(len(S_prime), k, replace=False)]
    return samples

# Returns pmf of a peptide
def get_x_pmf(x):
    pmf = 1
    for char in x[1:]: #skips first char "M"
        pmf *= aa_dict[char]
    return pmf

# Returns pmf of an aptamer
def get_y_pmf():
    return 0.25**40

S_prime = get_S_prime(n)


# ## SGD

# In[11]:


def update(type="original"):
    if type == "original":
        xy = get_xy(1)[0]
    else:
        xy = get_xy_prime(1)[0]
    x = one_hot(xy[0], seq_type='aptamer') 
    y = one_hot(xy[1], seq_type='peptide') 
    #x = torch.FloatTensor(np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))) #(1, 40, 1, 4) original
    x = torch.FloatTensor(np.reshape(x, (1, x.shape[0], x.shape[2], x.shape[1]))) # (1, 40, 4, 1)
    #y = torch.FloatTensor(np.reshape(y, (1, y.shape[0], y.shape[1], y.shape[2]))) #(1, 8, 1, 20) original
    y = torch.FloatTensor(np.reshape(y, (1, y.shape[0], y.shape[2], y.shape[1]))) #(1, 8, 20, 1)
    x.requires_grad=True
    y.requires_grad=True
    x = x.cuda()
    y = y.cuda()
    print(x.shape)
    print(y.shape)
    
    out = model(x, y)
    return xy, out


# In[12]:


def sgd(t=1000, #num of iter
        lamb=1e-5, #hyperparam
        gamma=1e-4): #step size
    
    model.train()
    for a, _ in enumerate(tqdm.tqdm(range(t))):
        xy, out = update()
        out.retain_grad()
        log_out = torch.log(out)
        log_out.retain_grad()
        model.zero_grad()
        log_out.backward()
        
        xy_prime, out_prime = update("prime")
        out_prime = out_prime * get_x_pmf(xy_prime[0]) * get_y_pmf() * 2 * n
        out_prime.retain_grad()
        model.zero_grad()
        out_prime.backward()
        
        const = 0 if xy_prime in full_dataset else 1 #indicator
        g = log_out.grad - lamb*const*out_prime.grad
        g = g.item()
        
        #Update the weights according to SGD
        for param in model.parameters():
            param.data += gamma * g


# ## Recall & evaluate

# In[13]:


# Eval on test set of size k (split from our dataset)
def recall_eval(k):
    correct = 0
    count = 0
    binding_outputs = []
    model.eval()
    for _, (aptamer, peptide) in enumerate(tqdm.tqdm(test_loader)):
        if count > k:
            break
        pep = one_hot(peptide, seq_type='peptide')
        apt = one_hot(aptamer, seq_type='aptamer')
        #pep = torch.FloatTensor(np.reshape(pep, (1, pep.shape[1], pep.shape[2], pep.shape[0]))).cuda()
        pep = torch.FloatTensor(np.reshape(pep, (1, pep.shape[2], pep.shape[1], pep.shape[0]))).cuda()
        #apt = torch.FloatTensor(np.reshape(apt, (1, apt.shape[1], apt.shape[2], apt.shape[0]))).cuda()
        apt = torch.FloatTensor(np.reshape(apt, (1, apt.shape[2], apt.shape[1], apt.shape[0]))).cuda()
        output = model(apt, pep).cpu().detach().numpy().flatten()[0]
        binding_outputs.append('%.2f'% output)
        if output > 0.5:
            correct += 1
        count += 1
    recall = 100*correct/count #recall rate of k samples
    return recall, binding_outputs #list of k outputs


# In[14]:


def convert(apt, pep): 
    apt = one_hot(apt, seq_type='aptamer') #(40, 1, 4)
    pep = one_hot(pep, seq_type='peptide') #(8, 1, 20)
    #apt = torch.FloatTensor(np.reshape(apt, (1, apt.shape[0], apt.shape[2], apt.shape[1]))).cuda() #(1, 40, 4, 1)
    apt = torch.FloatTensor(np.reshape(apt, (1, apt.shape[1], apt.shape[0], apt.shape[2]))).cuda() #(1, 1, 40, 4)
    #pep = torch.FloatTensor(np.reshape(pep, (1, pep.shape[0], pep.shape[2], pep.shape[1]))).cuda() #(1, 8, 20, 1)
    pep = torch.FloatTensor(np.reshape(pep, (1, pep.shape[1], pep.shape[0], pep.shape[2]))).cuda() #(1, 1, 8, 20)
    return apt, pep

# Eval on m new unseen pairs(not in our dataset)
def evaluate(m):
    model.eval()
    outputs = []
    for _ in range(m):
        x, y = get_x(), get_y()
        apt, pep = convert(y, x)
        output = model(apt, pep).cpu().detach().numpy().flatten()[0]
        outputs.append('%.2f'% output)
    return outputs #list of m outputs


# ## Hyperparameter tuning

# In[17]:


gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
lambdas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
recalls = []
scores = []

m = int(1e2) # number of unknown samples
k = int(30) # number of binding samples (test set size is 118262, k is just some limit we set)

M = np.zeros((len(gammas), len(lambdas)))
for g in range(len(gammas)):
    for l in range(len(lambdas)):
        model = SimpleConvNet()
        model.apply(weights_init)
        model.cuda()
        print("Training...")
        sgd(t=100, gamma=gammas[g], lamb=lambdas[l])
        print("Evaluating...")
        # use for AUC
        recall, binding_outputs = recall_eval(k)
        unknown_outputs = evaluate(m)
        scores.append((unknown_outputs, binding_outputs))
        # use for heatmap
        M[g][l] += recall
        recalls.append(recall)
        print("Recall with gamma: "+ str(gammas[g]) + " , lambda: " + str(lambdas[l]) + " recall: ", '%.2f'% recall)


# ## Table and plots

# In[ ]:


# Table of recalls with different params
idx = sorted(range(len(recalls)), key=lambda k: recalls[k])
for i in idx:
    g = gammas[i//len(gammas)]
    l = lambdas[i%len(lambdas)]
    print("Gamma: ", "%.5f" % g, "Lambda: ", "%.5f" % l, "Recall: ", "%.2f" % recalls[i])


# In[ ]:


# Heatmap of recalls
mat = sns.heatmap(M, vmin=0, vmax=100)
plt.show()


# In[ ]:


# AUC (one config for now)
unknown, binding = scores[0]
total = unknown + binding
plt.hist(total, 50, histtype='step', density=True, cumulative=True)
plt.show()


# ## Test other NN models

# In[ ]:


model = SimpleConvNet()
model.apply(weights_init)
model.cuda()
print("Training...")
sgd(t=1000, gamma=0.01, lamb=0.01)
print("Evaluating...")
# use for AUC
recall, binding_outputs = recall_eval(50)
unknown_outputs = evaluate(500)
# use for heatmap
print("Recall with gamma: "+ str(0.01) + " , lambda: " + str(0.01) + " recall: ", '%.2f'% recall)


# In[ ]:




