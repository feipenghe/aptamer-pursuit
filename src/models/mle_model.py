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
from torch.optim import Adam, SGD
import tqdm


# ## Dataset and one-hot encoding

# In[2]:


BATCH_SIZE = 1
random.seed(42)
device = torch.device("cudo:0" if torch.cuda.is_available() else 'cpu')


# In[3]:


# TODO: Take out controls and generate a new dataset
aptamer_dataset_file = "../data/aptamer_dataset.json"
clustered_dataset_file = "../data/clustered_aptamer_dataset.json"

def construct_dataset():
    with open(clustered_dataset_file, 'r') as f:
        aptamer_data = json.load(f)
    full_dataset = []
    for aptamer in aptamer_data:
        peptides = aptamer_data[aptamer]
        for peptide, _ in peptides:
            peptide = peptide.replace("_", "")
            if len(aptamer) == 40 and len(peptide) == 7:
                peptide = "M" + peptide
                full_dataset.append((aptamer, peptide))
    return list(set(full_dataset)) #removed duplicates


# In[4]:


full_dataset = construct_dataset()
aptamers = [p[0] for p in full_dataset]
peptides = [p[1] for p in full_dataset]
training_set = full_dataset[:int(0.8*len(full_dataset))]
test_set = full_dataset[int(0.8*len(full_dataset)):]
print(str(len(test_set)))


# In[5]:


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, training_set):
        super(TrainDataset, self).__init__() 
        self.training_set = training_set
        n = len(training_set)
        self.training_set = training_set[:n-n%BATCH_SIZE]
        
    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, idx):
        aptamer, peptide = self.training_set[idx]
        return aptamer, peptide
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_set):
        super(TestDataset, self).__init__() 
        self.test_set = test_set
        n = len(test_set)
        self.test_set = test_set[:n-n%BATCH_SIZE]
        
    def __len__(self):
        return len(self.test_set)

    def __getitem__(self, idx):
        aptamer, peptide = self.test_set[idx]
        return aptamer, peptide


# In[6]:


train_dataset = TrainDataset(training_set)
test_dataset = TestDataset(test_set)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


# In[7]:


na_list = ['A', 'C', 'G', 'T']
aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W', 'Y']
pvals = [0.089]*3 + [0.065]*5 + [0.034]*12
aa_dict = dict(zip(aa_list, pvals))


# In[8]:


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


# ## NN Model

# In[9]:


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
        
        #self.dropout = nn.Dropout(0.1)
        
        self.sequential_pep = nn.Sequential(self.cnn_pep_1,
                                            #self.dropout,
                                            self.relu, 
                                            self.pool, 
                                            self.cnn_pep_2)
        
        self.sequential_apt = nn.Sequential(self.cnn_apt_1, 
                                            #self.dropout,
                                            self.relu, 
                                            self.pool, 
                                            self.cnn_apt_2, 
                                            #self.dropout,
                                            self.relu, 
                                            self.pool, 
                                            self.cnn_apt_3)
        
        self.fc1 = nn.Linear(64, BATCH_SIZE)
        
    def forward(self, apt, pep):
        apt = self.sequential_apt(apt)
        pep = self.sequential_pep(pep)
        
        apt = apt.view(-1, 1).T
        pep = pep.view(-1, 1).T
        
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
    
    # Change loss to be custom loss
    def loss(self, prediction, label):
        l = nn.MSELoss()
        label = torch.FloatTensor(label)
        return l(torch.FloatTensor(prediction), label)


# In[10]:


model = ConvNet()
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.zeros_(m.bias.data)

model.apply(weights_init)
model.to(device)


# ## Sampling methods

# In[11]:


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
    S_prime_new = S_prime[:]
    S_prime_new = S_prime_new[-n:]
    return list(set(S_prime)), S_prime_new

# Sample from S' without replacement
def get_xy_prime(k):
    samples = [S_prime[i] for i in np.random.choice(len(S_prime), k, replace=False)]
    return samples


# ## Test whether naive sampling would generate pairs in S -- nope

# In[12]:


n = len(full_dataset)
S_prime, S_prime_new = get_S_prime(n)
print("Size of S: ", n)
print("Size of naively sampled dataset: ", len(S_prime_new))
diff = set(full_dataset) - set(S_prime_new)
print("Size of set difference: ", len(diff))


# ## Motivates importance sampling in SGD

# In[13]:


# Returns pmf of a peptide
def get_x_pmf(x):
    pmf = 1
    for char in x[1:]: #skips first char "M"
        pmf *= aa_dict[char]
    return pmf

# Returns pmf of an aptamer
def get_y_pmf():
    return 0.25**40


# ## SGD

# In[14]:


def sgd(t=500, #num of iter
        lamb=1e-5, #hyperparam
        gamma=1e-5): #step size
    
    model.train()
    for a, _ in enumerate(tqdm.tqdm(range(t))):
        # From our full dataset
        xy = get_xy(1)[0] #sample (x, y) from S
        x = one_hot(xy[0], seq_type='aptamer') 
        y = one_hot(xy[1], seq_type='peptide') 
        x = torch.FloatTensor(np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2])))
        y = torch.FloatTensor(np.reshape(y, (1, y.shape[0], y.shape[1], y.shape[2])))
        x.requires_grad=True
        y.requires_grad=True
        out = model(x, y)
        out.retain_grad()
        model.zero_grad()
        out.backward()
        
        # From the generated dataset
        xy_prime = get_xy_prime(1)[0] #sample (x', y') from S'
        x_prime = one_hot(xy_prime[0], seq_type='aptamer') 
        y_prime = one_hot(xy_prime[1], seq_type='peptide')
        x_prime = torch.FloatTensor(np.reshape(x_prime, (1, x_prime.shape[0], x_prime.shape[1], x_prime.shape[2])))
        y_prime = torch.FloatTensor(np.reshape(y_prime, (1, y_prime.shape[0], y_prime.shape[1], y_prime.shape[2])))
        x_prime.requires_grad=True
        y_prime.requires_grad=True
        
        out_prime = model(x_prime, y_prime)
        out_prime = out_prime * get_x_pmf(xy_prime[0]) * get_y_pmf() * 2 * n
        out_prime.retain_grad()
        model.zero_grad()
        out_prime.backward()
        
        const = 0 if xy_prime in full_dataset else 1 #indicator
        g = torch.log(out.grad) - lamb*const*out_prime.grad
        g = g.item()
        
        # Update the weights according to SGD
        for param in model.parameters():
            param.data += gamma * g


# In[15]:


#sgd(t=1000)


# ### Calculating recall

# In[ ]:


# Recall test
def evaluate():
    correct = 0
    incorrect = 0
    model.eval()
    for batch_idx, (aptamer, peptide) in enumerate(tqdm.tqdm(test_loader)):
        pep = one_hot(peptide, seq_type='peptide')
        apt = one_hot(aptamer, seq_type='aptamer')

        pep = torch.FloatTensor(np.reshape(pep, (1, pep.shape[1], pep.shape[2], pep.shape[0])))
        apt = torch.FloatTensor(np.reshape(apt, (1, apt.shape[1], apt.shape[2], apt.shape[0])))

        output = model(apt, pep).detach().numpy().flatten()
        for i in range(output.shape[0]):
            o = output[i]
            if o > 0.5:
                correct += 1
            else:
                incorrect += 1
        
    return (100* correct/(correct + incorrect))


# ## Hyperparameter tuning

# In[ ]:


gamma_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
lambda_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for g in gamma_values:
    for l in lambda_values:
        model = ConvNet()
        model.apply(weights_init)
        model.to(device)
        print("Training...")
        sgd(t=1000, gamma=g, lamb=l)
        print("Evaluating...")
        recall = evaluate()
        print("Recall with gamma: "+ str(g) + " , lambda: " + str(l) + " recall: ", recall)


# ## Evaluate

# In[ ]:




