#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from ledidi import Ledidi
from ledidi import TensorFlowRegressor, PyTorchRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam


# ## Tensorflow Regressor

# In[ ]:


# These two objects are necessary for the Basenji model
class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__()
    
    def call(self, x, training):
        return tf.keras.activations.sigmoid(1.702 * x) * x

class StochasticShift(tf.keras.layers.Layer):
    def __init__(self, shift_max=0, pad='uniform', **kwargs):
        super(StochasticShift, self).__init__()

    def call(self, seq_1hot, training):
        return seq_1hot

custom_objects = {
    'StochasticShift': StochasticShift, 
    'GELU': GELU
}

model = load_model("model_human.h5", custom_objects)
model.compile()

regressor = TensorFlowRegressor(model=model)

# Index 687 is CTCF signal in GM12878
mask = np.zeros((1, 1024, 5313), dtype='float32')
mask[:, :, 687] = 1.0

mutator = Ledidi(regressor, mask=None, l=1e2)

sequence = np.load("CTCF-seqs.npz")['arr_0'].astype('float32')[0].reshape(1, 131072, 4)
epi = model.predict(sequence)
print("Epi shape: ", epi.shape)

desired_epi = epi.copy()
desired_epi[0, 487:537, 687] = 0.0

edited_sequence = mutator.fit_transform(sequence, desired_epi)
found_epi = model.predict(edited_sequence.astype('float32'))[0, :, 687]


# ## PyTorch Regressor

# In[2]:


# Pytorch oracle model
class SingleAlphabetComplexNet(nn.Module):
    def __init__(self):
        super(SingleAlphabetComplexNet, self).__init__()
        self.name = "SingleAlphabetComplexNet"
        
        self.cnn_1 = nn.Conv1d(24, 50, 3) 
        self.cnn_2 = nn.Conv1d(50, 100, 3)
        self.cnn_3 = nn.Conv1d(100, 200, 3)
        self.cnn_4 = nn.Conv1d(200, 400, 3)
        self.cnn_5 = nn.Conv1d(400, 800, 3)
        self.cnn_6 = nn.Conv1d(800, 1000, 3, padding=2)
        self.cnn_7 = nn.Conv1d(1000, 800, 3, padding=2)
        self.cnn_8 = nn.Conv1d(800, 700, 3, padding=2)
        self.cnn_9 = nn.Conv1d(700, 500, 1)
        
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)         
        self.fc1 = nn.Linear(500, 1800)
        self.fc2 = nn.Linear(1800, 1)
        
    def forward(self, pair):
        # pair input size [1, 48, 24]
        pair = pair.permute(0, 2, 1)
        
        pair = self.relu(self.cnn_1(pair))
        pair = self.relu(self.cnn_2(pair))
        pair = self.pool1(self.relu(self.cnn_3(pair)))
        pair = self.pool1(self.relu(self.cnn_4(pair)))
        pair = self.pool1(self.relu(self.cnn_5(pair)))        
        pair = self.pool1(self.relu(self.cnn_6(pair)))
        pair = self.pool1(self.relu(self.cnn_7(pair))) 
        pair = self.pool1(self.relu(self.cnn_8(pair)))
        pair = self.pool1(self.relu(self.cnn_9(pair))) 

        pair = pair.view(-1, 1).T
        
        pair = self.fc2(self.fc1(pair))
        x = torch.sigmoid(pair)
        return x

model = SingleAlphabetComplexNet()
checkpointed_model = '../model_checkpoints/binary/%s/%s.pth' % ("SingleAlphabetComplexNet", "06172020")
checkpoint = torch.load(checkpointed_model)
model.load_state_dict(checkpoint['model_state_dict'])


# In[3]:


random_seq = torch.FloatTensor(np.zeros((1, 48, 24)))
model(random_seq)


# In[5]:


regressor = PyTorchRegressor(model=model, verbose=False)


mutator = Ledidi(regressor, mask=None, l=1e2)

sequence = random_seq
epi = model(sequence)

desired_epi = epi.clone()
print("Desired Epi shape: ", desired_epi.shape)

edited_sequence = mutator.fit_transform(sequence, desired_epi)
#found_epi = model.predict(edited_sequence.astype('float32'))[0, :, 687]


# ## Recover original sequence

# In[ ]:


# Need to reshape found_epi into the desired epi shape and then 
edited_sequence.shape


# In[ ]:


sequence = numpy.argmax(edited_sequence, axis=2)[0]


# In[ ]:


str_seq = ""
for i in range(len(sequence)):
    if sequence[i] == 0:
        str_seq += 'A'
    elif sequence[i] == 1:
        str_seq += 'C'
    elif sequence[i] == 2:
        str_seq += 'G'
    else:
        str_seq += 'T'
str_seq


# In[ ]:




