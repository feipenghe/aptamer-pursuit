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
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder


class ConvTwoHead(nn.Module):
    def __init__(self, embedding_type = 'one_hot'):
        super(ConvTwoHead, self).__init__()
        self.name = "ConvTwoHead"
        self.single_alphabet=False
        self.apt_vocab_size = 4
        self.pep_vocab_size = 20
        self.apt_embedding_dim = self.apt_vocab_size
        self.pep_embedding_dim = self.pep_vocab_size
        self.apt_length = 40
        self.pep_length = 8
        self.embedding_type = embedding_type
        # self.embedding_type = "embedding"
        if self.embedding_type == "embedding":
            self.apt_embedding  = nn.Embedding(num_embeddings=self.apt_vocab_size, embedding_dim=self.apt_embedding_dim)
            self.pep_embedding  =  nn.Embedding(num_embeddings=self.pep_vocab_size, embedding_dim=self.pep_embedding_dim)
        elif self.embedding_type == "bio_emb":
            # only initialize apt embedding
            # pep embedding is used in
            self.apt_embedding = nn.Embedding(num_embeddings=self.apt_vocab_size, embedding_dim=self.apt_embedding_dim)
            self.pep_embedder = SeqVecEmbedder()
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
        if self.embedding_type == "one_hot":
            apt, pep = vectorize_token(apt, pep)
        elif self.embedding_type == "embedding":
            apt = self.apt_embedding(apt)
            pep = self.pep_embedding(pep)
        else:
            apt = self.apt_embedding(apt)
            pep = self.pep_embedder(pep)


        apt = apt.view(len(apt), 4, -1).float()
        pep = pep.view(len(pep), 20, -1).float()
        apt = self.cnn_apt(apt)
        pep = self.cnn_pep(pep)
        
        apt = apt.view(len(apt), -1)
        pep = pep.view(len(pep), -1)
        x = torch.cat((apt, pep), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    def accuracy(self, predictions, labels):
        return (torch.sum((predictions > 0.5) == labels).item()/len(labels))
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
def loss(prediction, label, reduction = 'mean'):
    loss_val = F.binary_cross_entropy(prediction.squeeze(), label.squeeze(), reduction = reduction)
    return loss_val
'''
enc_apt = OneHotEncoder()
enc_pep = OneHotEncoder()
na_list = ['A', 'C', 'G', 'T']  # nucleic acids
aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W',
                   'Y']
enc_apt.fit(np.array(na_list).T)
enc_pep.fit(np.array(aa_list).T)
'''
def vectorize_token(apt, pep):
    """
    One hot encoding
    """
    apt_vocab_size = 4
    pep_vocab_size = 20
    apt = nn.functional.one_hot(apt, num_classes=apt_vocab_size).float()
    pep = nn.functional.one_hot(pep, num_classes=pep_vocab_size).float()
    return apt, pep




class LinearTwoHead(nn.Module):
    def __init__(self, embedding_type):
        super(LinearTwoHead, self).__init__()
        self.model_name = "LinearTwoHead"
        self.single_alphabet=False
        self.apt_vocab_size = 4
        self.pep_vocab_size = 20
        self.apt_embedding_dim = self.apt_vocab_size * 4
        self.pep_embedding_dim = 1024
        self.apt_length = 40
        self.pep_length = 8

        self.embedding_type = embedding_type
        # self.embedding_type = "embedding"
        if self.embedding_type == "embedding":
            self.apt_embedding  = nn.Embedding(num_embeddings=self.apt_vocab_size, embedding_dim=self.apt_embedding_dim)
            self.pep_embedding  =  nn.Embedding(num_embeddings=self.pep_vocab_size, embedding_dim=self.pep_embedding_dim)
        elif self.embedding_type == "bio_emb":
            # only initialize apt embedding
            # pep embedding is used in
            self.apt_embedding = nn.Embedding(num_embeddings=self.apt_vocab_size, embedding_dim=self.apt_embedding_dim)
            # self.pep_embedder =  SeqVecEmbedder()

        apt_dim = self.apt_embedding_dim * self.apt_length  # 4 x 40 = 160
        self.fc_apt_1 = nn.Linear(apt_dim, apt_dim*4) # batch size x length(40) x embedding
        self.fc_apt_2 = nn.Linear(apt_dim*4, apt_dim*2)
        self.fc_apt_3 = nn.Linear(apt_dim*2, 100)

        # pep_dim = self.pep_embedding_dim * self.pep_length
        BIO_EMB_DIM = 1024
        pep_dim = BIO_EMB_DIM * 3
        self.fc_pep_1 = nn.Linear(pep_dim, pep_dim*2)
        self.fc_pep_2 = nn.Linear(pep_dim*2, 100)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2) 

        
        
        self.fc_apt = nn.Sequential(self.fc_apt_1, nn.ReLU(), self.fc_apt_2, nn.ReLU(),  self.fc_apt_3)
        self.fc_pep = nn.Sequential(self.fc_pep_1, nn.ReLU(),  self.fc_pep_2)
        
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 1)
        self.name = self.model_name + "_" + self.embedding_type  # name for saving



    def forward(self, apt, pep):
        # apt = apt.view(-1, 1).T
        # pep = pep.view(-1, 1).T

        if self.embedding_type == "one_hot":
            apt, pep = vectorize_token(apt, pep)
        elif self.embedding_type == "embedding":
            apt = self.apt_embedding(apt)
            pep = self.pep_embedding(pep)
        elif self.embedding_type == "bio_emb": # bio embedding
            apt = self.apt_embedding(apt)
            # import pdb
            # pdb.set_trace()
            # pep = torch.tensor(self.pep_embedder.embed(pep)).cuda() # embed_batch(pep)
        # if self.embedding_type == "one_hot":
        #     apt, pep = vectorize_token(apt, pep)
        #
        # else:
        #     apt = self.apt_embedding(apt)
        #     pep = self.pep_embedding(pep)
        # print(apt.shape)
        # import pdb
        # pdb.set_trace()
        apt = apt.view(apt.size(0), -1).float()
        # import pdb
        # pdb.set_trace()
        # pep = pep.permute(1, 0, 2)
        pep = pep.contiguous().view(pep.size(0), -1).float()

        # print(apt.shape)
        # exit()
        apt = self.fc_apt(apt)
        pep = self.fc_pep(pep)
        x = torch.cat((apt, pep), 1)

        x = self.fc2(self.fc1(x))
        x = torch.sigmoid(x)
        return x       

    def accuracy(self, predictions, labels):
        return (torch.sum((predictions > 0.5) == labels).item()/len(labels))


class RNNtwoHead(nn.Module):
    def __init__(self, RNN_type, embedding_type):
        super(RNNtwoHead, self).__init__()
        self.model_name = RNN_type
        self.single_alphabet = False
        self.apt_vocab_size = 4
        self.pep_vocab_size = 20
        self.apt_embedding_dim = self.apt_vocab_size * 4
        self.pep_embedding_dim = 1024
        self.apt_length = 40
        self.pep_length = 8

        self.hidden_dim = 300

        self.embedding_type = embedding_type
        # self.embedding_type = "embedding"
        if self.embedding_type == "embedding":
            self.apt_embedding = nn.Embedding(num_embeddings=self.apt_vocab_size, embedding_dim=self.apt_embedding_dim)
            self.pep_embedding = nn.Embedding(num_embeddings=self.pep_vocab_size, embedding_dim=self.pep_embedding_dim)
        elif self.embedding_type == "bio_emb":
            # only initialize apt embedding
            # pep embedding is used in
            self.apt_embedding = nn.Embedding(num_embeddings=self.apt_vocab_size, embedding_dim=self.apt_embedding_dim)
            # self.pep_embedder =  SeqVecEmbedder()

        apt_hidden_dim = 100
        pep_hidden_dim = 300

        BIO_EMB_DIM = 1024
        pep_dim = BIO_EMB_DIM
        if RNN_type == "LSTM":
            self.apt_rnn = nn.LSTM(input_size = self.apt_embedding_dim, hidden_size = apt_hidden_dim, batch_first=True )
            self.pep_rnn = nn.LSTM(input_size =pep_dim, hidden_size = pep_hidden_dim, batch_first=True)


        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)

        self.fc = nn.Sequential(nn.Linear(apt_hidden_dim + pep_hidden_dim, 100), nn.Linear(100, 1), nn.Sigmoid())

        self.name = self.model_name + "_" + self.embedding_type  # name for saving

    def forward(self, apt, pep):
        # apt = apt.view(-1, 1).T
        # pep = pep.view(-1, 1).T

        if self.embedding_type == "one_hot":
            apt, pep = vectorize_token(apt, pep)
        elif self.embedding_type == "embedding":
            apt = self.apt_embedding(apt)
            pep = self.pep_embedding(pep)
        elif self.embedding_type == "bio_emb":  # bio embedding
            apt = self.apt_embedding(apt)
            # import pdb
            # pdb.set_trace()
            # pep = torch.tensor(self.pep_embedder.embed(pep)).cuda() # embed_batch(pep)

        # apt = apt.view(apt.size(0), -1).float()
        #
        # pep = pep.contiguous().view(pep.size(0), -1).float()


        _, (apt_hidden, _ ) = self.apt_rnn(apt) # batch
        _, (pep_hidden, _ ) = self.pep_rnn(pep) # over consdier the embedding of three heads as a sequential data
        # apt = self.fc_apt(apt)
        # pep = self.fc_pep(pep)
        # import pdb
        # pdb.set_trace()
        apt_hidden = apt_hidden.squeeze()
        pep_hidden = pep_hidden.squeeze()
        x = torch.cat((apt_hidden, pep_hidden), 1)
        x = self.fc(x)
        return x

    def accuracy(self, predictions, labels):
        return (torch.sum((predictions > 0.5) == labels).item() / len(labels))



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #         x = x + self.pe[:x.size(0), :]
        #         print("x in forward: ", x.shape)
        x = x.permute(1, 0, 2)
        # seq len x batch x embedding size
        #         print("x: ", x.shape)
        #         print(" self.pe[:, :x.size(0)]: ",  self.pe[:, :x.size(0)].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


import math
#
#
#
class AptPepTransformer(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(AptPepTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)  # they should share the same encoder
        #         self.tgt_encoder = nn.Embedding(self.vocab_size, self.feature_size)

        # TODO: another dimentions = self.embed_src(src) * math.sqrt(self.d_model)
        self.pos_encoder = PositionalEncoding(self.feature_size)  # simple argument setting
        encoder_layers = nn.TransformerEncoderLayer(self.feature_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.decoder = nn.Linear(self.feature_size, self.vocab_size)

        # This shares the encoder and decoder weights as described in lecture.
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.zero_()

        self.best_accuracy = -1

    def forward(self, src, src_mask, tgt=None):  # hidden_state=None, cell_state = None
        batch_size = src.shape[0]
        sequence_length = src.shape[1]

        # TODO finish defining the forward pass.
        # You should return the output from the decoder as well as the hidden state given by the gru.
        src = self.encoder(src) * math.sqrt(self.feature_size)  # TOTHINK: why math.sqrt(self.feature_size) here
        src = self.pos_encoder(src)
        #         print("data in transformer: ", src.shape)
        #         print('mask in transformer: ', src_mask.shape)
        output = self.transformer_encoder(src, src_mask)

        output = self.decoder(output)  # TODO: shouldn't it predict a single character

        # 100, 256, 512
        # seq len, batch size, feature size

        # nn shouldn't depend on seq len and batch size
        # 100, 256, 89
        # 100, 256, 1 (sampling)
        # 1, 256, 1 (take the last one)

        # permute (1, 0, 2)
        # view(x.size(0), -1)
        # nn.Linear(seq_len x feature_dim, vocab)

        # 256 x 100 x 89
        # view(256, -1)
        # it should be 1 x 89

        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # This defines the function that gives a probability distribution and implements the temperature computation.
    def inference(self, x, x_mask, temperature=1):
        """
        x: 1-D tensor representing the input sequence
        """
        x = x.unsqueeze(0)  # add a batch dimension
        #         import ipdb
        #         ipdb.set_trace()
        x = self.forward(x, x_mask)
        # after forward, seq len x batch x embedding
        x = x[-1, :, :]  # take last character output   # TODO: it shouldn't predict a single character
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)

        return x

    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)

# class LSTM(nn.Module):
#     ## pure LSTM
#     def __init__(self):
#         super(Transformer).__init__()






# # return number of correct (not percentage)
# def num_correct(label, pred):
#     #assert label.ndim == 1 and label.size() == pred.size()
#     pred = pred.squeeze() > 0.5
#     return (label.squeeze() == pred).sum().item()
#
# def train(model, device, train_loader, optimizer, epoch, log_interval):
#     model.train()
#     losses = []
#     for batch_idx, (apt, pep, label) in enumerate(train_loader):
#         apt, pep, label = apt.to(device), pep.to(device), label.to(device)
#         optimizer.zero_grad()
#         output = model(apt, pep)
#         loss = loss(output, label)
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(apt), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#     return np.mean(losses)
#
# def test(model, device, test_loader, log_interval=None):
#     model.eval()
#     test_loss = 0
#     correct = 0
#
#     with torch.no_grad():
#         for batch_idx, (apt, pep, label) in enumerate(test_loader):
#             apt, pep, label = apt.to(device), pep.to(device), label.to(device)
#             output = model(apt, pep)
#             test_loss_on = loss(output, label, reduction='sum').item()
#             test_loss += test_loss_on
#             correct += num_correct(output, label)
#             if log_interval is not None and batch_idx % log_interval == 0:
#                 print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     batch_idx * len(apt), len(test_loader.dataset),
#                     100. * batch_idx / len(test_loader), test_loss_on))
#
#     test_loss /= len(test_loader.dataset)
#     test_accuracy = 100. * correct / len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset), test_accuracy))
#     return test_loss, test_accuracy