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
from util import BinaryDataset





if __name__ == '__main__':
    torch.manual_seed(12345)
    k = 10000
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    na_list = ['A', 'C', 'G', 'T']  # nucleic acids
    aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W',
               'Y']  # amino acids

    NNK_freq = [0.09375] * 3 + [0.0625] * 5 + [0.03125] * 12  # freq of 21 NNK codons including the stop codon
    sum_20 = 0.0625 * 5 + 0.09375 * 3 + 0.03125 * 12  # sum of freq without the stop codon
    pvals = [0.09375 / sum_20] * 3 + [0.0625 / sum_20] * 5 + [0.03125 / sum_20] * 12  # normalize freq for 20 codons
    pvals = [0.09375 / sum_20] * 3 + [0.0625 / sum_20] * 5 + [0.03125 / sum_20] * 11 + \
            [1 - sum([0.09375 / sum_20] * 3 + [0.0625 / sum_20] * 5 + [0.03125 / sum_20] * 11)]
    # adjust sum to 1 due to numerical issue
    uniform_pvals = [0.05] * 20

    encoding_style = 'regular'
    lambda_val = 1
    alpha = 0.9
    beta = 0.1

    aa_list_2 = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    pvals_2 = [0.07668660126327106, 0.035596693992742914, 0.02474465797607849, 0.04041795457599785, 0.02319916677865878,
               0.1149711060341352, 0.02187206020696143, 0.021972853111140975, 0.030170675984410696, 0.0904280338664158,
               0.030069883080231154, 0.017672355866147026, 0.03937642789947588, 0.03156497782556108, 0.1183812659588765,
               0.07880325225104153, 0.043290552345114905, 0.08557317564843435, 0.053369842763069476,
               0.02183846257223492]

    original_blosum62 = {}

    blosum_matrix = np.zeros((20, 20))
    for i, aa in enumerate(original_blosum62.keys()):
        sims = original_blosum62[aa]
        for j, s in enumerate(sims):
            blosum_matrix[i][j] = s
    u, V = LA.eig(blosum_matrix)
    clipped_u = u
    clipped_u[clipped_u < 0] = 0
    lamb = np.diag(clipped_u)
    T = V
    clip_blosum62 = {}
    for i, aa in enumerate(original_blosum62.keys()):
        clip_blosum62[aa] = np.dot(np.sqrt(lamb), V[i])

    binary_ds1 = BinaryDataset(filepath="./data/pos_datasets/experimental_replicate_1.json", distribution='NNK',
                               negfilepath='./data/neg_datasets/neg1_all_pairs_noArgi_noHis.json')
    # binary_ds2 = BinaryDataset(filepath="./data/pos_datasets/experimental_replicate_2.json", distribution='NNK',
    #                            negfilepath='./data/neg_datasets/neg2_all_pairs_noArgi_noHis.json')
    data_lodaer1 = DataLoader(binary_ds1, batch_size = 20)
    print(data_lodaer1.next())


    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]


    src = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    fields = {'response': ('r', src)}

    train_data, test_data, validation_data = TabularDataset.splits(
        path='FilePath',
        train='trainset.json',
        test='testset.json',
        validation='validationset.json',
        format='json',
        fields=fields
    )
