import os, sys
import numpy as np
import json
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from util import BinaryDataset
from model import *
from tqdm import tqdm

# TODO: add training accuracy
# TODO: check performance of nn
# TODO: data classes and write into json files
# TODO: scheduler with hyperparameter



def comp_loss(prediction, label, reduction = 'mean'):
    # loss_val = F.binary_cross_entropy(prediction.squeeze(), label.squeeze().float(), reduction = reduction)
    loss_val = F.binary_cross_entropy(prediction.squeeze(), label.squeeze().float())
    return loss_val

def to_device(mode, apt, pep, label):
    if mode == "parallel":
        apt, pep, label = apt.cuda(), pep.cuda(), label.cuda()
    else:
        apt, pep, label = apt.to(device), pep.to(device), label.to(device)
    return apt, pep, label

def comp_accuracy(mode, model, output, label):
    return get_model(mode, model).accuracy(output, label) * len(label)


def get_model(mode, model):
    if mode == "parallel":
        return model.module
    else:
        return model

def train(model, comp_loss, device, train_loader, val_loader, optimizer, epoch= 200, log_interval = 100):
    if type(device) == list:
        print("enable data parallel on: ", device)
        mode = "parallel"
        model = nn.DataParallel(model, device_ids=device)
        model.to(device[0])
    else:
        mode = "single"
        print("start running model on: ", device)
        model = model.to(device)

    losses = []
    val_acc_l = []
    epoch_train_loss_l = [] # record average train loss
    epoch_val_loss_l = [] # record average validation loss
    best_val_acc = 0
    for e in range(epoch):
        model.train()
        tqdm_train_loader = tqdm(train_loader)
        epoch_train_loss = 0.
        count = 0
        for batch_idx, (apt, pep, label) in enumerate(tqdm_train_loader):
            apt, pep, label = to_device(mode, apt, pep, label)
            # if mode == "parallel":
            #     apt, pep, label = apt.cuda(), pep.cuda(), label.cuda()
            # else:
            #     apt, pep, label = apt.to(device), pep.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(apt, pep)
            loss = comp_loss(output, label)

            loss.backward()
            loss = loss.detach().item()
            epoch_train_loss += loss
            losses.append(loss)  # record loss and detach the computation graph
            optimizer.step()
            if batch_idx % log_interval == 0:
                tqdm_train_loader.set_description_str('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(apt), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
            count += 1
        epoch_train_loss_l.append(epoch_train_loss/count)


        model.eval()
        tqdm_val_loader = tqdm(val_loader)
        val_loss = 0.
        val_correct = 0
        cur_num_labels = 0

        count = 0
        epoch_val_loss = 0.
        for batch_idx, (apt, pep, label) in enumerate(tqdm_val_loader):
            apt, pep, label = to_device(mode, apt, pep, label)
            with torch.no_grad():
                output = model(apt, pep)
                loss = comp_loss(output, label).item()
                val_loss += loss
                epoch_val_loss += loss
                val_correct += get_model(mode, model).accuracy(output, label) * len(label)
                cur_num_labels += len(label)
                tqdm_val_loader.set_description_str(f"Validation Epoch: {e}      [Loss]: {val_loss/(batch_idx+1):.4f}    [Acc]: {val_correct/cur_num_labels}")
            count += 1
            epoch_val_loss += val_loss
        epoch_val_loss_l.append(epoch_val_loss/count)
        avg_val_acc = val_correct *1.0 / cur_num_labels
        val_acc_l.append(avg_val_acc)
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({"best_epoch": e,
                        "best_model": get_model(mode, model).state_dict(),
                        "best_val_acc": best_val_acc}, f"./best_model_{get_model(mode, model).name}.pt"
                       )
        # print("epoch_train_loss_l: ", epoch_train_loss_l)
        # print("epoch_val_loss_l: ", epoch_val_loss_l)
        # epoch_train_loss_l = []  # record average train loss
        # epoch_val_loss_l = []  # record average validation loss
    with open(f"./best_model_{get_model(mode, model).name}.json", "w") as fp:
        json.dump({"epoch_train_loss_l": epoch_train_loss_l,
        "epoch_val_loss_l": epoch_val_loss_l}, fp)

    return np.mean(losses)



def test(model, comp_loss, device, test_loader, log_interval=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (apt, pep, label) in enumerate(test_loader):
            apt, pep, label = apt.to(device), pep.to(device), label.to(device)
            output = model(apt, pep)
            test_loss_on = comp_loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            correct += num_correct(output, label)
            if log_interval is not None and batch_idx % log_interval == 0:
                print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(apt), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy


def num_correct(label, pred):
    #assert label.ndim == 1 and label.size() == pred.size()
    pred = pred.squeeze() > 0.5
    return (label.squeeze() == pred).sum().item()

import argparse
if __name__ == '__main__':
    # torch.manual_seed(12345)
    # k = 10000
    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # na_list = ['A', 'C', 'G', 'T']  # nucleic acids
    # aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W',
    #            'Y']  # amino acids
    #
    # NNK_freq = [0.09375] * 3 + [0.0625] * 5 + [0.03125] * 12  # freq of 21 NNK codons including the stop codon
    # sum_20 = 0.0625 * 5 + 0.09375 * 3 + 0.03125 * 12  # sum of freq without the stop codon
    # pvals = [0.09375 / sum_20] * 3 + [0.0625 / sum_20] * 5 + [0.03125 / sum_20] * 12  # normalize freq for 20 codons
    # pvals = [0.09375 / sum_20] * 3 + [0.0625 / sum_20] * 5 + [0.03125 / sum_20] * 11 + \
    #         [1 - sum([0.09375 / sum_20] * 3 + [0.0625 / sum_20] * 5 + [0.03125 / sum_20] * 11)]
    # # adjust sum to 1 due to numerical issue
    # uniform_pvals = [0.05] * 20
    #
    # encoding_style = 'regular'
    # lambda_val = 1
    # alpha = 0.9
    # beta = 0.1
    #
    # aa_list_2 = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    # pvals_2 = [0.07668660126327106, 0.035596693992742914, 0.02474465797607849, 0.04041795457599785, 0.02319916677865878,
    #            0.1149711060341352, 0.02187206020696143, 0.021972853111140975, 0.030170675984410696, 0.0904280338664158,
    #            0.030069883080231154, 0.017672355866147026, 0.03937642789947588, 0.03156497782556108, 0.1183812659588765,
    #            0.07880325225104153, 0.043290552345114905, 0.08557317564843435, 0.053369842763069476,
    #            0.02183846257223492]
    #
    # original_blosum62 = {}
    #
    # blosum_matrix = np.zeros((20, 20))
    # for i, aa in enumerate(original_blosum62.keys()):
    #     sims = original_blosum62[aa]
    #     for j, s in enumerate(sims):
    #         blosum_matrix[i][j] = s
    # u, V = LA.eig(blosum_matrix)
    # clipped_u = u
    # clipped_u[clipped_u < 0] = 0
    # lamb = np.diag(clipped_u)
    # T = V
    # clip_blosum62 = {}
    # for i, aa in enumerate(original_blosum62.keys()):
    #     clip_blosum62[aa] = np.dot(np.sqrt(lamb), V[i])

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", default= 128, type=int)
    argparser.add_argument("--weight_decay", default=0.00005, type=float, help = "weight penalty")
    argparser.add_argument("--embedding_type", default="one_hot", type=str)
    argparser.add_argument("--test", default=False, type=bool, help = "used for debug")
    argparser.add_argument("--device", default="cpu", type=str)
    argparser.add_argument("--optimizer", default="Adam", type=str)
    args = argparser.parse_args()

    if args.test:
        train_dataset = BinaryDataset(data_path="data/dataset/tmp.tsv")
        val_dataset = BinaryDataset(data_path="data/dataset/tmp.tsv")
    else:
        train_dataset = BinaryDataset(data_path="data/dataset/train.tsv")
        val_dataset = BinaryDataset(data_path="data/dataset/val.tsv")
    samples_weight = train_dataset.comp_weights()
    train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))  # how likely to draw sample from each class


    """  
    python train.py --batch_size 128 --embedding_type embedding --weight_decay 0.0005
    python train.py --batch_size 128 --embedding_type one_hot
    python train.py --batch_size 128 --embedding_type one_hot --test True
    """


    """
    log for terminal
    python train.py --batch_size 128 --embedding_type one_hot --device 0 --weight_decay 0.0005
    python train.py --batch_size 128 --embedding_type embedding  --device 0 --weight_decay 0.0005
    python train.py --batch_size 128 --embedding_type one_hot --device 1 --weight_decay 0.0001   # overfit
    python train.py --batch_size 128 --embedding_type embedding  --device 1 --weight_decay 0.0001 --optimizer SGD
    """

    val_sampler = RandomSampler(val_dataset, replacement= False) # ensure a uniform display of accuracy during validation
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, sampler=val_sampler)
    # device = [0, 1]


    # TODO: make it support device list
    if args.device == "cpu":
        device = args.device
    else:
        device = int(args.device)
    # device = "cpu"
    log_interval = 100
    epoch = 200
    check_point = "./best_model_LinearTwoHead_one_hot.pt"
    check_point = None
    if check_point == None:
        model = LinearTwoHead(args.embedding_type)
    else:
        with open(check_point, "r") as fp:
            model = torch.load(fp)["best_model"]

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train(model, comp_loss, device, train_loader, val_loader, optimizer, epoch, log_interval)



    # def tokenize_en(text):
    #     return [tok.text for tok in spacy_en.tokenizer(text)]
    #
    #
    # src = Field(tokenize=tokenize_en,
    #             init_token='<sos>',
    #             eos_token='<eos>',
    #             lower=True)
    #
    # fields = {'response': ('r', src)}
    #
    # train_data, test_data, validation_data = TabularDataset.splits(
    #     path='FilePath',
    #     train='trainset.json',
    #     test='testset.json',
    #     validation='validationset.json',
    #     format='json',
    #     fields=fields
    # )
