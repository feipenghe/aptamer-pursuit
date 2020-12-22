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
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import argparse


# TODO: generate data
# TODO: cutomize dataset that return embeddings(NO)
# TODO: transformer
# TODO: save both up-to-date model and best model
# TODO: add model name and relevant info in torch.save for easier evaluation (like embedding type,
# TODO: write randomly generated data into files
# TODO: data classes and write into json files
# TODO: scheduler with hyperparameter



def comp_loss(prediction, label, reduction = 'mean'):
    # loss_val = F.binary_cross_entropy(prediction.squeeze(), label.squeeze().float(), reduction = reduction)
    # import pdb
    # pdb.set_trace()
    loss_val = F.binary_cross_entropy(prediction.squeeze(), label.squeeze().float())
    return loss_val

def to_device(mode, device, apt, pep, label):
    if mode == "parallel":
        apt, pep, label = apt.cuda(), pep.cuda(), label.cuda()
        # apt, pep, label = apt.cuda(), pep, label.cuda()
    else:
        apt, pep, label = apt.to(device), pep.to(device), label.to(device)
        # apt, pep, label = apt.to(device), pep, label.to(device)
    return apt, pep, label

def comp_accuracy(mode, model, output, label):
    return get_model(mode, model).accuracy(output, label) * len(label)


def get_model(mode, model):
    """
    Fetch the model in case of parallel
    """
    if mode == "parallel":
        return model.module
    else:
        return model

def train(expr_name, model, comp_loss, pep_embedder, emb_type, train_loader, val_loader, optimizer,  device, start_epoch=0, epoch= 200, log_interval = 100):
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

    def generate_pep_emb(pep, emb_type, pep_embedder):
        """
        Generate different types of embedding for peptides.
        Note that pep_embedder are different for different emb_type.
        """
        if emb_type == None:
            return pep
        elif emb_type == "SeqEmb":
            pep_embedding_batch = torch.tensor(pep_embedder.embed(pep))
            pep_embedding_batch = pep_embedding_batch.permute(1, 0, 2)
            return pep_embedding_batch
        elif emb_type == "TokEmb":
            pep_embedding_batch = torch.from_numpy(np.array([pep_embedder.embed(seq) for seq in pep]))
            # TODO: check dimension

            return pep_embedding_batch
        else:
            print("Wrong embedding type ")
            exit()


    for e in range(epoch):
        model.train()
        tqdm_train_loader = tqdm(train_loader)
        epoch_train_loss = 0.
        count = 0
        for batch_idx, (apt, pep, label) in enumerate(tqdm_train_loader):

            pep = generate_pep_emb(pep, emb_type, pep_embedder)

            apt, pep, label = to_device(mode, device, apt, pep, label)
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
            pep = generate_pep_emb(pep, emb_type, pep_embedder)
            apt, pep, label = to_device(mode, device, apt, pep, label)

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
        avg_val_acc = val_correct * 1.0 / cur_num_labels
        val_acc_l.append(avg_val_acc)
        cur_model_state = get_model(mode, model).state_dict()
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = cur_model_state
            best_epoch = e

        torch.save({"best_epoch": best_epoch,
                    "best_model": best_model_state,
                    "best_val_acc": best_val_acc,
                    "cur_epoch": e,
                    "cur_model":cur_model_state,
                    "emb_type":emb_type
                    },
                   f"./best_model_{get_model(mode, model).name}_{expr_name}.pt",
                   )
        # print("epoch_train_loss_l: ", epoch_train_loss_l)
        # print("epoch_val_loss_l: ", epoch_val_loss_l)
        # epoch_train_loss_l = []  # record average train loss
        # epoch_val_loss_l = []  # record average validation loss

    with open(f"./best_model_{get_model(mode, model).name}_{expr_name}.json", "w") as fp:
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




def main(args):

    if args.embedding_type == "bio_emb":
        encode = False
    else:
        encode = True

    if args.model == "Transformer":
        dataset_type = 2
    else:
        dataset_type = 1

    if args.test:
        train_dataset = BinaryDataset(data_path="data/dataset/tmp.tsv", encode= encode, dataset_type = dataset_type)
        val_dataset = BinaryDataset(data_path="data/dataset/tmp.tsv", encode= encode, dataset_type = dataset_type)
        epoch = 10
        expr_name = "test"
    else:
        train_dataset = BinaryDataset(data_path="data/dataset/train.tsv", encode= encode, dataset_type = dataset_type)
        val_dataset = BinaryDataset(data_path="data/dataset/val.tsv", encode= encode, dataset_type = dataset_type)
        epoch = 50
        expr_name = args.expr_name
    samples_weight = train_dataset.comp_weights()
    train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))  # how likely to draw sample from each class


    """  
    python train.py --batch_size 128 --embedding_type embedding --weight_decay 0.0005
    python train.py --batch_size 128 --embedding_type one_hot
    python train.py --batch_size 128 --embedding_type bio_emb --device 0 --test True
    python train.py --batch_size 32 --embedding_type embedding --device 1 --weight_decay 0.0005 --expr_name apt_dim_15_2
    python train.py --batch_size 32 --embedding_type bio_emb --device 1 --expr_name apt_dim_15_2
    """


    """
    log for terminal
    python train.py --batch_size 512 --embedding_type bio_emb --device 1 --weight_decay 0.0005 --expr_name apt_dim_15 # test larger batch size
    python train.py --batch_size 32 --embedding_type bio_emb --device 1 --weight_decay 0.0005 --expr_name apt_dim_15_2 # test smaller batch size
    # LSTM
    CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 256 --embedding_type bio_emb --device 0 --weight_decay 0.0005 --expr_name first_lstm # weight decay is too high
    """

    val_sampler = RandomSampler(val_dataset, replacement= False) # ensure a uniform display of accuracy during validation
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, sampler=val_sampler)
    # device = [0, 1]



    if args.device == "cpu":
        device = args.device
    elif args.device == "parallel":
        device = [0, 1]
    else:
        device = int(args.device)

    start_epoch = 0 # default
    # device = "cpu"
    log_interval = 100
    # epoch = 200
    check_point = "./best_model_LinearTwoHead_one_hot.pt"
    check_point = None
    if check_point == None:
        if args.model == "Linear":
            model = LinearTwoHead(args.embedding_type)
        elif args.model == "LSTM":
            model = RNNtwoHead(RNN_type="LSTM", embedding_type=args.embedding_type)
        elif args.model == "Transformer":
            model = AptPepTransformer()
        else:
            print("Please provide valid model name")
            exit()
        # model = RNNtwoHead(RNN_type = "LSTM", embedding_type = args.embedding_type)
        # model = LinearTwoHead(args.embedding_type)
        # model = ConvTwoHead(args.embedding_type)
    else:
        with open(check_point, "r") as fp:
            loaded_cp = torch.load(fp)
            model = loaded_cp["best_model"] # TODO: it should load the lastest model
            start_epoch = loaded_cp["best_epoch"]






    # training configuration
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # TODO: embedder is decided by the the model type
    if args.embedding_type == "bio_emb":
        if args.model in ["Linear"]:
            pep_embedder = SeqVecEmbedder()  # this embedder should be used for linear two head
            emb_type = "SeqEmb"
        else:
            pep_embedder = ProtTransBertBFDEmbedder()
            emb_type = "TokEmb"
    else:
        # initialize embedding
        pep_embedder = None
        emb_type = None
    train(expr_name, model , comp_loss, pep_embedder, emb_type, train_loader, val_loader, optimizer, device, start_epoch, epoch, log_interval)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type = str)
    argparser.add_argument("--batch_size", default= 128, type=int)
    argparser.add_argument("--weight_decay", default=0.00005, type=float, help = "weight penalty")
    argparser.add_argument("--embedding_type", default="one_hot", type=str)
    argparser.add_argument("--lr", default=0.00035, type=float)
    argparser.add_argument("--test", default=False, type=bool, help = "used for debug")
    argparser.add_argument("--device", default="cpu", type=str)
    argparser.add_argument("--optimizer", default="Adam", type=str)
    argparser.add_argument("--expr_name", default=None, type=str, help = "can specify the feature of this training")
    argparser.add_argument("--epoch", default=50, type=int)
    argparser.add_argument("--recover", default=False, type=bool)
    argparser.add_argument("--check_point_path", default=None, type=str, help="recover from the checkpoint path if provided ")

    args = argparser.parse_args()

    """
    python train.py --model Linear --batch_size 32 --embedding_type bio_emb --device 1 --expr_name apt_dim_15_2
    python train.py --model ConvNet --batch_size 32 --embedding_type bio_emb --device 1 --expr_name apt_dim_15_2
    python train.py --model LSTM --batch_size 32 --embedding_type bio_emb --device 1 --expr_name apt_dim_15_2
    python train.py --model Transformer --batch_size 512 --embedding_type bio_emb --device parallel --lr 0.00035 --expr_name apt_dim_15_2
    """

    # check argument
    if args.recover == True:
        assert args.check_point_path != None, "must provide check point path"
    main(args)