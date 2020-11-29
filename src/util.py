import os, sys
import numpy as np
import json
import random
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA


class AptPepDataset(Dataset):
    def __init__(self, pos_data_file, neg_data_file, batch_size):
        super(AptPepDataset, self).__init__()
        self.batch_size = batch_size

        with open(pos_data_file, "r") as pos_fp:
            pos_json_f = json.load(pos_fp)
            # gives label 1 and
            pos_json_f.items
            print(list(pos_json_f.keys())[:3])
            exit()

        with open(neg_data_file, "r") as neg_fp:
            json.load(neg_fp)


import re
import os
import itertools
import csv

def prepare_data(pos_data_dir, neg_data_dir, data_dir, train_ratio = 0.7, val_ratio = 0.1):
    """
    This method is for seldom splitting train/val/test.
    """
    def prepare_dir():
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        else:
            # remove the old split
            for old_f in os.listdir(data_dir):
                os.remove(f"{data_dir}/{old_f}")

    def read_dict():
        pos_d = dict()
        for pos_f in os.listdir(pos_data_dir):
            with open(f"{pos_data_dir}/{pos_f}", "r") as fp:

                pos_d.update(json.load(fp))

        neg_d = dict()
        for neg_f in os.listdir(neg_data_dir):
            if neg_f.endswith("json"):
                with open(f"{neg_data_dir}/{neg_f}", "r") as fp:
                    j = json.load(fp)
                    neg_d.update(j)
        return pos_d, neg_d


    def write_dict(data_dir):
        """
        Split and write data into tsv file.

        # islice('ABCDEFG', 2) --> A B
        # islice('ABCDEFG', 2, 4) --> C D
        # islice('ABCDEFG', 2, None) --> C D E F G
        # islice('ABCDEFG', 0, None, 2) --> A C E G
        """
        # positive dataset
        print(list(pos_d.items())[:2])
        pos_train_d = dict(itertools.islice(pos_d.items(), 0, n_pos_train))
        pos_val_d = dict(itertools.islice(pos_d.items(), n_pos_train+1, n_pos_train + n_pos_val))
        pos_test_d = dict(itertools.islice(pos_d.items(), n_pos_train + n_pos_val+1, None))
        # negative dataset
        neg_train_d = dict(itertools.islice(neg_d.items(), 0, n_neg_train))
        neg_val_d = dict(itertools.islice(neg_d.items(), n_neg_train + 1, n_neg_train + n_neg_val))
        neg_test_d = dict(itertools.islice(neg_d.items(), n_neg_train + n_pos_val + 1, None))

        def write_tsv(data_dir, pos_d, neg_d, type="train"):
            with open(f"{data_dir}/{type}.tsv", "w") as fp:
                tsv_writer = csv.writer(fp, delimiter="\t")
                tsv_writer.writerow(["aptamer", "peptide", "label"])
                for key, val in pos_d.items():
                    if len(val) > 1:
                        for v in val:
                            tsv_writer.writerow([key, v, 1])
                    else:
                        tsv_writer.writerow([key, val[0], 1])
                for key, val in neg_d.items():
                    if len(val) > 1:
                        for v in val:
                            tsv_writer.writerow([key, v, 0])
                    else:
                        tsv_writer.writerow([key, val[0], 0])


        write_tsv(data_dir, pos_train_d, neg_train_d, type="train")
        write_tsv(data_dir, pos_val_d, neg_val_d, type="val")
        write_tsv(data_dir, pos_test_d, neg_test_d, type="test")

    prepare_dir()
    pos_d, neg_d = read_dict()

    # compute slicing
    n_pos_train = (int)(len(pos_d) * train_ratio)
    n_pos_val = (int)(len(pos_d) * val_ratio)

    n_neg_train =(int)(len(neg_d) * train_ratio)
    n_neg_val = (int)(len(neg_d) * val_ratio)


    write_dict(data_dir)

    # na_list = ['A', 'C', 'G', 'T']  # nucleic acids
    # aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W',
    #            'Y']  # amino acids
class BinaryDataset(Dataset):
    def __init__(self, data_path,  dataset_type = 0):

        '''
        :param sentence_data_path:
        :param token_level:
        :param unk_cutoff:
        :param tokenizer_path:
        :param tokenizer_training_path:  Both used for train the tokenizer and the model
        '''
        super().__init__()

        aptamers = []
        peptides = []
        labels = []


        print("loading corruption dataset")
        neg_count = 0
        pos_count = 0
        # sentences
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # Ignore header
            for row in reader:
                # Each row contains a sentence and label (either 0 or 1)
                aptamer, peptide, label = row
                aptamers.append(aptamer)
                peptides.append(peptide)
                labels.append([int(label)])
                if int(label) == 0:
                    neg_count += 1
                else:
                    pos_count += 1
        self.pos_count = pos_count
        self.neg_count = neg_count


        na_list = ['A', 'C', 'G', 'T']  # nucleic acids
        aa_list = ['R', 'L', 'S', 'A', 'G', 'P', 'T', 'V', 'N', 'D', 'C', 'Q', 'E', 'H', 'I', 'K', 'M', 'F', 'W',
                   'Y']  # amino acids

        # sample_weights
        # how to access sample weights

        self.aptamer_decode = dict(enumerate(na_list))
        self.peptide_decode = dict(enumerate(aa_list))
        self.aptamer_encode = dict({(v, k) for k,v in self.aptamer_decode.items()})
        self.peptide_encode = dict({(v, k) for k,v in self.peptide_decode.items()})

        self.aptamers =  [self.encode_aptamer(a) for a in aptamers ]
        self.peptides  = [self.encode_peptide(p) for p in peptides]
        self.dataset_type =  dataset_type


        self.labels =  torch.tensor(labels)



    def __getitem__(self, index):
        if self.dataset_type == 0:
            return self.aptamers[index], self.peptides[index], self.labels[index]
        elif self.dataset_type == 1:
            return self.sentences[index] + self.peptides[index], self.labels[index]
        else:
            print("get item error ")
            exit()
    def __len__(self):
        return len(self.sentences)

    def comp_weights(self):
        """
        Compute weights based on the data class distribution.
        Used for weighted sampling.

        Example usage:
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

        """
        class_sample_count = np.array([len(np.where(self.labels == t)[0]) for t in np.unique(self.labels)])
        print("class sample count: ", class_sample_count)
        weight = 1.0/class_sample_count
        samples_weight = np.array([weight[t] for t in self.labels] )
        print("sample weight: ", samples_weight)
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight.double()

    def encode_aptamer(self, aptamer):
        indices = []
        for a in aptamer:
            indices.append(self.aptamer_encode[a])
        return torch.tensor(indices)
    def encode_peptide(self, peptide):
        indices = []
        for p in peptide:
            indices.append(self.peptide_encode[p])
        return torch.tensor(indices)

    def decode_aptamer(self, indices):
        tokens = []
        for i in indices:
            tokens.append(self.aptamer_decode[i])
        return tokens

    def decode_peptide(self, indices):
        # tokens = []
        # for i in indices:
        #     tokens.append(self.peptide_decode[i])
        return [self.peptide_decode[i] for i in indices]


class AUCDataset(Dataset):
    def __init__(self, filepath, negfilepath=None):
        def construct_dataset(filepath, negfilepath=None):
            with open(filepath, 'r') as f:
                aptamer_data = json.load(f)
            bio_ds = []
            neg_ds = []
            gen_ds = []
            for aptamer in aptamer_data:
                peptides = aptamer_data[aptamer]
                for peptide in peptides:
                    bio_ds.append((aptamer, peptide, 1))
                    gen_ds.append((get_x(), get_y('NNK'), 0))
            with open(negfilepath, 'r') as f:
                neg_data = json.load(f)
            for aptamer in neg_data:
                peptides = neg_data[aptamer]
                for peptide in peptides:
                    neg_ds.append((aptamer, peptide, 0))
            bio_ds = list(set(bio_ds))  # removed duplicates, random order
            gen_ds = list(set(gen_ds))  # removed duplicates, random order
            neg_ds = list(set(neg_ds))

            return bio_ds, neg_ds, gen_ds

        # Sample x from P_X (assume apatamers follow uniform)
        def get_x():
            x_idx = np.random.randint(0, 4, 40)
            x = ""
            for i in x_idx:
                x += na_list[i]
            return x

        # Sample y from P_y (assume peptides follow NNK)
        def get_y(distribution='NNK'):
            if distribution == 'NNK':
                y_idx = np.random.choice(20, 7, p=pvals)
                lst = aa_list
            elif distribution == 'uniform':
                y_idx = np.random.choice(20, 7, p=uniform_pvals)
                lst = aa_list
            elif distribution == 'new_nnk':
                y_idx = np.random.choice(20, 7, p=pvals_2)
                lst = aa_list_2
            y = "M"
            for i in y_idx:
                y += lst[i]
            return y

        self.bio_ds, self.neg_ds, self.gen_ds = construct_dataset(filepath, negfilepath)

    def __len__(self):
        return min(len(self.bio_ds), len(self.neg_ds))

    def __getitem__(self, idx):
        return (self.bio_ds[idx], self.neg_ds[idx], self.gen_ds[idx])


if __name__ == '__main__':

    from os import path

    pos_data_dir = "./data/pos_datasets"
    # pos_filepath = path.join(pos_folder_path, "experimental_replicate_4.json")

    neg_data_dir = "./data/neg_datasets"
    # neg_filepath = path.join(neg_folder_path, "neg1_all_pairs_noArgi_noHis.json")

    data_dir = "./data/dataset"

    out_filepath = path.join(  pos_data_dir,"data.json")
    # prepare_data(pos_data_dir, neg_data_dir, data_dir, train_ratio = 0.7, val_ratio = 0.1)

    train_dataset = BinaryDataset(data_path="data/dataset/val.tsv")
    samples_weight = train_dataset.comp_weights()
    print("sample weigtht: " , samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))  # how likely to draw sample from each class
    train_loader = DataLoader(train_dataset, batch_size=512, num_workers=32, sampler=sampler)
    for i, (a, p, target) in enumerate(train_loader):
        print("batch index {}, 0/1: {}/{}".format(i,len(np.where(target.numpy() == 0)[0]),len(np.where(target.numpy() == 1)[0])))
