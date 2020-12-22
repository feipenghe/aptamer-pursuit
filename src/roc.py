from model import *
from tqdm import tqdm
from util import RocDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch
import os
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import numpy as np


def evaluate(model, device, data_loader, bio_embedding):
    def generate_pep_emb(pep, emb_type, pep_embedder):
        """
        Generate different types of embedding for peptides.
        Note that pep_embedder are different for different emb_type.
        """

        if emb_type == None or emb_type == "embedding":
            return pep
        elif emb_type == "SeqEmb" or emb_type == "bio_emb":
            pep_embedding_batch = torch.tensor(pep_embedder.embed(pep))
            pep_embedding_batch = pep_embedding_batch.permute(1, 0, 2)
            return pep_embedding_batch
        elif emb_type == "TokEmb":
            pep_embedding_batch = torch.from_numpy(np.array([pep_embedder.embed(seq) for seq in pep]))
            # TODO: check dimension
            return pep_embedding_batch
        else:
            print("Wrong embedding type ", emb_type)
            exit()



    model = model.to(device)
    model.eval()

    thresholds = [i for i in range(1, 100, 5)]
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    pep_embedder = SeqVecEmbedder()
    for i, t in enumerate(thresholds):
        t = t*0.01
        print("thresholds: ", t)
        tp, tn, fp, fn = 0, 0, 0, 0
        #test_loader = tqdm(data_loader)
        pos_count = 0
        neg_count = 0

        with torch.no_grad():

            for batch_idx, (apt, pep, label) in enumerate(data_loader):
                pep = generate_pep_emb(pep, emb_type, pep_embedder)
                # import pdb
                # pdb.set_trace()
                # if bio_embedding:
                #     pep = [pep[0]]
                #     pep = torch.tensor(pep_embedder.embed(pep))
                #     pep = pep.permute(1, 0, 2)


                apt, pep, label = apt.to(device), pep.to(device), label.to(device)

                output = model(apt, pep)

                if int(label) == 0:
                    neg_count += 1
                    if output > t:
                        fp += 1
                    else:
                        tn += 1
                else:
                    pos_count += 1
                    if output > t:
                        tp += 1
                    else:
                        fn += 1

        try:
            precision = tp/float(tp + fp)
        except:
            precision = tp/1
        try:
            recall = tp/float(tp + fn)
        except:
            recall = tp/1

        #print("Precision: ", precision)
        #print("Recall: ", recall)
        precisions.append(precision)
        recalls.append(recall)

        # ROC stats
        fpr = fp/max(float(fp+tn),1.)
        tpr = tp/max(float(tp+fn),1.)
        fprs.append(fpr)
        tprs.append(tpr)

    roc_curve = zip(fprs, tprs)
    pr_curve = zip(recalls, precisions)
    
    roc_curve = sorted(roc_curve, key=lambda x: x[0])
    pr_curve = sorted(pr_curve, key=lambda x: x[0])
    
    roc_curve = list(zip(*roc_curve))
    pr_curve = list(zip(*pr_curve))
    return list(roc_curve[0]), list(roc_curve[1]), list(pr_curve[0]), list(pr_curve[1])


if __name__ == '__main__':
    # models_name = ["LinearTwoHead_embedding","ConvTwoHead"]
    # legend_name = ["LinearTwoHead","ConvTwoHead"]
    # device = "cpu"
    #
    #
    # #print(val_dataset[0])
    # colors = ["lightcoral", "crimson"]
    # # best_model_LSTM_bio_emb_apt_dim_16_bs_256.pt
    # check_point = "best_model_LSTM_bio_emb_apt_dim_16_bs_256.pt"
    #
    # model = RNNtwoHead(RNN_type = "LSTM", embedding_type = "bio_emb")
    #
    #
    # loaded_cp = torch.load(check_point, map_location=torch.device('cpu'))
    # model.load_state_dict(loaded_cp["best_model"])
    check_points = [cp for cp in os.listdir() if cp.endswith(".pt")]


    device = 0
    for cp in check_points:
        model_name = cp.replace("best_model_", "")
        if "bio" in cp:
            emb_type = "bio_emb"
        elif "one_hot" in cp:
            emb_type = "one_hot"
        else: # some names don't have embedding
            emb_type = "embedding"




        print("loading check point ", model_name,  " embedding type is ", emb_type)

        encode = True
        if "Linear" in cp:
            model = LinearTwoHead(embedding_type = emb_type)
            model_name = "Linear"
            dataset_type = 0
        elif "LSTM" in cp:
            model = RNNtwoHead(RNN_type = "LSTM", embedding_type = emb_type)
            model_name = "LSTM"
            dataset_type = 0
            encode = False
        elif "Conv" in cp:
            model = ConvTwoHead(embedding_type = emb_type)
            model_name = "ConvNet"
            dataset_type = 0
        elif "Transformer" in cp:
            model = AptPepTransformer()
            dataset_type = 2

        test_dataset = RocDataset(data_path="data/dataset/test.tsv", encode=encode, dataset_type = dataset_type)
        test_sampler = RandomSampler(test_dataset, replacement=False)
        test_loader = DataLoader(test_dataset, batch_size=1)

        loaded_cp = torch.load(cp)
        print("loading ", model_name, "       the best val acc is  ", loaded_cp["best_val_acc"])
        model.load_state_dict(loaded_cp["best_model"])

        # if emb_type == "bio_emb":
        # Plot
        fprs, tprs, recalls, precisions = evaluate(model, device, test_loader, bio_embedding = emb_type == "bio_emb")

        plt.plot(fprs, tprs, label=model_name)

    plt.plot([0,1], [0,1],'--', color = 'k',linewidth=0.5)
    
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

        

