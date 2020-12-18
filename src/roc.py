from model import *
from tqdm import tqdm
from util import RocDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch

def evaluate(model, device, data_loader):
    model.eval()

    thresholds = [i for i in range(1, 100, 5)]
    precisions = []
    recalls = []
    fprs = []
    tprs = []

    for i, t in enumerate(thresholds):
        t = t*0.01
        print("thresholds: ", t)
        tp, tn, fp, fn = 0, 0, 0, 0
        #test_loader = tqdm(data_loader)
        pos_count = 0
        neg_count = 0

        with torch.no_grad():

            for batch_idx, (apt, pep, label) in enumerate(data_loader):
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
    models_name = ["LinearTwoHead_embedding","ConvTwoHead"]
    legend_name = ["LinearTwoHead","ConvTwoHead"]
    device = "cpu"
    val_dataset = RocDataset(data_path="./test.tsv")
    print(len(val_dataset))
    val_sampler = RandomSampler(val_dataset, replacement= False) 
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    #print(val_dataset[0])
    colors = ["lightcoral", "crimson"]

    for i in range(2):
        model_name = models_name[i]
        if model_name == "LinearTwoHead_embedding":
            model =  LinearTwoHead("embedding")
        else:
            model =  ConvTwoHead("embedding")
        check_point = "./best_model_"+ model_name +".pt"
        model.load_state_dict(torch.load(check_point, map_location=torch.device('cpu'))["best_model"])
        # Plot
        fprs, tprs, recalls, precisions = evaluate(model, device, val_loader)

        plt.plot(fprs, tprs, c=colors[i], label=legend_name[i])

    plt.plot([0,1], [0,1],'--', color = 'k',linewidth=0.5)
    
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

        

