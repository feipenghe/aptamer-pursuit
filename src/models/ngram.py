# A bigram predicts the likelihood of the current element based on the previous element.
import json
import random
import numpy as np
from sklearn import linear_model, metrics
from sklearn.svm import SVC
from scipy import stats
import random
import re
from progressbar import ProgressBar

# Construct a training dataset:
dataset_file = "../data/ngram_dataset.json"

with open(dataset_file, 'r') as f:
    ngram_dataset = json.load(f)

'''
Generate and evaluate the performance of features to predict whether a protein will bind to a specific allele. 
'''
def classify_affinity(affinity):
    affinity = float(affinity)
    if affinity <= 1.0:
        return 0
    elif affinity > 1.0 and affinity <= 8000:
        return 1
    return 2

def predict_proteins(k, d):
    total_train_acc = []
    total_test_acc = []
    pbar = ProgressBar()
    for allele in pbar(ngram_dataset):
        # These are protein/binding affinity pairs
        proteins = ngram_dataset[allele]

        # Divide them into train/test sets
        num_samples = len(proteins)
        if num_samples < 150:
            continue

        # Randomly select training and test
        random.shuffle(proteins)

        training_set = proteins[:int(num_samples*0.8)]
        testing_set = proteins[int(num_samples*0.8):]

        # Generate features using ngram structure
        training_peptides = [p for (p,b) in training_set]
        train_y = [classify_affinity(b) for (p, b) in training_set]
        testing_peptides = [p for (p,b) in testing_set]

        test_y = [classify_affinity(b) for (p, b) in testing_set]
        # Features --> Quartile
        features = []
        for i in range(d):
            # Find a random sequence in the training set
            seq = random.choice(training_peptides)

            # Find a random subsection of k elements from this sequence
            start = random.randint(0, len(seq) - k)
            quartile_pctg = (start + 1)/float(len(seq))
            if quartile_pctg <= 0.25:
                quartile = 1
            elif quartile_pctg > 0.25 and quartile_pctg <= 0.5:
                quartile = 2
            elif quartile_pctg > 0.5 and quartile_pctg <= 0.75:
                quartile = 3
            else:
                quartile = 4

            feature = seq[start:start+k]

            features.append((feature, quartile))

        # Generate train and test matrices
        train_features = np.zeros((len(training_peptides), d))
        test_features = np.zeros((len(testing_set), d))

        for i in range(len(training_peptides)):
            sequence = training_peptides[i]
            for j in range(len(features)):
                feature, quartile = features[j]
                # If it's in the correct quartile, it's 1
                starts = [m.start() for m in re.finditer(feature, sequence)]
                if len(starts) == 0:
                    train_features[i, j] = 0
                else:
                    for s in starts:
                        # Each s is an index of the beginning of this features
                        # If one of them appears in the correct quartile, then this is 1
                        pctg = (s+1)/len(sequence)
                        if pctg <= 0.25 and quartile == 1:
                            train_features[i, j] = 1
                        elif (pctg > 0.25 and pctg <= 0.5) and quartile == 2:
                            train_features[i, j] = 1
                        elif (pctg > 0.5 and pctg <= 0.75) and quartile == 3:
                            train_features[i, j] = 1
                        elif pctg > 0.75 and quartile == 4:
                            train_features[i, j] = 1

        for i in range(len(testing_peptides)):
            sequence = testing_peptides[i]
            for j in range(len(features)):
                feature, quartile = features[j]
                # If it's in the correct quartile, it's 1
                starts = [m.start() for m in re.finditer(feature, sequence)]
                for s in starts:
                    # Each s is an index of the beginning of this features
                    # If one of them appears in the correct quartile, then this is 1
                    pctg = (s + 1) / len(sequence)
                    if pctg <= 0.25 and quartile == 1:
                        test_features[i, j] = 1
                    elif (pctg > 0.25 and pctg <= 0.5) and quartile == 2:
                        test_features[i, j] = 1
                    elif (pctg > 0.5 and pctg <= 0.75) and quartile == 3:
                        test_features[i, j] = 1
                    elif pctg > 0.75 and quartile == 4:
                        test_features[i, j] = 1
                    else:
                        test_features[i, j] = 0

        # Use a linear model here to calculate the best parameters for the linear regression model
        model = SVC(kernel='linear')
        model.fit(train_features, train_y)

        train_predict = model.predict(train_features)
        gt_train = train_y

        train_acc = metrics.accuracy_score(gt_train, train_predict)

        test_predict = model.predict(test_features)
        gt_test = test_y

        test_acc = metrics.accuracy_score(gt_test, test_predict)

        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)

    print("D: " + str(d) + " K: " + str(k))

    print("Average Train Accuracy: ", np.mean(total_train_acc))
    print("Average Test Accuracy: ", np.mean(total_test_acc))

    return (d, k, np.mean(total_train_acc), np.mean(total_test_acc))

'''
Generate statistics about the proteins that I predicted.
TODO: Understand motifs, right now it just looks at length
@param: predicted_proteins = a list of predicted proteins (for all alleles)
@param: seq_length = the length of peptide sequence to generate. 
'''
def generate_stats(predicted_proteins, seq_length):
    # Shortest Length
    shortest_length = seq_length
    longest_length = 0
    for i in range(len(predicted_proteins)):
        shortest_length = min(shortest_length, len(predicted_proteins[i]))
        longest_length = max(longest_length, len(predicted_proteins[i]))

    print("-----Stats-----")
    print("Shortest Length Peptide: ", shortest_length)
    print("Longest Length Peptide: ", longest_length)


def run_experiments():
    # k --> (d, total_train_mse, total_test_mse)
    experiments = {}
    for k in range(2, 5):
        for d in range(100, 500, 100):
            d, k, total_train_mse, total_test_mse = predict_proteins(k, d)
            if k not in experiments:
                experiments[k] = []
            experiments[k].append((d, total_train_mse, total_test_mse))


    with open("../../experiments/ngram_svm.json", 'w') as f:
        json.dump(experiments, f)


run_experiments()
#predict_proteins(k=2, d=100)







