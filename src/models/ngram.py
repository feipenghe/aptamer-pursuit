# A bigram predicts the likelihood of the current element based on the previous element.
import json
import nltk
import random
import numpy as np
import math

# Construct a training dataset:
dataset_file = "../data/ngram_dataset.json"

with open(dataset_file, 'r') as f:
    ngram_dataset = json.load(f)

# k = k-gram value, d = number of features
k = 4
d = 200

'''
Generate predicted protein for each allele in the dataset separately
@param: n = the number of characters to use in predictions. (i.e. bigram means n=2)
@param: seq_length = the length of peptide sequence to generate. 
'''
def predict_proteins(n=3, seq_length=8):
    sum_of_losses = 0.0
    for allele in ngram_dataset:
        # Sentences are all the peptides that bound to this allele
        sentences = []
        # These are protein/binding affinity pairs
        proteins = ngram_dataset[allele]

        # Divide them into train/test sets
        num_samples = len(proteins)
        if num_samples < 20:
            continue
        training_set = proteins[:int(num_samples*0.8)]
        testing_set = proteins[int(num_samples*0.8):]

        # Generate features using ngram structure
        training_peptides = [p for (p,b) in training_set]
        train_y = [float(b) for (p, b) in training_set]
        testing_peptides = [p for (p,b) in testing_set]
        test_y = [float(b) for (p, b) in testing_set]
        features = []
        for i in range(d):
            # Find a random sequence in the training set
            seq = random.choice(training_peptides)

            # Find a random subsection of k elements from this sequence
            start = random.randint(0, len(seq) - k)
            feature = seq[start:start+k]

            # Add it to the list of features here
            features.append(feature)

        # Generate train and test matrices
        train_features = np.zeros((len(training_peptides), d))
        test_features = np.zeros((len(testing_set), d))

        train_ones = 0
        for i in range(len(training_peptides)):
            sequence = training_peptides[i]
            for j in range(len(features)):
                feature = features[j]
                train_features[i, j] = 1 if str(feature) in str(sequence) else 0
                if train_features[i, j] == 1:
                    train_ones += 1

        for i in range(len(testing_peptides)):
            sequence = testing_peptides[i]
            for j in range(len(features)):
                feature = features[j]
                test_features[i, j] = 1 if feature in sequence else 0

        # Use a linear model here to calculate wx+b for the training set
        # Calculate w, with a gradient based method
        w = np.random.randn(d)
        lr = 0.01
        for i in range(20):
            derivative = 0
            for i in range(len(training_set)):
                row = train_features[i]
                binding_affinity = float(train_y[i])
                mult = binding_affinity - np.matmul(np.transpose(row), w, dtype=float)

                derivative += row * mult
            derivative *= -1

            # update the weight
            w -= lr * derivative


        # TODO: skipping the regularizer for now
        b = random.random()


        train_loss = np.min(np.power(np.matmul(w, np.transpose(train_features)) - np.log(train_y), 2))

        # Calculate the test loss
        #TODO: this doesn't seem like the best loss.
        test_loss = np.min(np.power(np.matmul(w, np.transpose(test_features)) - np.log(test_y), 2))

        sum_of_losses += test_loss

        print("Allele: " + str(allele) + ", Train Loss: " + str(train_loss) + ", Test Loss: " + str(test_loss) + " Num Samples: " + str(num_samples))

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


predicted_proteins = predict_proteins(n=1, seq_length=8)
#generate_stats(predicted_proteins, seq_length=8)





